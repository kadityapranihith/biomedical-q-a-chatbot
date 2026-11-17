"""
evaluate_generation_all_methods.py
===================================
Generation Evaluation across 4 Retrieval Paradigms:
1. Closed-Book (Zero-shot / No Retrieval)
2. BM25 Only (Sparse)
3. MedCPT Only (Dense)
4. Hybrid (BM25 + MedCPT)

Metrics computed per method:
- Decision Accuracy
- Decision Macro-F1 (and per-class F1 for yes, no, maybe)
- Confusion Matrix
- ROUGE-1, ROUGE-2, ROUGE-L (against PubMedQA LONG_ANSWER)
- BERTScore F1, Precision, Recall (against PubMedQA LONG_ANSWER)
- Average Retrieval & Generation Latency (ms)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from rouge_score import rouge_scorer
import bert_score


# ============================================================
# PATHS & SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"

MEDCPT_INDEX_PATH = BASE_DIR / "data" / "medcpt_index" / "medcpt_abstract.index"
MEDCPT_PMIDS_PATH = BASE_DIR / "data" / "medcpt_index" / "pmids.json"

RESULTS_DIR = BASE_DIR / "evaluation" / "generation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATE_K = 100
RAG_TOP_K = 5
MEDCPT_MAX_LENGTH = 64
BM25_WEIGHT = 0.5
MEDCPT_WEIGHT = 0.5


# ============================================================
# API KEYS POOL & LLM ROTATION
# ============================================================

def get_groq_keys():
    keys = [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY2"),
        os.getenv("GROQ_API_KEY3")
    ]
    valid_keys = [k.strip() for k in keys if k and k.strip()]
    if not valid_keys:
        raise RuntimeError("No GROQ_API_KEY found in .env")
    return valid_keys


# ============================================================
# DATA LOADERS
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_corpus():
    print(f"\nLoading corpus from: {CORPUS_PATH}")
    documents = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            abstract = record.get("abstract", "").strip()
            if not abstract:
                continue
            documents.append({
                "pmid": str(record["pmid"]),
                "title": record.get("title", ""),
                "abstract": abstract
            })
    print(f"Loaded {len(documents):,} corpus abstracts.")
    return documents


def load_test_questions(max_questions=100):
    print(f"\nLoading test questions (max={max_questions})...")
    with open(PQA_PATH, "r", encoding="utf-8") as f:
        pqa_data = json.load(f)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_ground_truth = json.load(f)

    questions = []
    for pmid, gold_label in test_ground_truth.items():
        pmid = str(pmid)
        if pmid not in pqa_data:
            continue

        record = pqa_data[pmid]
        question = record.get("QUESTION", "").strip()
        long_answer = record.get("LONG_ANSWER", "").strip()
        if not question:
            continue

        questions.append({
            "pmid": pmid,
            "question": question,
            "gold_label": str(gold_label).lower().strip(),
            "gold_explanation": long_answer
        })

    if max_questions is not None and max_questions > 0:
        questions = questions[:max_questions]

    print(f"Prepared {len(questions)} test questions.")
    return questions


# ============================================================
# RETRIEVAL FUNCTIONS
# ============================================================

def build_bm25_index(documents):
    print("\nBuilding BM25 index on abstracts...")
    tokenized = [tokenize(doc["abstract"]) for doc in documents]
    return BM25Okapi(tokenized)


def load_medcpt_components():
    print("\nLoading MedCPT index and query model...")
    if not MEDCPT_INDEX_PATH.exists() or not MEDCPT_PMIDS_PATH.exists():
        raise FileNotFoundError(f"MedCPT index files not found in {MEDCPT_INDEX_PATH.parent}")

    index = faiss.read_index(str(MEDCPT_INDEX_PATH))
    with open(MEDCPT_PMIDS_PATH, "r", encoding="utf-8") as f:
        pmids = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
    model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL).to(DEVICE)
    model.eval()

    return index, pmids, tokenizer, model


def encode_query(query, tokenizer, model):
    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=MEDCPT_MAX_LENGTH,
        return_tensors="pt"
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
        emb = out.last_hidden_state[:, 0, :]
    return emb.cpu().numpy().astype(np.float32)


def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def retrieve_bm25_only(query, bm25, documents, docs_by_pmid, top_k=RAG_TOP_K):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        doc = documents[int(idx)]
        results.append({
            "rank": rank,
            "pmid": doc["pmid"],
            "title": doc["title"],
            "abstract": doc["abstract"],
            "score": float(scores[int(idx)])
        })
    return results


def retrieve_medcpt_only(query, medcpt_index, medcpt_pmids, tokenizer, model, docs_by_pmid, top_k=RAG_TOP_K):
    q_emb = encode_query(query, tokenizer, model)
    scores, indices = medcpt_index.search(q_emb, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if int(idx) < 0:
            continue
        pmid = str(medcpt_pmids[int(idx)])
        doc = docs_by_pmid.get(pmid)
        if doc is None:
            continue
        results.append({
            "rank": rank,
            "pmid": pmid,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "score": float(score)
        })
    return results


def retrieve_hybrid(query, bm25, documents, medcpt_index, medcpt_pmids, tokenizer, model, docs_by_pmid, top_k=RAG_TOP_K):
    # BM25 Top 100
    bm25_scores_all = bm25.get_scores(tokenize(query))
    bm25_top_idx = np.argsort(bm25_scores_all)[::-1][:CANDIDATE_K]
    bm25_pmids = [documents[int(i)]["pmid"] for i in bm25_top_idx]
    bm25_raw = [float(bm25_scores_all[int(i)]) for i in bm25_top_idx]

    # MedCPT Top 100
    q_emb = encode_query(query, tokenizer, model)
    med_scores, med_indices = medcpt_index.search(q_emb, CANDIDATE_K)
    med_pmids, med_raw = [], []
    for s, i in zip(med_scores[0], med_indices[0]):
        if int(i) < 0:
            continue
        med_pmids.append(str(medcpt_pmids[int(i)]))
        med_raw.append(float(s))

    # Normalize
    bm25_norm = min_max_normalize(bm25_raw)
    med_norm = min_max_normalize(med_raw)

    fused_scores = {}
    for pmid, score in zip(bm25_pmids, bm25_norm):
        fused_scores[pmid] = fused_scores.get(pmid, 0.0) + BM25_WEIGHT * float(score)
    for pmid, score in zip(med_pmids, med_norm):
        fused_scores[pmid] = fused_scores.get(pmid, 0.0) + MEDCPT_WEIGHT * float(score)

    ranked_pmids = sorted(fused_scores.keys(), key=lambda p: fused_scores[p], reverse=True)[:top_k]
    results = []
    for rank, pmid in enumerate(ranked_pmids, start=1):
        doc = docs_by_pmid.get(pmid)
        if doc is None:
            continue
        results.append({
            "rank": rank,
            "pmid": pmid,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "score": float(fused_scores[pmid])
        })
    return results


# ============================================================
# PROMPTS & CONTEXT BUILDER
# ============================================================

CLOSED_BOOK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a biomedical question-answering system.
Answer the biomedical question based on your clinical and scientific knowledge.
Determine whether the answer to the question is YES, NO, or MAYBE.

Return exactly this format:

Answer: YES
Explanation: <brief evidence-based explanation>

OR

Answer: NO
Explanation: <brief evidence-based explanation>

OR

Answer: MAYBE
Explanation: <brief evidence-based explanation>

If there is insufficient medical consensus or conflicting evidence to distinguish YES from NO, answer MAYBE."""
    ),
    (
        "human",
        "Question:\n\n{question}"
    )
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a biomedical question-answering system.
Use ONLY the supplied PubMed abstracts as evidence.
Do not use outside knowledge.
Do not use web search.

Determine whether the answer to the question is YES, NO, or MAYBE.

Return exactly this format:

Answer: YES
Explanation: <brief evidence-based explanation>

OR

Answer: NO
Explanation: <brief evidence-based explanation>

OR

Answer: MAYBE
Explanation: <brief evidence-based explanation>

If the supplied evidence is insufficient to distinguish YES from NO, answer MAYBE.
Do not invent facts that are not supported by the supplied abstracts."""
    ),
    (
        "human",
        """Question:

{question}

PubMed Evidence:

{context}"""
    )
])


def format_context(retrieved_docs):
    sections = []
    for i, doc in enumerate(retrieved_docs, start=1):
        sections.append(
            f"[Evidence {i}]\n"
            f"Title: {doc['title']}\n"
            f"PMID: {doc['pmid']}\n"
            f"Abstract: {doc['abstract']}"
        )
    return "\n\n".join(sections)


def parse_answer(text):
    if not text:
        return None, ""

    pred_label = None
    match = re.search(r"Answer\s*:\s*(YES|NO|MAYBE)\b", text, flags=re.IGNORECASE)
    if match:
        pred_label = match.group(1).lower()
    else:
        stripped = text.strip().upper()
        if stripped in {"YES", "NO", "MAYBE"}:
            pred_label = stripped.lower()
        elif "YES" in stripped and "NO" not in stripped and "MAYBE" not in stripped:
            pred_label = "yes"
        elif "NO" in stripped and "YES" not in stripped and "MAYBE" not in stripped:
            pred_label = "no"
        elif "MAYBE" in stripped:
            pred_label = "maybe"

    pred_explanation = ""
    exp_match = re.search(r"Explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if exp_match:
        pred_explanation = exp_match.group(1).strip()
    else:
        # If model did not output Explanation:, use text minus Answer line
        pred_explanation = re.sub(r"Answer\s*:\s*(YES|NO|MAYBE)", "", text, flags=re.IGNORECASE).strip()

    return pred_label, pred_explanation


# ============================================================
# LLM INVOCATION WITH RATE LIMIT HANDLING & KEY ROTATION
# ============================================================

def invoke_llm_with_retry(prompt_val, keys, key_idx_ref, model_name="openai/gpt-oss-120b", max_retries=6):
    backoffs = [10, 20, 30, 45, 60, 60]
    for attempt in range(max_retries):
        current_key = keys[key_idx_ref[0] % len(keys)]
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=512,
            api_key=current_key
        )
        try:
            start_t = time.perf_counter()
            resp = llm.invoke(prompt_val)
            lat_ms = (time.perf_counter() - start_t) * 1000
            return resp.content, lat_ms, None
        except Exception as e:
            err_str = str(e)
            print(f"\n[API Notice attempt {attempt+1} on key #{key_idx_ref[0] % len(keys)}]: {err_str[:80]}...")
            key_idx_ref[0] += 1
            sleep_sec = backoffs[min(attempt, len(backoffs) - 1)]
            print(f"Waiting {sleep_sec}s before next attempt...")
            time.sleep(sleep_sec)
    return "", 0.0, "Max retries exceeded"


# ============================================================
# METRICS COMPUTATION (Accuracy, Macro-F1, ROUGE, BERTScore)
# ============================================================

def compute_all_metrics(detailed_results):
    valid_results = [r for r in detailed_results if r.get("predicted_label") is not None]
    if not valid_results:
        return {}

    y_true = [r["gold_label"] for r in valid_results]
    y_pred = [r["predicted_label"] for r in valid_results]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=["yes", "no", "maybe"], average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=["yes", "no", "maybe"], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["yes", "no", "maybe"]).tolist()

    preds_text = [r.get("predicted_explanation", "").strip() or "none" for r in valid_results]
    refs_text = [r.get("gold_explanation", "").strip() or "none" for r in valid_results]

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_list, r2_list, rl_list = [], [], []
    for ref, pred in zip(refs_text, preds_text):
        s = scorer.score(ref, pred)
        r1_list.append(s['rouge1'].fmeasure)
        r2_list.append(s['rouge2'].fmeasure)
        rl_list.append(s['rougeL'].fmeasure)

    rouge1 = float(np.mean(r1_list)) if r1_list else 0.0
    rouge2 = float(np.mean(r2_list)) if r2_list else 0.0
    rougeL = float(np.mean(rl_list)) if rl_list else 0.0

    # BERTScore
    print("\nComputing BERTScore across explanations...")
    try:
        P, R, F1 = bert_score.score(preds_text, refs_text, lang="en", verbose=False)
        bertscore_f1 = float(F1.mean().item())
        bertscore_p = float(P.mean().item())
        bertscore_r = float(R.mean().item())
    except Exception as exc:
        print(f"Warning: BERTScore computation failed: {exc}")
        bertscore_f1, bertscore_p, bertscore_r = 0.0, 0.0, 0.0

    gen_lats = [r["generation_latency_ms"] for r in detailed_results if "generation_latency_ms" in r and r["generation_latency_ms"] > 0]
    ret_lats = [r["retrieval_latency_ms"] for r in detailed_results if "retrieval_latency_ms" in r and r["retrieval_latency_ms"] > 0]

    avg_gen_lat = float(np.mean(gen_lats)) if gen_lats else 0.0
    avg_ret_lat = float(np.mean(ret_lats)) if ret_lats else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_per_class": {
            "yes": float(per_class[0]),
            "no": float(per_class[1]),
            "maybe": float(per_class[2])
        },
        "confusion_matrix": cm,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bertscore_f1": bertscore_f1,
        "bertscore_precision": bertscore_p,
        "bertscore_recall": bertscore_r,
        "avg_generation_latency_ms": avg_gen_lat,
        "avg_retrieval_latency_ms": avg_ret_lat,
        "avg_total_latency_ms": avg_gen_lat + avg_ret_lat,
        "evaluated": len(valid_results),
        "total": len(detailed_results)
    }


# ============================================================
# RUN EVALUATION FOR A SINGLE MODE
# ============================================================

def run_evaluation_for_mode(
    mode,
    questions,
    keys,
    model_name="openai/gpt-oss-120b",
    delay_sec=3.0,
    bm25=None,
    documents=None,
    docs_by_pmid=None,
    medcpt_index=None,
    medcpt_pmids=None,
    medcpt_tok=None,
    medcpt_model=None
):
    print("\n" + "=" * 65)
    print(f"STARTING GENERATION EVALUATION: [{mode.upper()}]")
    print("=" * 65)

    out_file = RESULTS_DIR / f"{mode}_results.json"
    checkpoint_file = RESULTS_DIR / f"{mode}_checkpoint.json"

    detailed_results = []
    completed_pmids = set()

    # Load checkpoint if exists
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
                # Keep only successfully evaluated items
                detailed_results = [r for r in saved_results if r.get("predicted_label") is not None]
                completed_pmids = {r["pmid"] for r in detailed_results}
            print(f"Resuming from checkpoint: {len(detailed_results)} questions successfully completed.")
        except Exception:
            pass

    key_idx_ref = [0]

    for item in tqdm(questions, desc=f"Evaluating {mode}"):
        pmid = item["pmid"]
        if pmid in completed_pmids:
            continue

        q_text = item["question"]
        gold_label = item["gold_label"]
        gold_exp = item["gold_explanation"]

        ret_latency = 0.0
        retrieved_docs = []

        # 1. Retrieval
        if mode == "closed_book":
            prompt_input = CLOSED_BOOK_PROMPT.format_messages(question=q_text)
        else:
            t0 = time.perf_counter()
            if mode == "bm25":
                retrieved_docs = retrieve_bm25_only(q_text, bm25, documents, docs_by_pmid)
            elif mode == "medcpt":
                retrieved_docs = retrieve_medcpt_only(q_text, medcpt_index, medcpt_pmids, medcpt_tok, medcpt_model, docs_by_pmid)
            elif mode == "hybrid":
                retrieved_docs = retrieve_hybrid(q_text, bm25, documents, medcpt_index, medcpt_pmids, medcpt_tok, medcpt_model, docs_by_pmid)
            ret_latency = (time.perf_counter() - t0) * 1000

            ctx_text = format_context(retrieved_docs)
            prompt_input = RAG_PROMPT.format_messages(question=q_text, context=ctx_text)

        # 2. Generation with LLM
        gen_text, gen_latency, err = invoke_llm_with_retry(
            prompt_input, keys, key_idx_ref, model_name=model_name
        )

        pred_label, pred_exp = parse_answer(gen_text)
        is_correct = (pred_label == gold_label) if pred_label is not None else False

        result_entry = {
            "pmid": pmid,
            "question": q_text,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "is_correct": is_correct,
            "gold_explanation": gold_exp,
            "predicted_explanation": pred_exp,
            "raw_generation": gen_text,
            "retrieval_latency_ms": ret_latency,
            "generation_latency_ms": gen_latency,
            "error": err,
            "retrieved_pmids": [d["pmid"] for d in retrieved_docs]
        }

        detailed_results.append(result_entry)
        completed_pmids.add(pmid)

        # Save checkpoint incrementally
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2)

        if delay_sec > 0:
            time.sleep(delay_sec)

    # Compute final metrics for this mode
    metrics = compute_all_metrics(detailed_results)

    final_payload = {
        "mode": mode,
        "model": model_name,
        "sample_size": len(detailed_results),
        "metrics": metrics,
        "results": detailed_results
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nCompleted {mode.upper()} evaluation:")
    print(f"  Accuracy:      {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro-F1:      {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  ROUGE-L:       {metrics.get('rougeL', 0.0):.4f}")
    print(f"  BERTScore F1:  {metrics.get('bertscore_f1', 0.0):.4f}")
    print(f"Saved to: {out_file}\n")

    return final_payload


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generation across RAG paradigms.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "closed_book", "bm25", "medcpt", "hybrid"], help="Mode to evaluate")
    parser.add_argument("--max_questions", type=int, default=100, help="Number of questions to evaluate (default: 100)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between LLM calls in seconds (default: 2.0)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Groq model name")
    args = parser.parse_args()

    keys = get_groq_keys()
    print(f"Loaded {len(keys)} Groq API key(s) from .env")

    questions = load_test_questions(max_questions=args.max_questions)

    # Need retrieval assets?
    needs_retrieval = (args.mode in ["all", "bm25", "medcpt", "hybrid"])
    documents = None
    docs_by_pmid = None
    bm25 = None
    medcpt_index, medcpt_pmids, medcpt_tok, medcpt_model = None, None, None, None

    if needs_retrieval:
        documents = load_corpus()
        docs_by_pmid = {doc["pmid"]: doc for doc in documents}

        if args.mode in ["all", "bm25", "hybrid"]:
            bm25 = build_bm25_index(documents)

        if args.mode in ["all", "medcpt", "hybrid"]:
            medcpt_index, medcpt_pmids, medcpt_tok, medcpt_model = load_medcpt_components()

    modes_to_run = ["closed_book", "bm25", "medcpt", "hybrid"] if args.mode == "all" else [args.mode]
    all_summary = {}

    for m in modes_to_run:
        res = run_evaluation_for_mode(
            mode=m,
            questions=questions,
            keys=keys,
            model_name=args.model,
            delay_sec=args.delay,
            bm25=bm25,
            documents=documents,
            docs_by_pmid=docs_by_pmid,
            medcpt_index=medcpt_index,
            medcpt_pmids=medcpt_pmids,
            medcpt_tok=medcpt_tok,
            medcpt_model=medcpt_model
        )
        all_summary[m] = res["metrics"]

    # Save summary
    summary_path = RESULTS_DIR / "generation_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    print("\n" + "=" * 65)
    print("ALL EVALUATIONS COMPLETE! SUMMARY TABLE:")
    print("=" * 65)
    print(f"{'Method':<18} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ROUGE-L':<10} | {'BERTScore':<10}")
    print("-" * 65)
    for m, met in all_summary.items():
        acc = f"{met.get('accuracy', 0.0) * 100:.1f}%"
        mf1 = f"{met.get('macro_f1', 0.0):.4f}"
        rl = f"{met.get('rougeL', 0.0):.4f}"
        bs = f"{met.get('bertscore_f1', 0.0):.4f}"
        print(f"{m:<18} | {acc:<10} | {mf1:<10} | {rl:<10} | {bs:<10}")
    print("=" * 65)
    print(f"Summary JSON saved to: {summary_path}\n")


if __name__ == "__main__":
    main()

