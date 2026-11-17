import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from tqdm import tqdm

# ============================================================
# FINAL BIOBERT CROSS-ENCODER RERANKER EVALUATION
# ============================================================
# Test set = 500 PMIDs from test_ground_truth.json.
# The reranker model must already be trained on dev only.
#
# Candidate generation is frozen:
#   BM25 Top-100 + BioBERT Dense Top-100
#   min-max normalization
#   alpha=0.95, beta=0.05
#
# Rerank the full 100 hybrid candidates, then evaluate the final Top-10.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"
INDEX_DIR = BASE_DIR / "data" / "biobert_index"
FAISS_INDEX_PATH = INDEX_DIR / "biobert.index"
PMIDS_PATH = INDEX_DIR / "pmids.json"
RERANKER_PATH = BASE_DIR / "data" / "reranker_model"
OUTPUT_PATH = BASE_DIR / "evaluation" / "biobert_reranker_results.json"

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
ALPHA = 0.95
BETA = 0.05
CANDIDATE_K = 100
RERANK_K = 100
TOP_K = 10
MAX_LENGTH = 512
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def min_max_normalize(score_dict):
    if not score_dict:
        return {}
    values = np.asarray(list(score_dict.values()), dtype=np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return {pmid: 1.0 for pmid in score_dict}
    return {pmid: float((score - lo) / (hi - lo)) for pmid, score in score_dict.items()}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_query(query, tokenizer, model):
    encoded = tokenizer(
        [query], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.inference_mode():
        out = model(**encoded)
        emb = mean_pooling(out, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def load_corpus():
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            abstract = r.get("abstract", "")
            if abstract.strip():
                docs.append({"pmid": str(r["pmid"]), "abstract": abstract})
    return docs


def build_bm25(documents):
    return BM25Okapi([tokenize(d["abstract"]) for d in documents])


def get_hybrid_candidates(question, documents, bm25, index, index_pmids, encoder_tokenizer, encoder_model):
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]
    bm25_dict = {documents[int(i)]["pmid"]: float(bm25_scores[int(i)]) for i in bm25_indices}

    qvec = encode_query(question, encoder_tokenizer, encoder_model)
    dense_scores, dense_indices = index.search(qvec, CANDIDATE_K)
    dense_dict = {}
    for score, idx in zip(dense_scores[0], dense_indices[0]):
        if int(idx) < 0:
            continue
        dense_dict[index_pmids[int(idx)]] = float(score)

    bm25_norm = min_max_normalize(bm25_dict)
    dense_norm = min_max_normalize(dense_dict)
    candidates = set(bm25_norm) | set(dense_norm)

    fused = {
        pmid: ALPHA * bm25_norm.get(pmid, 0.0) + BETA * dense_norm.get(pmid, 0.0)
        for pmid in candidates
    }

    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def rerank_candidates(question, ranked_candidates, doc_by_pmid, tokenizer, model):
    candidates = ranked_candidates[:RERANK_K]
    questions = [question] * len(candidates)
    abstracts = [doc_by_pmid[pmid]["abstract"] for pmid, _ in candidates]

    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), BATCH_SIZE):
            q_batch = questions[start:start + BATCH_SIZE]
            a_batch = abstracts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                q_batch,
                a_batch,
                padding=True,
                truncation="only_second",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().numpy().tolist())

    return sorted(
        [(pmid, float(score)) for (pmid, _), score in zip(candidates, scores)],
        key=lambda x: (-x[1], x[0]),
    )


def metrics(gold_ranks, total):
    r1 = sum(r is not None and r <= 1 for r in gold_ranks) / total
    r5 = sum(r is not None and r <= 5 for r in gold_ranks) / total
    r10 = sum(r is not None and r <= 10 for r in gold_ranks) / total
    mrr = sum(0.0 if r is None else 1.0 / r for r in gold_ranks) / total
    return {"recall@1": r1, "recall@5": r5, "recall@10": r10, "mrr": mrr}


def main():
    print("=" * 72)
    print("BIOBERT CROSS-ENCODER RERANKER — FINAL TEST")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Candidate K: {CANDIDATE_K}")
    print(f"Rerank K: {RERANK_K}")

    documents = load_corpus()
    doc_by_pmid = {d["pmid"]: d for d in documents}
    pqa = json.load(open(PQA_PATH, "r", encoding="utf-8"))
    test_data = json.load(open(TEST_PATH, "r", encoding="utf-8"))
    test_pmids = sorted(str(x) for x in test_data.keys())
    pqa_by_pmid = {str(k): v for k, v in pqa.items()}

    if len(test_pmids) != 500:
        raise ValueError(f"Expected 500 test PMIDs, found {len(test_pmids)}")

    print("Building BM25...")
    bm25_build_start = time.perf_counter()
    bm25 = build_bm25(documents)
    bm25_build_seconds = time.perf_counter() - bm25_build_start

    print("Loading FAISS dense index...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(PMIDS_PATH, "r", encoding="utf-8") as f:
        index_pmids = [str(x) for x in json.load(f)]
    if index.ntotal != len(index_pmids) or index.ntotal != len(documents):
        raise ValueError("FAISS index / PMID mapping / corpus size mismatch")

    print("Loading BioBERT encoder...")
    encoder_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    encoder_model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
    encoder_model.eval()

    print("Loading trained cross-encoder reranker...")
    reranker_tokenizer = BertTokenizer.from_pretrained(RERANKER_PATH)
    reranker = BertForSequenceClassification.from_pretrained(RERANKER_PATH).to(DEVICE)
    reranker.eval()

    gold_ranks = []
    candidate_recall_at_100 = []
    total_latencies = []
    candidate_latencies = []
    rerank_latencies = []
    detailed = []

    for pmid in tqdm(test_pmids, desc="Test questions"):
        question = pqa_by_pmid[pmid]["QUESTION"].strip()

        overall_start = time.perf_counter()
        candidate_start = time.perf_counter()
        hybrid_ranked = get_hybrid_candidates(
            question,
            documents,
            bm25,
            index,
            index_pmids,
            encoder_tokenizer,
            encoder_model,
        )
        candidate_ms = (time.perf_counter() - candidate_start) * 1000

        candidate_pmids = [p for p, _ in hybrid_ranked]
        candidate_recall_at_100.append(int(pmid in candidate_pmids[:RERANK_K]))

        rerank_start = time.perf_counter()
        reranked = rerank_candidates(
            question,
            hybrid_ranked,
            doc_by_pmid,
            reranker_tokenizer,
            reranker,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        final_pmids = [p for p, _ in reranked[:TOP_K]]
        rank = final_pmids.index(pmid) + 1 if pmid in final_pmids else None

        total_ms = (time.perf_counter() - overall_start) * 1000
        gold_ranks.append(rank)
        total_latencies.append(total_ms)
        candidate_latencies.append(candidate_ms)
        rerank_latencies.append(rerank_ms)

        detailed.append({
            "pmid": pmid,
            "question": question,
            "gold_rank": rank,
            "candidate_recall_at_100": bool(candidate_recall_at_100[-1]),
            "top_10": [
                {"rank": i + 1, "pmid": p, "reranker_score": s}
                for i, (p, s) in enumerate(reranked[:TOP_K])
            ],
            "candidate_generation_latency_ms": candidate_ms,
            "rerank_latency_ms": rerank_ms,
            "total_latency_ms": total_ms,
        })

    final_metrics = metrics(gold_ranks, len(test_pmids))
    output = {
        "method": "Tuned Hybrid + BioBERT Cross-Encoder Reranker",
        "base_model": MODEL_NAME,
        "retrieval_text": "abstract_only",
        "corpus_size": len(documents),
        "test_size": len(test_pmids),
        "candidate_k": CANDIDATE_K,
        "rerank_k": RERANK_K,
        "top_k": TOP_K,
        "alpha_bm25": ALPHA,
        "beta_biobert": BETA,
        "score_normalization": "min_max_per_query",
        "candidate_recall_at_100": float(np.mean(candidate_recall_at_100)),
        "metrics": {
            **final_metrics,
            "average_candidate_generation_latency_ms": float(np.mean(candidate_latencies)),
            "average_rerank_latency_ms": float(np.mean(rerank_latencies)),
            "average_total_latency_ms": float(np.mean(total_latencies)),
            "median_total_latency_ms": float(np.median(total_latencies)),
            "bm25_build_time_seconds": bm25_build_seconds,
        },
        "detailed_results": detailed,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL RERANKER RESULTS")
    print("=" * 72)
    print(f"Recall@1:  {final_metrics['recall@1'] * 100:.2f}%")
    print(f"Recall@5:  {final_metrics['recall@5'] * 100:.2f}%")
    print(f"Recall@10: {final_metrics['recall@10'] * 100:.2f}%")
    print(f"MRR:       {final_metrics['mrr']:.4f}")
    print(f"Candidate Recall@100: {float(np.mean(candidate_recall_at_100)) * 100:.2f}%")
    print(f"Avg candidate generation latency: {float(np.mean(candidate_latencies)):.2f} ms")
    print(f"Avg rerank latency:               {float(np.mean(rerank_latencies)):.2f} ms")
    print(f"Avg total latency:                {float(np.mean(total_latencies)):.2f} ms")
    print(f"Median total latency:             {float(np.median(total_latencies)):.2f} ms")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
