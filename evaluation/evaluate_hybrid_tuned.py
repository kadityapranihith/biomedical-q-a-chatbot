import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


# ============================================================
# FINAL TUNED HYBRID EVALUATION — HELD-OUT TEST SET
# ============================================================
#
# Alpha was selected on the 500-question development set:
#   alpha_bm25 = 0.95
#   beta_biobert = 0.05
#
# IMPORTANT:
# - Do NOT change alpha based on these test results.
# - Test set is read only from test_ground_truth.json.
# - Retrieval text is ABSTRACT ONLY.
# - BM25 Top-100 + BioBERT Top-100 are fused.
# - Scores are min-max normalized per query.
# - Final ranking is Top-10.
#
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"

INDEX_DIR = BASE_DIR / "data" / "biobert_index"
FAISS_INDEX_PATH = INDEX_DIR / "biobert.index"
PMIDS_PATH = INDEX_DIR / "pmids.json"

OUTPUT_PATH = BASE_DIR / "evaluation" / "hybrid_tuned_results.json"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

ALPHA_BM25 = 0.95
BETA_BIOBERT = 0.05

CANDIDATE_K = 100
TOP_K = 10
MAX_LENGTH = 512

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ------------------------------------------------------------
# Load corpus
# ABSTRACT ONLY
# ------------------------------------------------------------

def load_corpus():
    documents = []

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            abstract = record.get("abstract", "")

            if not abstract.strip():
                continue

            documents.append({
                "pmid": str(record["pmid"]),
                "abstract": abstract
            })

    return documents


# ------------------------------------------------------------
# Load PubMedQA
# ------------------------------------------------------------

def load_pqa():
    with open(PQA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Load test set
# test_ground_truth is:
# PMID -> yes/no/maybe
# The PMID is the gold source article for retrieval.
# ------------------------------------------------------------

def load_test_data():
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        return {
            str(pmid): answer
            for pmid, answer in json.load(f).items()
        }


# ------------------------------------------------------------
# Build BM25
# ------------------------------------------------------------

def build_bm25(documents):
    tokenized_documents = [
        tokenize(doc["abstract"])
        for doc in documents
    ]

    return BM25Okapi(tokenized_documents)


# ------------------------------------------------------------
# Min-max normalization
# Same fusion normalization used in the 50/50 baseline
# ------------------------------------------------------------

def min_max_normalize(score_dict):
    if not score_dict:
        return {}

    values = np.asarray(
        list(score_dict.values()),
        dtype=np.float32
    )

    min_score = float(values.min())
    max_score = float(values.max())

    if max_score == min_score:
        return {
            pmid: 1.0
            for pmid in score_dict
        }

    return {
        pmid: float(
            (score - min_score) /
            (max_score - min_score)
        )
        for pmid, score in score_dict.items()
    }


# ------------------------------------------------------------
# BioBERT mean pooling
# ------------------------------------------------------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    summed = torch.sum(
        token_embeddings * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ------------------------------------------------------------
# Encode query
# ------------------------------------------------------------

def encode_query(query, tokenizer, model):
    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output = model(**encoded)

        embedding = mean_pooling(
            output,
            encoded["attention_mask"]
        )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

    return embedding.cpu().numpy().astype(
        np.float32
    )


# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

def calculate_metrics(gold_ranks):
    total = len(gold_ranks)

    recall_at_1 = sum(
        rank is not None and rank <= 1
        for rank in gold_ranks
    ) / total

    recall_at_5 = sum(
        rank is not None and rank <= 5
        for rank in gold_ranks
    ) / total

    recall_at_10 = sum(
        rank is not None and rank <= 10
        for rank in gold_ranks
    ) / total

    mrr = sum(
        0.0 if rank is None else 1.0 / rank
        for rank in gold_ranks
    ) / total

    return {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "mrr": mrr
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("FINAL TUNED HYBRID EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"BM25 weight (alpha):   {ALPHA_BM25}")
    print(f"BioBERT weight (beta): {BETA_BIOBERT}")

    if abs(ALPHA_BM25 + BETA_BIOBERT - 1.0) > 1e-9:
        raise ValueError(
            "Alpha + beta must equal 1.0."
        )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print("\nLoading corpus...")

    documents = load_corpus()

    print(
        f"Corpus documents: {len(documents):,}"
    )

    # --------------------------------------------------------
    # Load PubMedQA
    # --------------------------------------------------------

    print("\nLoading PubMedQA...")

    pqa = load_pqa()

    pqa_by_pmid = {
        str(pmid): record
        for pmid, record in pqa.items()
    }

    # --------------------------------------------------------
    # Load test set
    # --------------------------------------------------------

    print("\nLoading test set...")

    test_data = load_test_data()
    test_pmids = sorted(test_data.keys())

    print(
        f"Test PMIDs: {len(test_pmids):,}"
    )

    if len(test_pmids) != 500:
        raise ValueError(
            f"Expected 500 test PMIDs, found {len(test_pmids)}."
        )

    missing_questions = [
        pmid
        for pmid in test_pmids
        if pmid not in pqa_by_pmid
    ]

    if missing_questions:
        raise ValueError(
            f"{len(missing_questions)} test PMIDs are missing "
            "from ori_pqal.json."
        )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index using ABSTRACT ONLY..."
    )

    build_start = time.perf_counter()

    bm25 = build_bm25(documents)

    build_time = time.perf_counter() - build_start

    print(
        f"BM25 build time: {build_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print("\nLoading BioBERT FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(
        PMIDS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        index_pmids = [
            str(pmid)
            for pmid in json.load(f)
        ]

    if index.ntotal != len(index_pmids):
        raise ValueError(
            "FAISS index size does not match PMID mapping size."
        )

    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS index size does not match corpus size."
        )

    # --------------------------------------------------------
    # Load BioBERT
    # --------------------------------------------------------

    print("\nLoading BioBERT...")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = BertModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nRunning final test evaluation...\n")

    detailed_results = []
    latencies = []
    gold_ranks = []

    for pmid in tqdm(
        test_pmids,
        desc="Evaluating"
    ):

        question = pqa_by_pmid[pmid]["QUESTION"].strip()

        start = time.perf_counter()

        # ----------------------------------------------------
        # BM25 Top-100
        # ----------------------------------------------------

        bm25_scores = bm25.get_scores(
            tokenize(question)
        )

        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:CANDIDATE_K]

        bm25_dict = {
            documents[i]["pmid"]:
                float(bm25_scores[i])
            for i in bm25_top_indices
        }

        # ----------------------------------------------------
        # BioBERT Top-100
        # ----------------------------------------------------

        query_embedding = encode_query(
            question,
            tokenizer,
            model
        )

        dense_scores, dense_indices = index.search(
            query_embedding,
            CANDIDATE_K
        )

        dense_dict = {}

        for score, idx in zip(
            dense_scores[0],
            dense_indices[0]
        ):
            if idx == -1:
                continue

            dense_dict[
                index_pmids[int(idx)]
            ] = float(score)

        # ----------------------------------------------------
        # Normalize scores independently
        # ----------------------------------------------------

        bm25_norm = min_max_normalize(
            bm25_dict
        )

        dense_norm = min_max_normalize(
            dense_dict
        )

        # ----------------------------------------------------
        # Union candidates
        # ----------------------------------------------------

        candidates = (
            set(bm25_norm.keys()) |
            set(dense_norm.keys())
        )

        # ----------------------------------------------------
        # Hybrid fusion
        #
        # Missing score in one retriever = 0.0
        # ----------------------------------------------------

        fused_scores = {}

        for candidate in candidates:

            bm25_score = bm25_norm.get(
                candidate,
                0.0
            )

            dense_score = dense_norm.get(
                candidate,
                0.0
            )

            fused_scores[candidate] = (
                ALPHA_BM25 * bm25_score +
                BETA_BIOBERT * dense_score
            )

        # ----------------------------------------------------
        # Final Top-10
        # ----------------------------------------------------

        ranked_candidates = sorted(
            fused_scores.items(),
            key=lambda x: (
                -x[1],
                x[0]
            )
        )[:TOP_K]

        retrieved_pmids = [
            candidate
            for candidate, _ in ranked_candidates
        ]

        # ----------------------------------------------------
        # Gold rank
        # ----------------------------------------------------

        if pmid in retrieved_pmids:
            rank = (
                retrieved_pmids.index(pmid) + 1
            )
        else:
            rank = None

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency_ms)
        gold_ranks.append(rank)

        # ----------------------------------------------------
        # Detailed result
        # ----------------------------------------------------

        top_10 = []

        for rank_number, (candidate, fused_score) in enumerate(
            ranked_candidates,
            start=1
        ):

            top_10.append({
                "rank": rank_number,
                "pmid": candidate,
                "bm25_score": (
                    bm25_dict.get(candidate)
                ),
                "dense_score": (
                    dense_dict.get(candidate)
                ),
                "bm25_normalized": (
                    bm25_norm.get(candidate, 0.0)
                ),
                "dense_normalized": (
                    dense_norm.get(candidate, 0.0)
                ),
                "hybrid_score": float(fused_score)
            })

        detailed_results.append({
            "pmid": pmid,
            "question": question,
            "gold_rank": rank,
            "top_10": top_10,
            "latency_ms": latency_ms
        })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        gold_ranks
    )

    average_latency = float(
        np.mean(latencies)
    )

    median_latency = float(
        np.median(latencies)
    )

    # --------------------------------------------------------
    # Save final results
    # --------------------------------------------------------

    output = {
        "method": "Hybrid BM25 + BioBERT",
        "model": MODEL_NAME,
        "retrieval_text": "abstract_only",
        "corpus_size": len(documents),
        "test_size": len(test_pmids),
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "alpha_bm25": ALPHA_BM25,
        "beta_biobert": BETA_BIOBERT,
        "score_normalization": "min_max_per_query",
        "alpha_selection": {
            "source": "development_set",
            "dev_size": 500,
            "selection_metric": "MRR",
            "selected_alpha": ALPHA_BM25,
            "selected_beta": BETA_BIOBERT
        },
        "metrics": {
            **metrics,
            "average_latency_ms": average_latency,
            "median_latency_ms": median_latency,
            "bm25_build_time_seconds": build_time
        },
        "detailed_results": detailed_results
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TUNED HYBRID TEST RESULTS")
    print("=" * 70)

    print(
        f"Test questions:   {len(test_pmids)}"
    )
    print(
        f"Alpha (BM25):     {ALPHA_BM25:.2f}"
    )
    print(
        f"Beta (BioBERT):   {BETA_BIOBERT:.2f}"
    )
    print(
        f"Recall@1:         {metrics['recall@1'] * 100:.2f}%"
    )
    print(
        f"Recall@5:         {metrics['recall@5'] * 100:.2f}%"
    )
    print(
        f"Recall@10:        {metrics['recall@10'] * 100:.2f}%"
    )
    print(
        f"MRR:              {metrics['mrr']:.4f}"
    )
    print(
        f"Average latency:  {average_latency:.2f} ms"
    )
    print(
        f"Median latency:   {median_latency:.2f} ms"
    )

    print("=" * 70)
    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
