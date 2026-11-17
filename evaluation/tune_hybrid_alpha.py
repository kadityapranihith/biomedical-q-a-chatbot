import json
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"

INDEX_DIR = BASE_DIR / "data" / "biobert_index"
FAISS_INDEX_PATH = INDEX_DIR / "biobert.index"
PMIDS_PATH = INDEX_DIR / "pmids.json"

OUTPUT_PATH = BASE_DIR / "evaluation" / "hybrid_alpha_tuning_results.json"


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

CANDIDATE_K = 100
TOP_K = 10
MAX_LENGTH = 512

# Tune BM25 weight.
ALPHAS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Tokenizer
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# Load corpus
# ABSTRACT ONLY
# ============================================================

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


# ============================================================
# Load PubMedQA
# ============================================================

def load_pqa():
    with open(PQA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Load held-out TEST PMIDs
# The remaining 500 PubMedQA examples become DEV.
# ============================================================

def load_test_pmids():
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(pmid) for pmid in data.keys()}


# ============================================================
# Build BM25
# ============================================================

def build_bm25(documents):
    tokenized_documents = [
        tokenize(doc["abstract"])
        for doc in documents
    ]

    return BM25Okapi(tokenized_documents)


# ============================================================
# Min-max normalization
# Same idea as the current 50/50 hybrid experiment.
# ============================================================

def min_max_normalize(score_dict):
    if not score_dict:
        return {}

    values = np.array(list(score_dict.values()), dtype=np.float32)

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


# ============================================================
# BioBERT mean pooling
# ============================================================

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


# ============================================================
# Encode query
# ============================================================

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

    with torch.no_grad():
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

    return embedding.cpu().numpy().astype(np.float32)


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(gold_ranks):
    total = len(gold_ranks)

    recall_1 = sum(
        rank is not None and rank <= 1
        for rank in gold_ranks
    ) / total

    recall_5 = sum(
        rank is not None and rank <= 5
        for rank in gold_ranks
    ) / total

    recall_10 = sum(
        rank is not None and rank <= 10
        for rank in gold_ranks
    ) / total

    mrr = sum(
        0.0 if rank is None else 1.0 / rank
        for rank in gold_ranks
    ) / total

    return {
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID ALPHA TUNING — DEVELOPMENT SET")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Alpha values: {ALPHAS}")

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print("\nLoading corpus...")

    documents = load_corpus()

    print(
        f"Corpus documents: {len(documents):,}"
    )

    pmid_to_index = {
        doc["pmid"]: i
        for i, doc in enumerate(documents)
    }

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
    # Split DEV / TEST
    # --------------------------------------------------------

    test_pmids = load_test_pmids()

    all_pmids = set(pqa_by_pmid.keys())

    dev_pmids = sorted(
        all_pmids - test_pmids
    )

    print(f"Total PubMedQA: {len(all_pmids):,}")
    print(f"Test PMIDs:     {len(test_pmids):,}")
    print(f"Dev PMIDs:      {len(dev_pmids):,}")

    if len(dev_pmids) == 0:
        raise ValueError("Development set is empty.")

    missing_dev = [
        pmid
        for pmid in dev_pmids
        if pmid not in pmid_to_index
    ]

    if missing_dev:
        raise ValueError(
            f"{len(missing_dev)} development PMIDs "
            "are missing from the retrieval corpus."
        )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 index using ABSTRACT ONLY...")

    bm25 = build_bm25(documents)

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print("\nLoading BioBERT FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(PMIDS_PATH, "r", encoding="utf-8") as f:
        index_pmids = [
            str(pmid)
            for pmid in json.load(f)
        ]

    if index.ntotal != len(index_pmids):
        raise ValueError(
            "FAISS index size does not match PMID mapping size."
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
    # Retrieve ONCE for every dev question.
    # Then alpha tuning happens without re-running retrieval.
    # --------------------------------------------------------

    cached_candidates = []

    print("\nRetrieving development candidates...")

    for pmid in tqdm(dev_pmids):

        question = pqa_by_pmid[pmid]["QUESTION"].strip()

        # -------------------------------
        # BM25 Top 100
        # -------------------------------

        bm25_scores = bm25.get_scores(
            tokenize(question)
        )

        bm25_indices = np.argsort(
            bm25_scores
        )[-CANDIDATE_K:][::-1]

        bm25_dict = {
            documents[i]["pmid"]:
                float(bm25_scores[i])
            for i in bm25_indices
        }

        # -------------------------------
        # BioBERT Top 100
        # -------------------------------

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
                index_pmids[idx]
            ] = float(score)

        # -------------------------------
        # Normalize independently
        # -------------------------------

        bm25_norm = min_max_normalize(
            bm25_dict
        )

        dense_norm = min_max_normalize(
            dense_dict
        )

        # -------------------------------
        # Union candidate set
        # -------------------------------

        candidates = (
            set(bm25_norm.keys()) |
            set(dense_norm.keys())
        )

        cached_candidates.append({
            "gold_pmid": pmid,
            "bm25_norm": bm25_norm,
            "dense_norm": dense_norm,
            "candidates": list(candidates)
        })

    # --------------------------------------------------------
    # Try every alpha WITHOUT rerunning retrieval.
    # --------------------------------------------------------

    tuning_results = []

    print("\nEvaluating alpha values...")

    for alpha in ALPHAS:

        beta = 1.0 - alpha

        gold_ranks = []

        for item in cached_candidates:

            scores = {}

            for candidate in item["candidates"]:

                bm25_score = item["bm25_norm"].get(
                    candidate,
                    0.0
                )

                dense_score = item["dense_norm"].get(
                    candidate,
                    0.0
                )

                scores[candidate] = (
                    alpha * bm25_score +
                    beta * dense_score
                )

            ranked = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_K]

            ranked_pmids = [
                candidate
                for candidate, _ in ranked
            ]

            try:
                rank = (
                    ranked_pmids.index(
                        item["gold_pmid"]
                    ) + 1
                )
            except ValueError:
                rank = None

            gold_ranks.append(rank)

        metrics = calculate_metrics(
            gold_ranks
        )

        tuning_results.append({
            "alpha_bm25": alpha,
            "beta_biobert": beta,
            **metrics
        })

        print(
            f"alpha={alpha:.2f} | "
            f"R@1={metrics['recall@1']:.4f} | "
            f"R@5={metrics['recall@5']:.4f} | "
            f"R@10={metrics['recall@10']:.4f} | "
            f"MRR={metrics['mrr']:.4f}"
        )

    # --------------------------------------------------------
    # Select alpha using DEV MRR.
    # MRR is the primary tuning objective because it rewards
    # placing the correct article higher in the ranking.
    # --------------------------------------------------------

    best = max(
        tuning_results,
        key=lambda x: x["mrr"]
    )

    output = {
        "method": "Hybrid BM25 + BioBERT",
        "model": MODEL_NAME,
        "retrieval_text": "abstract_only",
        "corpus_size": len(documents),
        "dev_size": len(dev_pmids),
        "test_size": len(test_pmids),
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "score_normalization": "min_max_per_query",
        "tuning_metric": "MRR",
        "alpha_search": ALPHAS,
        "best_alpha": best["alpha_bm25"],
        "best_beta": best["beta_biobert"],
        "best_dev_metrics": best,
        "all_results": tuning_results
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

    print("\n" + "=" * 70)
    print("BEST DEVELOPMENT-SET WEIGHT")
    print("=" * 70)

    print(
        f"alpha (BM25):    {best['alpha_bm25']:.2f}"
    )
    print(
        f"beta (BioBERT):  {best['beta_biobert']:.2f}"
    )
    print(
        f"R@1:             {best['recall@1']:.4f}"
    )
    print(
        f"R@5:             {best['recall@5']:.4f}"
    )
    print(
        f"R@10:            {best['recall@10']:.4f}"
    )
    print(
        f"MRR:             {best['mrr']:.4f}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
