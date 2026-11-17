import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Paths
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CORPUS_PATH = (
    BASE_DIR /
    "data" /
    "pubmed_retrieval_corpus.jsonl"
)

PQA_PATH = (
    BASE_DIR /
    "data" /
    "ori_pqal.json"
)

TEST_PATH = (
    BASE_DIR /
    "data" /
    "test_ground_truth.json"
)

INDEX_DIR = (
    BASE_DIR /
    "data" /
    "medcpt_index"
)

FAISS_INDEX_PATH = (
    INDEX_DIR /
    "medcpt_abstract.index"
)

PMIDS_PATH = (
    INDEX_DIR /
    "pmids.json"
)

OUTPUT_PATH = (
    BASE_DIR /
    "evaluation" /
    "bm25_medcpt_results.json"
)


# ============================================================
# Configuration
# ============================================================

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CANDIDATE_K = 100
TOP_K = 10
MAX_LENGTH = 64

# Initial unfitted fusion.
# A tuned alpha can be evaluated later on a separate
# development set; do not tune this value on the 500-question
# test set.
ALPHA_BM25 = 0.50
BETA_MEDCPT = 0.50


# ============================================================
# Tokenizer
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# Load corpus -- ABSTRACT ONLY
# ============================================================

def load_corpus():

    documents = []

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            abstract = record.get(
                "abstract",
                ""
            )

            if not abstract.strip():
                continue

            documents.append({
                "pmid": str(record["pmid"]),
                "title": record.get("title", ""),
                "abstract": abstract,
                "text": abstract
            })

    return documents


# ============================================================
# Load PubMedQA
# ============================================================

def load_pqa():

    with open(
        PQA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# Load test PMIDs
# ============================================================

def load_test_pmids():

    with open(
        TEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return {
        str(pmid)
        for pmid in data.keys()
    }


# ============================================================
# Build BM25
# ============================================================

def build_bm25(documents):

    tokenized_documents = [
        tokenize(doc["abstract"])
        for doc in documents
    ]

    return BM25Okapi(
        tokenized_documents
    )


# ============================================================
# Encode one MedCPT query
# ============================================================

def encode_query(
    query,
    tokenizer,
    model
):

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

        output = model(
            **encoded
        )

        embedding = (
            output
            .last_hidden_state[:, 0, :]
        )

    return (
        embedding
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# Per-query min-max normalization
#
# Scores are normalized only over each retriever's candidate
# list. Documents not retrieved by that retriever receive 0.
# ============================================================

def min_max_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    if len(scores) == 0:
        return scores

    minimum = float(
        np.min(scores)
    )

    maximum = float(
        np.max(scores)
    )

    if maximum == minimum:
        return np.ones_like(
            scores,
            dtype=np.float32
        )

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("BM25 + MedCPT Hybrid Retrieval Evaluation")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "\nRetrieval text: ABSTRACT ONLY"
    )

    print(
        f"Candidate K: {CANDIDATE_K}"
    )

    print(
        f"Final Top K: {TOP_K}"
    )

    print(
        f"BM25 weight: {ALPHA_BM25}"
    )

    print(
        f"MedCPT weight: {BETA_MEDCPT}"
    )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print("\nLoading corpus...")

    documents = load_corpus()

    print(
        f"Corpus documents: "
        f"{len(documents):,}"
    )

    # --------------------------------------------------------
    # Load PubMedQA and test set
    # --------------------------------------------------------

    print("\nLoading PubMedQA...")

    pqa = load_pqa()

    print(
        f"PubMedQA records: "
        f"{len(pqa):,}"
    )

    print("\nLoading test set...")

    test_pmids = load_test_pmids()

    print(
        f"Test PMIDs: "
        f"{len(test_pmids):,}"
    )

    pqa_by_pmid = {
        str(pmid): record
        for pmid, record in pqa.items()
    }

    missing_questions = [
        pmid
        for pmid in test_pmids
        if pmid not in pqa_by_pmid
    ]

    if missing_questions:

        raise RuntimeError(
            f"{len(missing_questions)} "
            "test PMIDs are missing from PubMedQA."
        )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index using ABSTRACT ONLY..."
    )

    build_start = time.perf_counter()

    bm25 = build_bm25(
        documents
    )

    bm25_build_time = (
        time.perf_counter()
        - build_start
    )

    print(
        f"BM25 build time: "
        f"{bm25_build_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Load MedCPT FAISS index
    # --------------------------------------------------------

    print(
        "\nLoading MedCPT FAISS index..."
    )

    if not FAISS_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found:\n"
            f"{FAISS_INDEX_PATH}\n\n"
            "Run your existing build_medcpt_index.py first."
        )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal:,}"
    )

    print(
        f"FAISS dimension: "
        f"{index.d}"
    )

    with open(
        PMIDS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        medcpt_pmids = json.load(f)

    print(
        f"PMID mappings: "
        f"{len(medcpt_pmids):,}"
    )

    if index.ntotal != len(
        medcpt_pmids
    ):

        raise ValueError(
            "FAISS index size does not "
            "match PMID mapping size."
        )

    medcpt_pmid_to_index = {
        str(pmid): i
        for i, pmid in enumerate(
            medcpt_pmids
        )
    }

    # --------------------------------------------------------
    # Load MedCPT query encoder
    # --------------------------------------------------------

    print(
        "\nLoading MedCPT Query Encoder..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            QUERY_MODEL_NAME
        )
    )

    model = (
        AutoModel.from_pretrained(
            QUERY_MODEL_NAME
        )
    )

    model.to(DEVICE)
    model.eval()

    print(
        "MedCPT Query Encoder loaded."
    )

    # --------------------------------------------------------
    # Check gold coverage
    # --------------------------------------------------------

    missing_gold = [
        pmid
        for pmid in test_pmids
        if pmid not in medcpt_pmid_to_index
    ]

    if missing_gold:

        raise RuntimeError(
            f"{len(missing_gold)} gold PMIDs "
            "are missing from the MedCPT index. "
            f"Example: {missing_gold[:10]}"
        )

    print(
        "Gold PMID coverage: PASSED"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    reciprocal_rank_sum = 0.0

    latencies = []

    detailed_results = []

    print(
        "\nRunning BM25 + MedCPT evaluation...\n"
    )

    for count, pmid in enumerate(
        sorted(test_pmids),
        start=1
    ):

        question = (
            pqa_by_pmid[pmid]["QUESTION"]
        )

        start = time.perf_counter()

        # ----------------------------------------------------
        # BM25 candidate retrieval
        # ----------------------------------------------------

        query_tokens = tokenize(
            question
        )

        bm25_scores_full = (
            bm25.get_scores(
                query_tokens
            )
        )

        bm25_candidate_indices = np.argpartition(
            -bm25_scores_full,
            min(
                CANDIDATE_K,
                len(bm25_scores_full)
            ) - 1
        )[
            :min(
                CANDIDATE_K,
                len(bm25_scores_full)
            )
        ]

        bm25_candidate_indices = sorted(
            bm25_candidate_indices,
            key=lambda i: bm25_scores_full[i],
            reverse=True
        )

        # ----------------------------------------------------
        # MedCPT candidate retrieval
        # ----------------------------------------------------

        query_embedding = encode_query(
            question,
            tokenizer,
            model
        )

        medcpt_scores, medcpt_indices = (
            index.search(
                query_embedding,
                CANDIDATE_K
            )
        )

        medcpt_scores = medcpt_scores[0]
        medcpt_indices = medcpt_indices[0]

        # ----------------------------------------------------
        # Normalize BM25 scores over BM25 candidates
        # ----------------------------------------------------

        bm25_candidate_scores = np.asarray(
            [
                bm25_scores_full[i]
                for i in bm25_candidate_indices
            ],
            dtype=np.float32
        )

        bm25_normalized = min_max_normalize(
            bm25_candidate_scores
        )

        bm25_norm_by_pmid = {
            documents[idx]["pmid"]: float(score)
            for idx, score in zip(
                bm25_candidate_indices,
                bm25_normalized
            )
        }

        # ----------------------------------------------------
        # Normalize MedCPT scores over MedCPT candidates
        # ----------------------------------------------------

        valid_medcpt_pairs = [
            (int(idx), float(score))
            for idx, score in zip(
                medcpt_indices,
                medcpt_scores
            )
            if int(idx) >= 0
        ]

        medcpt_candidate_scores = np.asarray(
            [
                score
                for _, score
                in valid_medcpt_pairs
            ],
            dtype=np.float32
        )

        medcpt_normalized = min_max_normalize(
            medcpt_candidate_scores
        )

        medcpt_norm_by_pmid = {
            str(medcpt_pmids[idx]): float(score)
            for (idx, _), score in zip(
                valid_medcpt_pairs,
                medcpt_normalized
            )
        }

        # ----------------------------------------------------
        # Candidate union
        # ----------------------------------------------------

        candidate_pmids = set(
            bm25_norm_by_pmid.keys()
        )

        candidate_pmids.update(
            medcpt_norm_by_pmid.keys()
        )

        # ----------------------------------------------------
        # Fuse scores
        #
        # Missing retriever score = 0.
        # ----------------------------------------------------

        fused_candidates = []

        for candidate_pmid in candidate_pmids:

            bm25_score = (
                bm25_norm_by_pmid.get(
                    candidate_pmid,
                    0.0
                )
            )

            medcpt_score = (
                medcpt_norm_by_pmid.get(
                    candidate_pmid,
                    0.0
                )
            )

            fused_score = (
                ALPHA_BM25 * bm25_score
                +
                BETA_MEDCPT * medcpt_score
            )

            fused_candidates.append(
                (
                    candidate_pmid,
                    fused_score,
                    bm25_score,
                    medcpt_score
                )
            )

        # Deterministic ordering:
        # fused score descending, then PMID ascending.
        fused_candidates.sort(
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        top_candidates = (
            fused_candidates[:TOP_K]
        )

        retrieved_pmids = [
            item[0]
            for item in top_candidates
        ]

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency
        )

        # ----------------------------------------------------
        # Gold rank
        # ----------------------------------------------------

        gold_rank = None

        if pmid in retrieved_pmids:

            gold_rank = (
                retrieved_pmids.index(pmid)
                + 1
            )

        if gold_rank is not None:

            if gold_rank <= 1:
                recall_1 += 1

            if gold_rank <= 5:
                recall_5 += 1

            if gold_rank <= 10:
                recall_10 += 1

            reciprocal_rank_sum += (
                1.0 / gold_rank
            )

        # ----------------------------------------------------
        # Detailed result
        # ----------------------------------------------------

        top_10 = []

        for rank, (
            candidate_pmid,
            fused_score,
            bm25_score,
            medcpt_score
        ) in enumerate(
            top_candidates,
            start=1
        ):

            top_10.append({
                "rank": rank,
                "pmid": candidate_pmid,
                "fused_score": fused_score,
                "bm25_normalized_score": bm25_score,
                "medcpt_normalized_score": medcpt_score
            })

        detailed_results.append({
            "pmid": pmid,
            "question": question,
            "gold_rank": gold_rank,
            "top_10": top_10,
            "latency_ms": latency
        })

        if count % 50 == 0:

            print(
                f"Processed "
                f"{count}/{len(test_pmids)}"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(
        detailed_results
    )

    recall_at_1 = (
        recall_1 / total
    )

    recall_at_5 = (
        recall_5 / total
    )

    recall_at_10 = (
        recall_10 / total
    )

    mrr = (
        reciprocal_rank_sum
        / total
    )

    average_latency = float(
        np.mean(latencies)
    )

    median_latency = float(
        np.median(latencies)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    output = {

        "method": "BM25 + MedCPT",

        "bm25_weight": ALPHA_BM25,

        "medcpt_weight": BETA_MEDCPT,

        "score_normalization": (
            "min_max_per_query"
        ),

        "candidate_k": CANDIDATE_K,

        "top_k": TOP_K,

        "retrieval_text": "abstract_only",

        "corpus_size": len(documents),

        "test_size": total,

        "device": str(DEVICE),

        "medcpt_query_model": (
            QUERY_MODEL_NAME
        ),

        "medcpt_article_encoder": (
            "ncbi/MedCPT-Article-Encoder"
        ),

        "metrics": {

            "recall@1": recall_at_1,

            "recall@5": recall_at_5,

            "recall@10": recall_at_10,

            "mrr": mrr,

            "average_latency_ms": (
                average_latency
            ),

            "median_latency_ms": (
                median_latency
            ),

            "bm25_index_build_time_seconds": (
                bm25_build_time
            )
        },

        "results": detailed_results
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

    print(
        "\n" + "=" * 70
    )

    print(
        "BM25 + MEDCPT RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Corpus size      : "
        f"{len(documents):,}"
    )

    print(
        f"Test questions   : "
        f"{total:,}"
    )

    print(
        f"Recall@1         : "
        f"{recall_at_1:.4f}"
    )

    print(
        f"Recall@5         : "
        f"{recall_at_5:.4f}"
    )

    print(
        f"Recall@10        : "
        f"{recall_at_10:.4f}"
    )

    print(
        f"MRR              : "
        f"{mrr:.4f}"
    )

    print(
        f"Average latency  : "
        f"{average_latency:.2f} ms"
    )

    print(
        f"Median latency   : "
        f"{median_latency:.2f} ms"
    )

    print(
        f"BM25 build time  : "
        f"{bm25_build_time:.2f} s"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
