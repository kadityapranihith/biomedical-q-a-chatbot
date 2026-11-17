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
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"

FAISS_INDEX_PATH = (
    BASE_DIR / "data" / "biobert_index" / "biobert.index"
)

PMIDS_PATH = (
    BASE_DIR / "data" / "biobert_index" / "pmids.json"
)

OUTPUT_PATH = (
    BASE_DIR / "evaluation" / "hybrid_results.json"
)


# ============================================================
# BioBERT
# ============================================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_LENGTH = 512


# ============================================================
# Hybrid settings
# ============================================================

CANDIDATE_K = 100
TOP_K = 10

# BM25 weight
ALPHA = 0.5

# BioBERT weight = 1 - ALPHA
BETA = 1.0 - ALPHA


# ============================================================
# BM25 tokenizer
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# Load corpus
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
                "abstract": abstract
            })

    return documents


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
# Mean pooling
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output.last_hidden_state
    )

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

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
# Encode query using BioBERT
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


# ============================================================
# Load test questions
# ============================================================

def load_questions():

    with open(
        PQA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        pqa_data = json.load(f)

    with open(
        TEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        test_ground_truth = json.load(f)

    questions = []

    for pmid, answer in test_ground_truth.items():

        pmid = str(pmid)

        if pmid not in pqa_data:
            continue

        record = pqa_data[pmid]

        question = record.get(
            "QUESTION",
            ""
        ).strip()

        if not question:
            continue

        questions.append({
            "pmid": pmid,
            "question": question,
            "answer": answer
        })

    return questions


# ============================================================
# Min-max normalization
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    min_score = np.min(scores)
    max_score = np.max(scores)

    # Avoid division by zero
    if max_score - min_score < 1e-12:
        return np.ones_like(scores)

    return (
        (scores - min_score)
        / (max_score - min_score)
    )


# ============================================================
# Hybrid retrieval
# ============================================================

def hybrid_retrieve(
    question,
    bm25,
    documents,
    faiss_index,
    pmids,
    tokenizer,
    model
):

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    query_tokens = tokenize(
        question
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_top_indices = np.argsort(
        bm25_scores
    )[::-1][:CANDIDATE_K]


    # --------------------------------------------------------
    # BioBERT
    # --------------------------------------------------------

    query_embedding = encode_query(
        question,
        tokenizer,
        model
    )

    dense_scores, dense_indices = (
        faiss_index.search(
            query_embedding,
            CANDIDATE_K
        )
    )

    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]


    # --------------------------------------------------------
    # Create candidate dictionary
    # --------------------------------------------------------

    candidates = {}

    # BM25 candidates
    for index in bm25_top_indices:

        pmid = str(
            documents[int(index)]["pmid"]
        )

        candidates[pmid] = {
            "bm25_score": float(
                bm25_scores[int(index)]
            ),
            "dense_score": None
        }


    # BioBERT candidates
    for score, index in zip(
        dense_scores,
        dense_indices
    ):

        pmid = str(
            pmids[int(index)]
        )

        if pmid not in candidates:

            candidates[pmid] = {
                "bm25_score": None,
                "dense_score": float(score)
            }

        else:

            candidates[pmid][
                "dense_score"
            ] = float(score)


    # --------------------------------------------------------
    # Normalize BM25 scores
    # --------------------------------------------------------

    bm25_values = [
        item["bm25_score"]
        for item in candidates.values()
        if item["bm25_score"] is not None
    ]

    dense_values = [
        item["dense_score"]
        for item in candidates.values()
        if item["dense_score"] is not None
    ]


    normalized_bm25 = normalize_scores(
        bm25_values
    )

    normalized_dense = normalize_scores(
        dense_values
    )


    # --------------------------------------------------------
    # Map normalized scores back to candidates
    # --------------------------------------------------------

    bm25_counter = 0
    dense_counter = 0

    for item in candidates.values():

        if item["bm25_score"] is not None:

            item["bm25_normalized"] = float(
                normalized_bm25[bm25_counter]
            )

            bm25_counter += 1

        else:

            item["bm25_normalized"] = 0.0


        if item["dense_score"] is not None:

            item["dense_normalized"] = float(
                normalized_dense[dense_counter]
            )

            dense_counter += 1

        else:

            item["dense_normalized"] = 0.0


        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        item["hybrid_score"] = (
            ALPHA
            * item["bm25_normalized"]
            +
            BETA
            * item["dense_normalized"]
        )


    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1]["hybrid_score"],
        reverse=True
    )

    return ranked[:TOP_K]


# ============================================================
# MRR
# ============================================================

def calculate_mrr(results):

    reciprocal_ranks = []

    for result in results:

        rank = result["gold_rank"]

        if rank is None:

            reciprocal_ranks.append(
                0.0
            )

        else:

            reciprocal_ranks.append(
                1.0 / rank
            )

    return (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Hybrid BM25 + BioBERT Retrieval Evaluation")
    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"BM25 weight: {ALPHA}"
    )

    print(
        f"BioBERT weight: {BETA}"
    )


    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print(
        "\nLoading corpus..."
    )

    documents = load_corpus()

    print(
        f"Corpus documents: "
        f"{len(documents):,}"
    )


    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    start = time.perf_counter()

    bm25 = build_bm25(
        documents
    )

    bm25_build_time = (
        time.perf_counter()
        - start
    )

    print(
        f"BM25 build time: "
        f"{bm25_build_time:.2f} seconds"
    )


    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print(
        "\nLoading BioBERT FAISS index..."
    )

    faiss_index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS vectors: "
        f"{faiss_index.ntotal:,}"
    )


    # --------------------------------------------------------
    # Load PMID mapping
    # --------------------------------------------------------

    with open(
        PMIDS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        pmids = json.load(f)

    print(
        f"PMID mappings: "
        f"{len(pmids):,}"
    )


    if (
        faiss_index.ntotal
        != len(pmids)
    ):

        raise ValueError(
            "FAISS index size does not "
            "match PMID mapping size."
        )


    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"Test questions: "
        f"{len(questions):,}"
    )


    # --------------------------------------------------------
    # Load BioBERT
    # --------------------------------------------------------

    print(
        "\nLoading BioBERT..."
    )

    tokenizer = (
        BertTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        BertModel.from_pretrained(
            MODEL_NAME
        )
    )

    model.to(DEVICE)
    model.eval()


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    detailed_results = []
    latencies = []

    print(
        "\nRunning hybrid evaluation..."
    )

    for item in tqdm(
        questions,
        desc="Evaluating"
    ):

        question = item["question"]
        gold_pmid = str(
            item["pmid"]
        )

        start = time.perf_counter()

        ranked = hybrid_retrieve(
            question,
            bm25,
            documents,
            faiss_index,
            pmids,
            tokenizer,
            model
        )

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency
        )


        # ----------------------------------------------------
        # Extract PMIDs
        # ----------------------------------------------------

        retrieved_pmids = [
            str(pmid)
            for pmid, _ in ranked
        ]


        # ----------------------------------------------------
        # Find gold rank
        # ----------------------------------------------------

        gold_rank = None

        for rank, retrieved_pmid in enumerate(
            retrieved_pmids,
            start=1
        ):

            if retrieved_pmid == gold_pmid:

                gold_rank = rank
                break


        # ----------------------------------------------------
        # Save detailed result
        # ----------------------------------------------------

        top_10 = []

        for rank, (
            pmid,
            scores
        ) in enumerate(
            ranked,
            start=1
        ):

            top_10.append({
                "rank": rank,
                "pmid": pmid,
                "bm25_score": scores[
                    "bm25_score"
                ],
                "dense_score": scores[
                    "dense_score"
                ],
                "bm25_normalized": scores[
                    "bm25_normalized"
                ],
                "dense_normalized": scores[
                    "dense_normalized"
                ],
                "hybrid_score": scores[
                    "hybrid_score"
                ]
            })


        detailed_results.append({
            "pmid": gold_pmid,
            "question": question,
            "gold_rank": gold_rank,
            "top_10": top_10,
            "latency_ms": latency
        })


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(
        detailed_results
    )


    recall_at_1 = sum(
        r["gold_rank"] is not None
        and r["gold_rank"] <= 1
        for r in detailed_results
    ) / total


    recall_at_5 = sum(
        r["gold_rank"] is not None
        and r["gold_rank"] <= 5
        for r in detailed_results
    ) / total


    recall_at_10 = sum(
        r["gold_rank"] is not None
        and r["gold_rank"] <= 10
        for r in detailed_results
    ) / total


    mrr = calculate_mrr(
        detailed_results
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

    results = {

        "method":
            "Hybrid BM25 + BioBERT",

        "model":
            MODEL_NAME,

        "retrieval_text":
            "abstract_only",

        "corpus_size":
            len(documents),

        "test_size":
            total,

        "candidate_k":
            CANDIDATE_K,

        "top_k":
            TOP_K,

        "alpha_bm25":
            ALPHA,

        "beta_biobert":
            BETA,

        "score_normalization":
            "min_max_per_query",

        "metrics": {

            "recall@1":
                recall_at_1,

            "recall@5":
                recall_at_5,

            "recall@10":
                recall_at_10,

            "mrr":
                mrr,

            "average_latency_ms":
                average_latency,

            "median_latency_ms":
                median_latency
        },

        "detailed_results":
            detailed_results
    }


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            results,
            f,
            indent=2
        )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "HYBRID RETRIEVAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nCorpus size      : "
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
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()