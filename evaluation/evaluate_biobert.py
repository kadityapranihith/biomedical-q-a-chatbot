import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "biobert_index"

FAISS_INDEX_PATH = INDEX_DIR / "biobert.index"
PMIDS_PATH = INDEX_DIR / "pmids.json"

PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"

OUTPUT_PATH = BASE_DIR / "evaluation" / "biobert_results.json"


# ============================================================
# BioBERT
# ============================================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TOP_K = 10
MAX_LENGTH = 512


# ============================================================
# Mean pooling
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
# Encode one query
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

        output = model(**encoded)

        embedding = mean_pooling(
            output,
            encoded["attention_mask"]
        )

        # Same normalization used when building
        # the document embeddings.
        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

    return embedding.cpu().numpy().astype(
        np.float32
    )


# ============================================================
# Load PubMedQA questions
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
            print(
                f"WARNING: PMID {pmid} "
                f"not found in ori_pqal.json"
            )
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
# Calculate MRR
# ============================================================

def calculate_mrr(results):

    reciprocal_ranks = []

    for result in results:

        rank = result["gold_rank"]

        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(
                1.0 / rank
            )

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("BioBERT Dense Retrieval Evaluation")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    print("\nLoading FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
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
        f"PMID mappings: {len(pmids):,}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if index.ntotal != len(pmids):

        raise ValueError(
            "FAISS index size does not match "
            "PMID mapping size."
        )

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"Test questions: {len(questions):,}"
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

    detailed_results = []

    latencies = []

    print("\nRunning retrieval evaluation...")

    for item in tqdm(
        questions,
        desc="Evaluating"
    ):

        question = item["question"]
        gold_pmid = item["pmid"]

        start = time.perf_counter()

        # Create query embedding
        query_embedding = encode_query(
            question,
            tokenizer,
            model
        )

        # FAISS similarity search
        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

        retrieved_pmids = [
            pmids[int(i)]
            for i in indices[0]
        ]

        retrieved_scores = [
            float(score)
            for score in scores[0]
        ]

        # Find gold PMID rank
        gold_rank = None

        for rank, retrieved_pmid in enumerate(
            retrieved_pmids,
            start=1
        ):

            if str(retrieved_pmid) == str(gold_pmid):

                gold_rank = rank
                break

        detailed_results.append({

            "pmid": gold_pmid,

            "question": question,

            "gold_rank": gold_rank,

            "top_10": [
                {
                    "rank": rank,
                    "pmid": retrieved_pmids[rank - 1],
                    "score": retrieved_scores[rank - 1]
                }

                for rank in range(
                    1,
                    len(retrieved_pmids) + 1
                )
            ]

        })

    # ========================================================
    # Metrics
    # ========================================================

    total = len(detailed_results)

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

    # ========================================================
    # Results
    # ========================================================

    results = {

        "method": "BioBERT Dense Retrieval",

        "model": MODEL_NAME,

        "retrieval_text": "abstract_only",

        "corpus_size": index.ntotal,

        "test_size": total,

        "metrics": {

            "recall_at_1": recall_at_1,

            "recall_at_5": recall_at_5,

            "recall_at_10": recall_at_10,

            "mrr": mrr,

            "average_latency_ms": average_latency,

            "median_latency_ms": median_latency

        },

        "detailed_results": detailed_results

    }

    # ========================================================
    # Save
    # ========================================================

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

    # ========================================================
    # Print
    # ========================================================

    print("\n" + "=" * 60)
    print("BioBERT Retrieval Results")
    print("=" * 60)

    print(
        f"\nCorpus size      : {index.ntotal:,}"
    )

    print(
        f"Test questions   : {total:,}"
    )

    print(
        f"Recall@1         : {recall_at_1:.4f}"
    )

    print(
        f"Recall@5         : {recall_at_5:.4f}"
    )

    print(
        f"Recall@10        : {recall_at_10:.4f}"
    )

    print(
        f"MRR              : {mrr:.4f}"
    )

    print(
        f"Average latency  : {average_latency:.2f} ms"
    )

    print(
        f"Median latency   : {median_latency:.2f} ms"
    )

    print(
        f"\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()