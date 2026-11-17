import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
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

OUTPUT_PATH = (
    BASE_DIR /
    "evaluation" /
    "medcpt_results.json"
)


# ============================================================
# MedCPT configuration
# ============================================================

QUERY_MODEL_NAME = (
    "ncbi/MedCPT-Query-Encoder"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TOP_K = 10

# Official MedCPT query encoder example
# uses max_length=64.
MAX_LENGTH = 64


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

        output = model(
            **encoded
        )

        # MedCPT uses the [CLS]
        # representation.
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
# Load PubMedQA test questions
# ============================================================

def load_questions():

    print(
        "\nLoading PubMedQA..."
    )

    with open(
        PQA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        pqa_data = json.load(f)

    print(
        f"PubMedQA records: "
        f"{len(pqa_data):,}"
    )

    print(
        "\nLoading test ground truth..."
    )

    with open(
        TEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        test_ground_truth = json.load(f)

    print(
        f"Test PMIDs: "
        f"{len(test_ground_truth):,}"
    )

    questions = []

    missing_pmids = []

    missing_questions = []

    for pmid, answer in (
        test_ground_truth.items()
    ):

        pmid = str(pmid)

        # ----------------------------------------------------
        # Check PMID exists in PubMedQA
        # ----------------------------------------------------

        if pmid not in pqa_data:

            missing_pmids.append(
                pmid
            )

            continue

        record = pqa_data[pmid]

        question = (
            record.get(
                "QUESTION",
                ""
            )
            or ""
        ).strip()

        if not question:

            missing_questions.append(
                pmid
            )

            continue

        questions.append({

            "pmid": pmid,

            "question": question,

            "answer": answer

        })

    # --------------------------------------------------------
    # Report problems
    # --------------------------------------------------------

    if missing_pmids:

        print(
            f"\nWARNING: "
            f"{len(missing_pmids)} test PMIDs "
            "were not found in ori_pqal.json."
        )

    if missing_questions:

        print(
            f"\nWARNING: "
            f"{len(missing_questions)} test PMIDs "
            "have no QUESTION."
        )

    return questions


# ============================================================
# Calculate MRR
# ============================================================

def calculate_mrr(results):

    reciprocal_ranks = []

    for result in results:

        rank = result[
            "gold_rank"
        ]

        if rank is None:

            reciprocal_ranks.append(
                0.0
            )

        else:

            reciprocal_ranks.append(
                1.0 / rank
            )

    if not reciprocal_ranks:

        return 0.0

    return (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MedCPT Dense Retrieval Evaluation")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "\nRetrieval text: ABSTRACT ONLY"
    )

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    print(
        "\nLoading FAISS index..."
    )

    if not FAISS_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found:\n"
            f"{FAISS_INDEX_PATH}\n\n"
            "Run build_medcpt_index.py first."
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

    # --------------------------------------------------------
    # Load PMID mapping
    # --------------------------------------------------------

    print(
        "\nLoading PMID mapping..."
    )

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

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (
        index.ntotal
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
        f"\nQuestions to evaluate: "
        f"{len(questions):,}"
    )

    if not questions:

        raise RuntimeError(
            "No test questions were loaded."
        )

    # --------------------------------------------------------
    # Check gold PMID coverage
    # --------------------------------------------------------

    pmid_set = set(
        str(pmid)
        for pmid in pmids
    )

    missing_gold = [
        item["pmid"]
        for item in questions
        if item["pmid"]
        not in pmid_set
    ]

    if missing_gold:

        print(
            f"\nWARNING: "
            f"{len(missing_gold)} gold PMIDs "
            "are not present in the FAISS index."
        )

        print(
            "Example missing PMIDs:",
            missing_gold[:10]
        )

    else:

        print(
            "Gold PMID coverage: PASSED"
        )

    # --------------------------------------------------------
    # Load MedCPT Query Encoder
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

    print(
        f"Embedding dimension: "
        f"{model.config.hidden_size}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    detailed_results = []

    latencies = []

    print(
        "\nRunning retrieval evaluation..."
    )

    for item in tqdm(
        questions,
        desc="Evaluating"
    ):

        question = item[
            "question"
        ]

        gold_pmid = str(
            item["pmid"]
        )

        # ----------------------------------------------------
        # Start latency measurement
        #
        # Same basic methodology as the existing
        # BioBERT evaluator:
        #
        # query encoding + FAISS search
        # ----------------------------------------------------

        start = time.perf_counter()

        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------

        query_embedding = encode_query(
            question,
            tokenizer,
            model
        )

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        scores, indices = (
            index.search(
                query_embedding,
                TOP_K
            )
        )

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency
        )

        # ----------------------------------------------------
        # Convert FAISS indices to PMIDs
        # ----------------------------------------------------

        retrieved_pmids = [
            str(pmids[int(i)])
            for i in indices[0]
            if int(i) >= 0
        ]

        retrieved_scores = [
            float(score)
            for score in scores[0][
                :len(retrieved_pmids)
            ]
        ]

        # ----------------------------------------------------
        # Find gold PMID rank
        # ----------------------------------------------------

        gold_rank = None

        for rank, retrieved_pmid in enumerate(
            retrieved_pmids,
            start=1
        ):

            if (
                str(retrieved_pmid)
                == gold_pmid
            ):

                gold_rank = rank

                break

        # ----------------------------------------------------
        # Save detailed result
        # ----------------------------------------------------

        top_10 = []

        for rank, (
            retrieved_pmid,
            score
        ) in enumerate(
            zip(
                retrieved_pmids,
                retrieved_scores
            ),
            start=1
        ):

            top_10.append({

                "rank": rank,

                "pmid": retrieved_pmid,

                "score": score

            })

        detailed_results.append({

            "pmid": gold_pmid,

            "question": question,

            "gold_rank": gold_rank,

            "top_10": top_10

        })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(
        detailed_results
    )

    recall_at_1 = (
        sum(
            r["gold_rank"] is not None
            and r["gold_rank"] <= 1
            for r in detailed_results
        )
        / total
    )

    recall_at_5 = (
        sum(
            r["gold_rank"] is not None
            and r["gold_rank"] <= 5
            for r in detailed_results
        )
        / total
    )

    recall_at_10 = (
        sum(
            r["gold_rank"] is not None
            and r["gold_rank"] <= 10
            for r in detailed_results
        )
        / total
    )

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
    # Results object
    # --------------------------------------------------------

    results = {

        "method": (
            "MedCPT Dense Retrieval"
        ),

        "model": QUERY_MODEL_NAME,

        "article_encoder": (
            "ncbi/MedCPT-Article-Encoder"
        ),

        "retrieval_text": (
            "abstract_only"
        ),

        "corpus_size": (
            index.ntotal
        ),

        "test_size": total,

        "metrics": {

            "recall_at_1": (
                recall_at_1
            ),

            "recall_at_5": (
                recall_at_5
            ),

            "recall_at_10": (
                recall_at_10
            ),

            "mrr": mrr,

            "average_latency_ms": (
                average_latency
            ),

            "median_latency_ms": (
                median_latency
            )

        },

        "detailed_results": (
            detailed_results
        )

    }

    # --------------------------------------------------------
    # Save results
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
    # Print final results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MEDCPT RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCorpus size      : "
        f"{index.ntotal:,}"
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
        "\nRetrieval text   : "
        "ABSTRACT ONLY"
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