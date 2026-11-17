import json
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
PQA_PATH = BASE_DIR / "data" / "ori_pqal.json"
TEST_PATH = BASE_DIR / "data" / "test_ground_truth.json"


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# Load corpus
# --------------------------------------------------

def load_corpus():

    documents = []

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:

        for line in f:

            record = json.loads(line)

            # IMPORTANT:
            # Retrieval uses ABSTRACT ONLY
            abstract = record.get("abstract", "")

            if not abstract.strip():
                continue

            documents.append({
                "pmid": str(record["pmid"]),
                "title": record.get("title", ""),
                "abstract": abstract,
                "text": abstract
            })

    return documents


# --------------------------------------------------
# Load PubMedQA
# --------------------------------------------------

def load_pqa():

    with open(PQA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Load 500 test PMIDs
# --------------------------------------------------

def load_test_pmids():

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(pmid) for pmid in data.keys()}


# --------------------------------------------------
# Build BM25
# --------------------------------------------------

def build_bm25(documents):

    tokenized_documents = [
        tokenize(doc["abstract"])
        for doc in documents
    ]

    return BM25Okapi(tokenized_documents)


# --------------------------------------------------
# Retrieve
# --------------------------------------------------

def retrieve(query, bm25, documents, top_k=10):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        documents[i]["pmid"]
        for i in top_indices
    ]


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------

    print("Loading corpus...")

    documents = load_corpus()

    print(f"Corpus documents: {len(documents):,}")


    # --------------------------------------------------
    # Load PubMedQA
    # --------------------------------------------------

    print("\nLoading PubMedQA...")

    pqa = load_pqa()

    print(f"PubMedQA questions: {len(pqa):,}")


    # --------------------------------------------------
    # Load test set
    # --------------------------------------------------

    print("\nLoading test set...")

    test_pmids = load_test_pmids()

    print(f"Test PMIDs: {len(test_pmids):,}")


    # --------------------------------------------------
    # PMID → question record
    # --------------------------------------------------

    pqa_by_pmid = {
        str(pmid): record
        for pmid, record in pqa.items()
    }


    # --------------------------------------------------
    # Check every test PMID has a question
    # --------------------------------------------------

    missing_questions = [
        pmid
        for pmid in test_pmids
        if pmid not in pqa_by_pmid
    ]

    if missing_questions:

        print(
            f"\nERROR: {len(missing_questions)} "
            "test PMIDs are missing from PubMedQA."
        )

        return


    # --------------------------------------------------
    # Build BM25
    # --------------------------------------------------

    print("\nBuilding BM25 index using ABSTRACT ONLY...")

    start = time.perf_counter()

    bm25 = build_bm25(documents)

    build_time = time.perf_counter() - start

    print(
        f"BM25 build time: "
        f"{build_time:.2f} seconds"
    )


    # --------------------------------------------------
    # Evaluation counters
    # --------------------------------------------------

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    reciprocal_rank_sum = 0

    latencies = []

    results = []


    # --------------------------------------------------
    # Evaluate all 500 questions
    # --------------------------------------------------

    print("\nEvaluating test questions...\n")

    for count, pmid in enumerate(
        sorted(test_pmids),
        start=1
    ):

        # Get question using Gold PMID
        item = pqa_by_pmid[pmid]

        question = item["QUESTION"]


        # --------------------------------------------------
        # Retrieve Top 10
        # --------------------------------------------------

        start = time.perf_counter()

        retrieved_pmids = retrieve(
            question,
            bm25,
            documents,
            top_k=10
        )

        latency = time.perf_counter() - start

        latencies.append(latency)


        # --------------------------------------------------
        # Recall@1
        # --------------------------------------------------

        if pmid in retrieved_pmids[:1]:
            recall_1 += 1


        # --------------------------------------------------
        # Recall@5
        # --------------------------------------------------

        if pmid in retrieved_pmids[:5]:
            recall_5 += 1


        # --------------------------------------------------
        # Recall@10
        # --------------------------------------------------

        if pmid in retrieved_pmids[:10]:
            recall_10 += 1


        # --------------------------------------------------
        # MRR
        # --------------------------------------------------

        if pmid in retrieved_pmids:

            rank = retrieved_pmids.index(pmid) + 1

            reciprocal_rank_sum += 1 / rank

        else:

            rank = None


        # --------------------------------------------------
        # Save detailed result
        # --------------------------------------------------

        results.append({
            "pmid": pmid,
            "question": question,
            "gold_rank": rank,
            "retrieved_pmids": retrieved_pmids,
            "latency_ms": latency * 1000
        })


        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if count % 50 == 0:

            print(
                f"Processed "
                f"{count}/{len(test_pmids)}"
            )


    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    total = len(test_pmids)

    recall_at_1 = recall_1 / total
    recall_at_5 = recall_5 / total
    recall_at_10 = recall_10 / total

    mrr = reciprocal_rank_sum / total

    avg_latency = (
        sum(latencies) /
        len(latencies)
    )

    median_latency = sorted(latencies)[
        len(latencies) // 2
    ]


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("BM25 ABSTRACT-ONLY EVALUATION RESULTS")
    print("=" * 50)

    print(
        f"Test questions:       {total}"
    )

    print(
        f"Recall@1:             "
        f"{recall_at_1 * 100:.2f}%"
    )

    print(
        f"Recall@5:             "
        f"{recall_at_5 * 100:.2f}%"
    )

    print(
        f"Recall@10:            "
        f"{recall_at_10 * 100:.2f}%"
    )

    print(
        f"MRR:                  "
        f"{mrr:.4f}"
    )

    print(
        f"Average latency:      "
        f"{avg_latency * 1000:.2f} ms"
    )

    print(
        f"Median latency:       "
        f"{median_latency * 1000:.2f} ms"
    )

    print(
        f"Index build time:     "
        f"{build_time:.2f} seconds"
    )

    print("=" * 50)


    # --------------------------------------------------
    # Save detailed results
    # --------------------------------------------------

    output_path = (
        BASE_DIR /
        "evaluation" /
        "bm25_results.json"
    )

    output = {

        "method": "BM25",

        "retrieval_text": "abstract_only",

        "corpus_size": len(documents),

        "test_size": total,

        "metrics": {

            "recall@1": recall_at_1,

            "recall@5": recall_at_5,

            "recall@10": recall_at_10,

            "mrr": mrr,

            "average_latency_ms":
                avg_latency * 1000,

            "median_latency_ms":
                median_latency * 1000,

            "index_build_time_seconds":
                build_time
        },

        "results": results
    }


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )


    print(
        "\nDetailed results saved to:"
    )

    print(output_path)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()