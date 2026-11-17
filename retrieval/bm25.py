import json
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"


def tokenize(text):
    """Simple BM25 tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def load_corpus():
    """Load PubMed corpus."""
    documents = []

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            text = f"{record.get('title', '')} {record.get('abstract', '')}"

            documents.append({
                "pmid": record["pmid"],
                "title": record.get("title", ""),
                "abstract": record.get("abstract", ""),
                "text": text
            })

    return documents


def build_bm25(documents):
    """Build BM25 index."""
    tokenized_documents = [
        tokenize(doc["text"])
        for doc in documents
    ]

    return BM25Okapi(tokenized_documents)


def retrieve(query, bm25, documents, top_k=10):
    """Retrieve top-k documents for a query."""
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    # Get indices of highest scoring documents
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "pmid": documents[index]["pmid"],
            "title": documents[index]["title"],
            "score": float(scores[index])
        })

    return results


if __name__ == "__main__":

    print("Loading corpus...")
    documents = load_corpus()

    print(f"Loaded {len(documents):,} documents")

    print("Building BM25 index...")
    start = time.perf_counter()

    bm25 = build_bm25(documents)

    build_time = time.perf_counter() - start

    print(f"BM25 index built in {build_time:.2f} seconds")

    # Test query
    query = (
        "Does treatment with aspirin reduce the risk "
        "of cardiovascular events?"
    )

    start = time.perf_counter()

    results = retrieve(
        query,
        bm25,
        documents,
        top_k=10
    )

    retrieval_time = time.perf_counter() - start

    print("\nQuery:")
    print(query)

    print(f"\nRetrieval time: {retrieval_time * 1000:.2f} ms")

    print("\nTop 10 results:")

    for result in results:
        print(
            f"{result['rank']}. "
            f"PMID={result['pmid']} | "
            f"Score={result['score']:.4f} | "
            f"{result['title']}"
        )