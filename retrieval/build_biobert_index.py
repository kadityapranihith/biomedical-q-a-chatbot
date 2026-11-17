import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"

INDEX_DIR = BASE_DIR / "data" / "biobert_index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = INDEX_DIR / "biobert_embeddings.npy"
FAISS_INDEX_PATH = INDEX_DIR / "biobert.index"
PMIDS_PATH = INDEX_DIR / "pmids.json"


# ============================================================
# BioBERT
# ============================================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
MAX_LENGTH = 512


# ============================================================
# Load corpus
# ============================================================

def load_corpus():

    documents = []

    print("Loading corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:

        for line in f:

            record = json.loads(line)

            abstract = record.get("abstract", "").strip()

            if not abstract:
                continue

            documents.append({
                "pmid": str(record["pmid"]),
                "abstract": abstract
            })

    return documents


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
# Generate embeddings
# ============================================================

def create_embeddings(
    texts,
    tokenizer,
    model
):

    embeddings = []

    print("\nCreating BioBERT embeddings...")

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Embedding abstracts"
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
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

            outputs = model(**encoded)

            batch_embeddings = mean_pooling(
                outputs,
                encoded["attention_mask"]
            )

            # Normalize embeddings.
            # After normalization, inner product = cosine similarity.
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings,
                p=2,
                dim=1
            )

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

    return np.vstack(embeddings).astype(
        np.float32
    )


# ============================================================
# Main
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("BioBERT Dense Retrieval Index Builder")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    documents = load_corpus()

    print(
        f"Documents loaded: {len(documents):,}"
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
    # Prepare texts
    # IMPORTANT: abstract only
    # --------------------------------------------------------

    abstracts = [
        doc["abstract"]
        for doc in documents
    ]

    pmids = [
        doc["pmid"]
        for doc in documents
    ]

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embedding_start = time.time()

    embeddings = create_embeddings(
        abstracts,
        tokenizer,
        model
    )

    embedding_time = time.time() - embedding_start

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding generation time: "
        f"{embedding_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    print("\nSaving embeddings...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print(
        f"Saved: {EMBEDDINGS_PATH}"
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------
    #
    # IndexFlatIP = exact inner-product search.
    # Since embeddings are normalized, this is cosine similarity.
    #
    # This is NOT approximate search.
    # It compares the query against all vectors.
    # --------------------------------------------------------

    print("\nBuilding FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    print(
        f"Saved: {FAISS_INDEX_PATH}"
    )

    # --------------------------------------------------------
    # Save PMID mapping
    # --------------------------------------------------------

    with open(
        PMIDS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pmids,
            f,
            indent=2
        )

    print(
        f"Saved: {PMIDS_PATH}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print("\nVerification:")

    print(
        f"  Documents : {len(documents):,}"
    )

    print(
        f"  Embeddings : {len(embeddings):,}"
    )

    print(
        f"  FAISS      : {index.ntotal:,}"
    )

    print(
        f"  Dimension  : {dimension}"
    )

    print(
        f"\nTotal time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    print("\nBioBERT index created successfully.")


if __name__ == "__main__":
    main()