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

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    BASE_DIR /
    "data" /
    "pubmed_retrieval_corpus.jsonl"
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

METADATA_PATH = (
    INDEX_DIR /
    "metadata.json"
)


# ============================================================
# MedCPT configuration
# ============================================================

MODEL_NAME = (
    "ncbi/MedCPT-Article-Encoder"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 16

# MedCPT article encoder supports 512 tokens.
MAX_LENGTH = 512


# ============================================================
# Load PubMed retrieval corpus
# ============================================================

def load_corpus():

    documents = []

    print("\nLoading PubMed corpus...")

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            pmid = str(record["pmid"])

            # IMPORTANT:
            # ABSTRACT ONLY.
            #
            # We intentionally do NOT use:
            # record["title"]
            #
            # This keeps MedCPT consistent with
            # the abstract-only retrieval experiment.

            abstract = (
                record.get("abstract", "")
                or ""
            ).strip()

            # Skip documents without an abstract.
            if not abstract:
                continue

            documents.append({
                "pmid": pmid,
                "abstract": abstract
            })

    return documents


# ============================================================
# Encode document abstracts
# ============================================================

def encode_documents(
    documents,
    tokenizer,
    model
):

    all_embeddings = []

    print(
        "\nEncoding abstracts with MedCPT..."
    )

    start_time = time.perf_counter()

    for start in tqdm(
        range(
            0,
            len(documents),
            BATCH_SIZE
        ),
        desc="Encoding abstracts"
    ):

        batch = documents[
            start:start + BATCH_SIZE
        ]

        abstracts = [
            doc["abstract"]
            for doc in batch
        ]

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        encoded = tokenizer(
            abstracts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                **encoded
            )

            # MedCPT uses the [CLS] representation.
            embeddings = (
                output
                .last_hidden_state[:, 0, :]
            )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_embeddings.append(
            embeddings
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    return embeddings, elapsed


# ============================================================
# Build FAISS index
# ============================================================

def build_faiss_index(
    embeddings
):

    dimension = embeddings.shape[1]

    print(
        "\nBuilding FAISS IndexFlatIP..."
    )

    # IMPORTANT:
    #
    # The official MedCPT retrieval example
    # uses inner-product similarity.
    #
    # We therefore do NOT L2-normalize the vectors.
    #
    # This preserves the original MedCPT
    # retrieval formulation.

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MedCPT Abstract-Only Index Builder")
    print("=" * 70)

    print(
        f"\nDevice       : {DEVICE}"
    )

    print(
        f"Model        : {MODEL_NAME}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Max length   : {MAX_LENGTH}"
    )

    print(
        "\nRetrieval text: ABSTRACT ONLY"
    )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    documents = load_corpus()

    print(
        f"\nDocuments with abstracts: "
        f"{len(documents):,}"
    )

    if not documents:

        raise RuntimeError(
            "No documents with abstracts "
            "were found."
        )

    # --------------------------------------------------------
    # Check PMID uniqueness
    # --------------------------------------------------------

    pmids = [
        doc["pmid"]
        for doc in documents
    ]

    if len(pmids) != len(set(pmids)):

        raise ValueError(
            "Duplicate PMIDs found in "
            "retrieval corpus."
        )

    print(
        "PMID uniqueness check: PASSED"
    )

    # --------------------------------------------------------
    # Load MedCPT Article Encoder
    # --------------------------------------------------------

    print(
        "\nLoading MedCPT Article Encoder..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModel.from_pretrained(
            MODEL_NAME
        )
    )

    model.to(DEVICE)
    model.eval()

    print(
        "MedCPT Article Encoder loaded."
    )

    print(
        f"Embedding dimension: "
        f"{model.config.hidden_size}"
    )

    # --------------------------------------------------------
    # Encode documents
    # --------------------------------------------------------

    embeddings, encoding_time = (
        encode_documents(
            documents,
            tokenizer,
            model
        )
    )

    print(
        f"\nEmbedding matrix shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Encoding time: "
        f"{encoding_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Validate embedding count
    # --------------------------------------------------------

    if (
        embeddings.shape[0]
        != len(documents)
    ):

        raise ValueError(
            "Number of embeddings does not "
            "match number of documents."
        )

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    index = build_faiss_index(
        embeddings
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
    # Create output directory
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    print(
        "\nSaving FAISS index..."
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    # --------------------------------------------------------
    # Save PMID mapping
    # --------------------------------------------------------

    print(
        "Saving PMID mapping..."
    )

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

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "method": (
            "MedCPT Dense Retrieval"
        ),

        "model": MODEL_NAME,

        "retrieval_text": (
            "abstract_only"
        ),

        "corpus_path": str(
            CORPUS_PATH
        ),

        "corpus_size": len(
            documents
        ),

        "embedding_dimension": int(
            embeddings.shape[1]
        ),

        "max_length": MAX_LENGTH,

        "batch_size": BATCH_SIZE,

        "pooling": "CLS",

        "normalization": False,

        "faiss_index": (
            "IndexFlatIP"
        ),

        "device": str(DEVICE),

        "encoding_time_seconds": (
            encoding_time
        )
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MEDCPT INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nCorpus size : "
        f"{len(documents):,}"
    )

    print(
        f"Vectors     : "
        f"{index.ntotal:,}"
    )

    print(
        f"Dimension   : "
        f"{index.d}"
    )

    print(
        f"Encoding    : "
        f"{encoding_time:.2f} seconds"
    )

    print(
        "\nFiles created:"
    )

    print(
        f"  {FAISS_INDEX_PATH}"
    )

    print(
        f"  {PMIDS_PATH}"
    )

    print(
        f"  {METADATA_PATH}"
    )

    print(
        "\nRetrieval text: ABSTRACT ONLY"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()