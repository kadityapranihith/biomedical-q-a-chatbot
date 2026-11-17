import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load API key from:
# D:\capstone-project\.env
load_dotenv(BASE_DIR / ".env")

CORPUS_PATH = (
    BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
)

PQA_PATH = (
    BASE_DIR / "data" / "ori_pqal.json"
)

TEST_PATH = (
    BASE_DIR / "data" / "test_ground_truth.json"
)

MEDCPT_INDEX_PATH = (
    BASE_DIR
    / "data"
    / "medcpt_index"
    / "medcpt_abstract.index"
)

MEDCPT_PMIDS_PATH = (
    BASE_DIR
    / "data"
    / "medcpt_index"
    / "pmids.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "rag_bm25_medcpt_langchain_results.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MEDCPT_QUERY_MODEL = (
    "ncbi/MedCPT-Query-Encoder"
)

# Can be overridden from .env:
#
# GROQ_MODEL=openai/gpt-oss-120b
#
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Retrieval configuration
# ------------------------------------------------------------

CANDIDATE_K = 100

RAG_TOP_K = 5

BM25_WEIGHT = 0.5

MEDCPT_WEIGHT = 0.5

MEDCPT_MAX_LENGTH = 64

# ------------------------------------------------------------
# Generation configuration
# ------------------------------------------------------------

TEMPERATURE = 0

MAX_COMPLETION_TOKENS = 512

# ------------------------------------------------------------
# Evaluation configuration
# ------------------------------------------------------------
#
# First run:
#
# MAX_QUESTIONS = 5
#
# After successful testing:
#
# MAX_QUESTIONS = None
#
# ------------------------------------------------------------

MAX_QUESTIONS = None


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Simple tokenizer used for BM25.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


# ============================================================
# LOAD PUBMED CORPUS
# ABSTRACT ONLY
# ============================================================

def load_corpus():

    documents = []

    print(
        f"Loading corpus from:\n"
        f"{CORPUS_PATH}"
    )

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            record = json.loads(line)

            abstract = (
                record
                .get("abstract", "")
                .strip()
            )

            if not abstract:
                continue

            documents.append(
                {
                    "pmid": str(
                        record["pmid"]
                    ),
                    "title": record.get(
                        "title",
                        "",
                    ),
                    "abstract": abstract,
                }
            )

    return documents


# ============================================================
# LOAD PUBMEDQA QUESTIONS
# ============================================================

def load_questions():

    print(
        f"Loading PubMedQA data from:\n"
        f"{PQA_PATH}"
    )

    with open(
        PQA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        pqa_data = json.load(f)

    print(
        f"Loading test split from:\n"
        f"{TEST_PATH}"
    )

    with open(
        TEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        test_ground_truth = json.load(f)

    questions = []

    for pmid, gold_label in (
        test_ground_truth.items()
    ):

        pmid = str(pmid)

        if pmid not in pqa_data:

            print(
                f"WARNING: PMID {pmid} "
                f"not found in ori_pqal.json"
            )

            continue

        record = pqa_data[pmid]

        question = (
            record
            .get("QUESTION", "")
            .strip()
        )

        if not question:
            continue

        questions.append(
            {
                "pmid": pmid,
                "question": question,
                "gold_label": (
                    str(gold_label)
                    .lower()
                    .strip()
                ),
            }
        )

    if MAX_QUESTIONS is not None:

        questions = questions[
            :MAX_QUESTIONS
        ]

    return questions


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(documents):

    print(
        "\nTokenizing abstracts for BM25..."
    )

    tokenized_documents = [
        tokenize(doc["abstract"])
        for doc in documents
    ]

    print(
        "Building BM25 index..."
    )

    return BM25Okapi(
        tokenized_documents
    )


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def min_max_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if len(scores) == 0:

        return scores

    min_score = float(
        scores.min()
    )

    max_score = float(
        scores.max()
    )

    if (
        max_score - min_score
        < 1e-12
    ):

        return np.zeros_like(
            scores
        )

    return (
        (scores - min_score)
        / (max_score - min_score)
    )


# ============================================================
# MEDCPT QUERY ENCODING
# ============================================================

def encode_medcpt_query(
    query,
    tokenizer,
    model,
):

    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=MEDCPT_MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        output = model(
            **encoded
        )

        # MedCPT query representation:
        # CLS token embedding.
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
# HYBRID RETRIEVAL
#
# BM25 TOP 100
# +
# MedCPT TOP 100
# ↓
# Min-max normalization
# ↓
# 50/50 score fusion
# ↓
# FINAL TOP 5
# ============================================================

def hybrid_retrieve(
    query,
    bm25,
    documents,
    medcpt_index,
    medcpt_pmids,
    medcpt_tokenizer,
    medcpt_model,
):

    # --------------------------------------------------------
    # BM25 RETRIEVAL
    # --------------------------------------------------------

    query_tokens = tokenize(
        query
    )

    bm25_scores_all = (
        bm25.get_scores(
            query_tokens
        )
    )

    bm25_top_indices = (
        np.argsort(
            bm25_scores_all
        )[::-1][:CANDIDATE_K]
    )

    bm25_pmids = [
        documents[int(i)]["pmid"]
        for i in bm25_top_indices
    ]

    bm25_raw_scores = [
        float(
            bm25_scores_all[
                int(i)
            ]
        )
        for i in bm25_top_indices
    ]

    # --------------------------------------------------------
    # MedCPT RETRIEVAL
    # --------------------------------------------------------

    query_embedding = (
        encode_medcpt_query(
            query,
            medcpt_tokenizer,
            medcpt_model,
        )
    )

    medcpt_scores, medcpt_indices = (
        medcpt_index.search(
            query_embedding,
            CANDIDATE_K,
        )
    )

    medcpt_scores = (
        medcpt_scores[0]
    )

    medcpt_indices = (
        medcpt_indices[0]
    )

    medcpt_pmids_top = []

    medcpt_raw_scores = []

    for score, index in zip(
        medcpt_scores,
        medcpt_indices,
    ):

        if int(index) < 0:
            continue

        medcpt_pmids_top.append(
            str(
                medcpt_pmids[
                    int(index)
                ]
            )
        )

        medcpt_raw_scores.append(
            float(score)
        )

    # --------------------------------------------------------
    # NORMALIZE SCORES
    # --------------------------------------------------------

    bm25_norm = (
        min_max_normalize(
            bm25_raw_scores
        )
    )

    medcpt_norm = (
        min_max_normalize(
            medcpt_raw_scores
        )
    )

    # --------------------------------------------------------
    # PMID -> DOCUMENT
    # --------------------------------------------------------

    documents_by_pmid = {
        doc["pmid"]: doc
        for doc in documents
    }

    # --------------------------------------------------------
    # FUSE SCORES
    # --------------------------------------------------------

    fused_scores = {}

    # BM25 contribution
    for pmid, score in zip(
        bm25_pmids,
        bm25_norm,
    ):

        fused_scores[pmid] = (
            fused_scores.get(
                pmid,
                0.0,
            )
            + BM25_WEIGHT
            * float(score)
        )

    # MedCPT contribution
    for pmid, score in zip(
        medcpt_pmids_top,
        medcpt_norm,
    ):

        fused_scores[pmid] = (
            fused_scores.get(
                pmid,
                0.0,
            )
            + MEDCPT_WEIGHT
            * float(score)
        )

    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    ranked_pmids = sorted(
        fused_scores.keys(),
        key=lambda pmid:
            fused_scores[pmid],
        reverse=True,
    )[:RAG_TOP_K]

    results = []

    for rank, pmid in enumerate(
        ranked_pmids,
        start=1,
    ):

        doc = (
            documents_by_pmid
            .get(pmid)
        )

        if doc is None:
            continue

        results.append(
            {
                "rank": rank,
                "pmid": pmid,
                "title": doc["title"],
                "abstract": doc[
                    "abstract"
                ],
                "hybrid_score": float(
                    fused_scores[pmid]
                ),
            }
        )

    return results


# ============================================================
# LANGCHAIN PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a biomedical question-answering system.

Use ONLY the supplied PubMed abstracts as evidence.

Do not use outside knowledge.

Do not use web search.

Determine whether the answer to the question
is YES, NO, or MAYBE.

Return exactly this format:

Answer: YES
Explanation: <brief evidence-based explanation>

OR

Answer: NO
Explanation: <brief evidence-based explanation>

OR

Answer: MAYBE
Explanation: <brief evidence-based explanation>

If the supplied evidence is insufficient to
distinguish YES from NO, answer MAYBE.

Do not invent facts that are not supported
by the supplied abstracts.
""",
        ),
        (
            "human",
            """
Question:

{question}

PubMed Evidence:

{context}
""",
        ),
    ]
)


# ============================================================
# CONVERT RETRIEVED RESULTS TO LANGCHAIN DOCUMENTS
# ============================================================

def create_langchain_documents(
    retrieved_docs,
):

    documents = []

    for item in retrieved_docs:

        page_content = (
            f"Title: {item['title']}\n"
            f"PMID: {item['pmid']}\n"
            f"Abstract: {item['abstract']}"
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "pmid": item["pmid"],
                    "title": item["title"],
                    "rank": item["rank"],
                    "hybrid_score": (
                        item[
                            "hybrid_score"
                        ]
                    ),
                },
            )
        )

    return documents


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    documents,
):

    sections = []

    for i, doc in enumerate(
        documents,
        start=1,
    ):

        sections.append(
            f"[Evidence {i}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# PARSE MODEL ANSWER
# ============================================================

def parse_answer(text):

    if not text:
        return None

    match = re.search(
        r"Answer\s*:\s*"
        r"(YES|NO|MAYBE)\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return (
            match.group(1)
            .lower()
        )

    # Fallback if model returns
    # only YES / NO / MAYBE.
    stripped = (
        text
        .strip()
        .upper()
    )

    if stripped in {
        "YES",
        "NO",
        "MAYBE",
    }:

        return stripped.lower()

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "LANGCHAIN RAG EVALUATION"
    )
    print(
        "BM25 + MedCPT + Groq"
    )
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Groq model: {GROQ_MODEL}"
    )

    print(
        f"Candidate K: {CANDIDATE_K}"
    )

    print(
        f"RAG Top K: {RAG_TOP_K}"
    )

    print(
        f"BM25 weight: {BM25_WEIGHT}"
    )

    print(
        f"MedCPT weight: "
        f"{MEDCPT_WEIGHT}"
    )

    print(
        f"Max questions: "
        f"{MAX_QUESTIONS}"
    )

    # ========================================================
    # CHECK API KEY
    # ========================================================

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:

        raise RuntimeError(
            "\nGROQ_API_KEY was not found.\n\n"
            "Make sure your .env file exists at:\n"
            f"{BASE_DIR / '.env'}\n\n"
            "and contains:\n\n"
            "GROQ_API_KEY=your_api_key_here\n"
        )

    print(
        "\nGroq API key loaded "
        "successfully from .env."
    )

    # ========================================================
    # LOAD CORPUS
    # ========================================================

    print(
        "\nLoading PubMed corpus..."
    )

    documents = load_corpus()

    print(
        f"Corpus documents: "
        f"{len(documents):,}"
    )

    # ========================================================
    # BUILD BM25
    # ========================================================

    print(
        "\nBuilding BM25 index..."
    )

    bm25_start = time.perf_counter()

    bm25 = build_bm25(
        documents
    )

    bm25_build_time = (
        time.perf_counter()
        - bm25_start
    )

    print(
        f"BM25 build time: "
        f"{bm25_build_time:.2f} seconds"
    )

    # ========================================================
    # LOAD MEDCPT FAISS INDEX
    # ========================================================

    print(
        "\nLoading MedCPT FAISS index..."
    )

    if not MEDCPT_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"MedCPT index not found:\n"
            f"{MEDCPT_INDEX_PATH}"
        )

    if not MEDCPT_PMIDS_PATH.exists():

        raise FileNotFoundError(
            f"MedCPT PMID mapping not found:\n"
            f"{MEDCPT_PMIDS_PATH}"
        )

    medcpt_index = (
        faiss.read_index(
            str(
                MEDCPT_INDEX_PATH
            )
        )
    )

    with open(
        MEDCPT_PMIDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        medcpt_pmids = json.load(f)

    print(
        f"MedCPT vectors: "
        f"{medcpt_index.ntotal:,}"
    )

    print(
        f"MedCPT PMID mappings: "
        f"{len(medcpt_pmids):,}"
    )

    if (
        medcpt_index.ntotal
        != len(medcpt_pmids)
    ):

        raise ValueError(
            "MedCPT FAISS index size "
            "does not match PMID mapping size."
        )

    # ========================================================
    # LOAD MEDCPT QUERY ENCODER
    # ========================================================

    print(
        "\nLoading MedCPT query encoder..."
    )

    medcpt_tokenizer = (
        AutoTokenizer.from_pretrained(
            MEDCPT_QUERY_MODEL
        )
    )

    medcpt_model = (
        AutoModel.from_pretrained(
            MEDCPT_QUERY_MODEL
        )
    )

    medcpt_model.to(
        DEVICE
    )

    medcpt_model.eval()

    print(
        "MedCPT query encoder loaded."
    )

    # ========================================================
    # LOAD TEST QUESTIONS
    # ========================================================

    print(
        "\nLoading PubMedQA test questions..."
    )

    questions = load_questions()

    print(
        f"Questions to evaluate: "
        f"{len(questions):,}"
    )

    # ========================================================
    # INITIALIZE LANGCHAIN + GROQ
    # ========================================================

    print(
        "\nInitializing LangChain + Groq..."
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_COMPLETION_TOKENS,
        api_key=groq_api_key,
    )

    rag_chain = (
        PROMPT
        | llm
    )

    print(
        "LangChain RAG chain initialized."
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    detailed_results = []

    retrieval_latencies_ms = []

    generation_latencies_ms = []

    correct = 0

    evaluated = 0

    print(
        "\nRunning RAG evaluation...\n"
    )

    for item in tqdm(
        questions,
        desc="RAG evaluation",
    ):

        question = item[
            "question"
        ]

        gold_pmid = item[
            "pmid"
        ]

        gold_label = item[
            "gold_label"
        ]

        # ====================================================
        # RETRIEVAL
        # ====================================================

        retrieval_start = (
            time.perf_counter()
        )

        try:

            retrieved = (
                hybrid_retrieve(
                    query=question,
                    bm25=bm25,
                    documents=documents,
                    medcpt_index=(
                        medcpt_index
                    ),
                    medcpt_pmids=(
                        medcpt_pmids
                    ),
                    medcpt_tokenizer=(
                        medcpt_tokenizer
                    ),
                    medcpt_model=(
                        medcpt_model
                    ),
                )
            )

            retrieval_error = None

        except Exception as exc:

            retrieved = []

            retrieval_error = str(
                exc
            )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        retrieval_latencies_ms.append(
            retrieval_latency_ms
        )

        # ====================================================
        # LANGCHAIN DOCUMENTS
        # ====================================================

        lc_documents = (
            create_langchain_documents(
                retrieved
            )
        )

        context = (
            format_context(
                lc_documents
            )
        )

        # ====================================================
        # GENERATION
        # ====================================================

        generation_start = (
            time.perf_counter()
        )

        generated_text = ""

        predicted_label = None

        generation_error = None

        usage = {}

        try:

            response = (
                rag_chain.invoke(
                    {
                        "question":
                            question,
                        "context":
                            context,
                    }
                )
            )

            generated_text = (
                response.content
            )

            predicted_label = (
                parse_answer(
                    generated_text
                )
            )

            response_metadata = (
                getattr(
                    response,
                    "response_metadata",
                    {},
                )
            )

            if isinstance(
                response_metadata,
                dict,
            ):

                usage = (
                    response_metadata.get(
                        "token_usage",
                        response_metadata.get(
                            "usage",
                            {},
                        ),
                    )
                )

        except Exception as exc:

            generation_error = str(
                exc
            )

        generation_latency_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        generation_latencies_ms.append(
            generation_latency_ms
        )

        # ====================================================
        # EVALUATION
        # ====================================================

        is_correct = None

        if predicted_label is not None:

            evaluated += 1

            is_correct = (
                predicted_label
                == gold_label
            )

            if is_correct:

                correct += 1

        # ====================================================
        # SAVE RESULT
        # ====================================================

        detailed_results.append(
            {
                "pmid": gold_pmid,
                "question": question,
                "gold_label": gold_label,
                "predicted_label": (
                    predicted_label
                ),
                "correct": is_correct,
                "generated_answer": (
                    generated_text
                ),
                "retrieval_latency_ms": (
                    retrieval_latency_ms
                ),
                "generation_latency_ms": (
                    generation_latency_ms
                ),
                "usage": usage,
                "retrieval_error": (
                    retrieval_error
                ),
                "generation_error": (
                    generation_error
                ),
                "retrieved_documents": (
                    retrieved
                ),
            }
        )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = (
        correct / evaluated
        if evaluated > 0
        else 0.0
    )

    average_retrieval_latency = (
        float(
            np.mean(
                retrieval_latencies_ms
            )
        )
        if retrieval_latencies_ms
        else 0.0
    )

    median_retrieval_latency = (
        float(
            np.median(
                retrieval_latencies_ms
            )
        )
        if retrieval_latencies_ms
        else 0.0
    )

    average_generation_latency = (
        float(
            np.mean(
                generation_latencies_ms
            )
        )
        if generation_latencies_ms
        else 0.0
    )

    median_generation_latency = (
        float(
            np.median(
                generation_latencies_ms
            )
        )
        if generation_latencies_ms
        else 0.0
    )

    average_total_latency = (
        average_retrieval_latency
        + average_generation_latency
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion = {

        "yes": {
            "yes": 0,
            "no": 0,
            "maybe": 0,
        },

        "no": {
            "yes": 0,
            "no": 0,
            "maybe": 0,
        },

        "maybe": {
            "yes": 0,
            "no": 0,
            "maybe": 0,
        },
    }

    for result in detailed_results:

        gold = result[
            "gold_label"
        ]

        pred = result[
            "predicted_label"
        ]

        if (
            gold in confusion
            and pred in {
                "yes",
                "no",
                "maybe",
            }
        ):

            confusion[
                gold
            ][pred] += 1

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "method":
            "LangChain RAG - BM25 + MedCPT",

        "retrieval_method":
            "BM25 + MedCPT",

        "rag_framework":
            "LangChain",

        "generation_provider":
            "Groq",

        "generation_model":
            GROQ_MODEL,

        "retrieval_text":
            "abstract_only",

        "corpus_size":
            len(documents),

        "test_size":
            len(questions),

        "candidate_k":
            CANDIDATE_K,

        "rag_top_k":
            RAG_TOP_K,

        "bm25_weight":
            BM25_WEIGHT,

        "medcpt_weight":
            MEDCPT_WEIGHT,

        "score_normalization":
            "min_max_per_query",

        "metrics": {

            "answer_accuracy":
                accuracy,

            "evaluated_answers":
                evaluated,

            "correct_answers":
                correct,

            "failed_or_unparsed_answers":
                (
                    len(questions)
                    - evaluated
                ),

            "average_retrieval_latency_ms":
                average_retrieval_latency,

            "median_retrieval_latency_ms":
                median_retrieval_latency,

            "average_generation_latency_ms":
                average_generation_latency,

            "median_generation_latency_ms":
                median_generation_latency,

            "average_total_latency_ms":
                average_total_latency,

            "bm25_build_time_seconds":
                bm25_build_time,
        },

        "confusion_matrix":
            confusion,

        "detailed_results":
            detailed_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LANGCHAIN RAG RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Questions: "
        f"{len(questions)}"
    )

    print(
        f"Successfully evaluated: "
        f"{evaluated}"
    )

    print(
        f"Correct: "
        f"{correct}"
    )

    print(
        f"Answer accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Average retrieval latency: "
        f"{average_retrieval_latency:.2f} ms"
    )

    print(
        f"Median retrieval latency: "
        f"{median_retrieval_latency:.2f} ms"
    )

    print(
        f"Average generation latency: "
        f"{average_generation_latency:.2f} ms"
    )

    print(
        f"Median generation latency: "
        f"{median_generation_latency:.2f} ms"
    )

    print(
        f"Average total latency: "
        f"{average_total_latency:.2f} ms"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()