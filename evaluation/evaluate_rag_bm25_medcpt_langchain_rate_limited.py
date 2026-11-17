import json
import os
import re
import time
import math
from collections import deque
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

# Full PubMedQA test split = 500 questions.
# Set this to 5 or 20 for a quick API/rate-limit test.
MAX_QUESTIONS = 100

# ------------------------------------------------------------
# Groq rate-limit configuration
# User-provided limits for GPT-OSS 120B Free plan:
#   RPM = 30
#   RPD = 1,000
#   TPM = 8,000
#   TPD = 200,000
# ------------------------------------------------------------

RPM_LIMIT = 30
RPD_LIMIT = 1000
TPM_LIMIT = 8000
TPD_LIMIT = 200_000

# We deliberately stay below the published limits.
# 20 seconds between requests = at most 3 requests/minute.
REQUEST_DELAY_SECONDS = 20

# Leave room below the 200k daily token quota.
DAILY_TOKEN_SAFETY_MARGIN = 10_000
DAILY_TOKEN_BUDGET = TPD_LIMIT - DAILY_TOKEN_SAFETY_MARGIN

# Conservative token estimate used BEFORE a request.
# Actual usage reported by Groq is tracked after each successful call.
ESTIMATED_CHARS_PER_TOKEN = 3.0

MAX_RETRIES = 4
BACKOFF_SECONDS = [30, 60, 120, 240]

# Save after every question so a quota/rate-limit interruption
# does not lose completed work.
SAVE_AFTER_EVERY_QUESTION = True

# Checkpoint used only by this rate-limited run.
# This prevents mixing an older experiment into a new fresh-key run.
CHECKPOINT_PATH = (
    BASE_DIR
    / "evaluation"
    / "rag_bm25_medcpt_langchain_rate_limited_progress.json"
)


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
# RATE-LIMIT / QUOTA HELPERS
# ============================================================

class DailyQuotaReached(Exception):
    """Raised when the local safety budget says to stop making API calls."""


def extract_total_tokens(usage):
    """Read total token usage from common LangChain/Groq metadata shapes."""

    if not isinstance(usage, dict):
        return 0

    # Common exact keys.
    for key in (
        "total_tokens",
        "totalTokens",
        "total_token_count",
    ):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)

    # Common prompt/completion naming.
    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("promptTokens")
        or usage.get("inputTokens")
        or 0
    )

    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completionTokens")
        or usage.get("outputTokens")
        or 0
    )

    if isinstance(prompt_tokens, (int, float)) and isinstance(
        completion_tokens, (int, float)
    ):
        return int(prompt_tokens + completion_tokens)

    return 0


def extract_usage(response):
    """Extract token-usage metadata without changing the RAG logic."""

    response_metadata = getattr(
        response,
        "response_metadata",
        {},
    )

    if not isinstance(response_metadata, dict):
        return {}

    usage = response_metadata.get(
        "token_usage",
        response_metadata.get(
            "usage",
            {},
        ),
    )

    return usage if isinstance(usage, dict) else {}


def estimate_request_tokens(question, context):
    """
    Conservative pre-request estimate.

    This is ONLY for quota protection. It does not change the prompt,
    retrieved documents, or generation parameters.
    """

    prompt_chars = len(question) + len(context) + 1500

    estimated_prompt_tokens = math.ceil(
        prompt_chars / ESTIMATED_CHARS_PER_TOKEN
    )

    return estimated_prompt_tokens + MAX_COMPLETION_TOKENS


class RateLimitManager:
    """
    Keeps the script below RPM/TPM/RPD/TPD limits.

    Important:
    - 20s spacing keeps RPM far below 30.
    - Rolling 60s token tracking protects TPM.
    - Local daily token/request counters protect against exhausting
      the daily quota during this run.
    - A safety margin is intentionally kept below the published TPD.
    """

    def __init__(self):
        self.last_request_time = None
        self.request_timestamps = deque()
        self.token_events = deque()  # (monotonic_time, tokens)
        self.total_requests = 0
        self.total_tokens = 0

    @staticmethod
    def _purge_old(queue, window_seconds=60):
        now = time.monotonic()
        while queue and now - queue[0][0] >= window_seconds:
            queue.popleft()

    def _sleep(self, seconds, reason):
        if seconds <= 0:
            return

        print(
            f"\nRate-limit wait: {seconds:.1f}s | {reason}"
        )
        time.sleep(seconds)

    def wait_for_slot(self, estimated_tokens):
        """Wait until the next request is safe to make."""

        if self.total_requests >= RPD_LIMIT:
            raise DailyQuotaReached(
                "Local RPD safety limit reached."
            )

        if self.total_tokens + estimated_tokens > DAILY_TOKEN_BUDGET:
            raise DailyQuotaReached(
                "Local TPD safety budget would be exceeded. "
                f"Used={self.total_tokens:,}, "
                f"Estimated next={estimated_tokens:,}, "
                f"Budget={DAILY_TOKEN_BUDGET:,}."
            )

        # --------------------------------------------------------
        # Fixed spacing: 20s => <= 3 RPM.
        # --------------------------------------------------------
        if self.last_request_time is not None:
            elapsed = time.monotonic() - self.last_request_time
            remaining = REQUEST_DELAY_SECONDS - elapsed
            if remaining > 0:
                self._sleep(
                    remaining,
                    "fixed request spacing",
                )

        # --------------------------------------------------------
        # RPM guard (extra protection if configuration changes).
        # --------------------------------------------------------
        now = time.monotonic()
        while self.request_timestamps and now - self.request_timestamps[0] >= 60:
            self.request_timestamps.popleft()
            now = time.monotonic()

        if len(self.request_timestamps) >= RPM_LIMIT:
            wait = 60 - (now - self.request_timestamps[0])
            self._sleep(
                max(wait, 0),
                "RPM limit protection",
            )

        # --------------------------------------------------------
        # TPM guard.
        # --------------------------------------------------------
        while True:
            now = time.monotonic()
            while self.token_events and now - self.token_events[0][0] >= 60:
                self.token_events.popleft()
                now = time.monotonic()

            tokens_last_minute = sum(
                tokens
                for _, tokens in self.token_events
            )

            if tokens_last_minute + estimated_tokens <= TPM_LIMIT:
                break

            if not self.token_events:
                raise DailyQuotaReached(
                    "Estimated single request exceeds the remaining TPM window."
                )

            wait = 60 - (now - self.token_events[0][0])
            self._sleep(
                max(wait, 0),
                (
                    "TPM protection: rolling 60s token usage is "
                    f"{tokens_last_minute:,}"
                ),
            )

        # Record the outgoing attempt for RPM protection.
        request_time = time.monotonic()
        self.last_request_time = request_time
        self.request_timestamps.append(request_time)
        self.total_requests += 1

    def record_success(self, usage):
        """Record actual token usage returned by the provider."""

        actual_tokens = extract_total_tokens(usage)

        if actual_tokens > 0:
            now = time.monotonic()
            self.token_events.append((now, actual_tokens))
            self.total_tokens += actual_tokens

        return actual_tokens

    def status(self):
        return {
            "requests_used_this_run": self.total_requests,
            "requests_remaining_from_local_rpd_budget": max(
                0,
                RPD_LIMIT - self.total_requests,
            ),
            "tokens_used_this_run": self.total_tokens,
            "tokens_remaining_from_local_tpd_budget": max(
                0,
                DAILY_TOKEN_BUDGET - self.total_tokens,
            ),
        }


def is_rate_limit_error(exc):
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


def is_daily_quota_error(exc):
    text = str(exc).lower()

    return (
        "tokens per day" in text
        or "token per day" in text
        or "tpd" in text
        or "requests per day" in text
        or "request per day" in text
        or "rpd" in text
        or "daily limit" in text
        or "daily quota" in text
        or "quota exceeded" in text
    )


def safe_rag_invoke(
    rag_chain,
    question,
    context,
    rate_manager,
):
    """
    Invoke the SAME LangChain RAG chain with rate-limit protection.

    This function does not modify the prompt or model parameters.
    """

    estimated_tokens = estimate_request_tokens(
        question,
        context,
    )

    last_exception = None

    for retry_index in range(MAX_RETRIES + 1):

        try:

            rate_manager.wait_for_slot(
                estimated_tokens
            )

            response = rag_chain.invoke(
                {
                    "question": question,
                    "context": context,
                }
            )

            usage = extract_usage(
                response
            )

            rate_manager.record_success(
                usage
            )

            return response, usage

        except DailyQuotaReached:
            raise

        except Exception as exc:

            last_exception = exc

            if not is_rate_limit_error(exc):
                raise

            if is_daily_quota_error(exc):
                raise DailyQuotaReached(
                    "Groq reported a daily quota limit:\n"
                    f"{exc}"
                )

            if retry_index >= MAX_RETRIES:
                raise RuntimeError(
                    "Maximum rate-limit retries exceeded.\n"
                    f"Last error: {exc}"
                )

            backoff = BACKOFF_SECONDS[retry_index]

            print(
                "\n429 rate-limit response. "
                f"Retry {retry_index + 1}/{MAX_RETRIES} "
                f"after {backoff}s.\n"
            )

            time.sleep(
                backoff
            )

    raise RuntimeError(
        f"Unexpected retry state: {last_exception}"
    )


def save_results(
    detailed_results,
    questions,
    corpus_size,
    bm25_build_time,
    rate_manager,
    output_path,
    final=False,
):
    """Save a checkpoint/final result using the same result schema."""

    valid_predictions = [
        result
        for result in detailed_results
        if result.get("predicted_label") in {
            "yes",
            "no",
            "maybe",
        }
    ]

    correct = sum(
        1
        for result in valid_predictions
        if result.get("correct") is True
    )

    evaluated = len(valid_predictions)

    accuracy = (
        correct / evaluated
        if evaluated > 0
        else 0.0
    )

    retrieval_values = [
        float(result.get("retrieval_latency_ms", 0.0) or 0.0)
        for result in detailed_results
        if result.get("retrieval_latency_ms") is not None
    ]

    generation_values = [
        float(result.get("generation_latency_ms", 0.0) or 0.0)
        for result in detailed_results
        if result.get("generation_latency_ms") is not None
    ]

    average_retrieval_latency = (
        float(np.mean(retrieval_values))
        if retrieval_values
        else 0.0
    )

    median_retrieval_latency = (
        float(np.median(retrieval_values))
        if retrieval_values
        else 0.0
    )

    average_generation_latency = (
        float(np.mean(generation_values))
        if generation_values
        else 0.0
    )

    median_generation_latency = (
        float(np.median(generation_values))
        if generation_values
        else 0.0
    )

    confusion = {
        "yes": {"yes": 0, "no": 0, "maybe": 0},
        "no": {"yes": 0, "no": 0, "maybe": 0},
        "maybe": {"yes": 0, "no": 0, "maybe": 0},
    }

    for result in detailed_results:
        gold = result.get("gold_label")
        pred = result.get("predicted_label")

        if (
            gold in confusion
            and pred in {"yes", "no", "maybe"}
        ):
            confusion[gold][pred] += 1

    results = {
        "method": "LangChain RAG - BM25 + MedCPT",
        "retrieval_method": "BM25 + MedCPT",
        "rag_framework": "LangChain",
        "generation_provider": "Groq",
        "generation_model": GROQ_MODEL,
        "retrieval_text": "abstract_only",
        "corpus_size": corpus_size,
        "test_size": len(questions),
        "candidate_k": CANDIDATE_K,
        "rag_top_k": RAG_TOP_K,
        "bm25_weight": BM25_WEIGHT,
        "medcpt_weight": MEDCPT_WEIGHT,
        "score_normalization": "min_max_per_query",
        "rate_limit_config": {
            "rpm_limit": RPM_LIMIT,
            "rpd_limit": RPD_LIMIT,
            "tpm_limit": TPM_LIMIT,
            "tpd_limit": TPD_LIMIT,
            "request_delay_seconds": REQUEST_DELAY_SECONDS,
            "daily_token_safety_margin": DAILY_TOKEN_SAFETY_MARGIN,
        },
        "metrics": {
            "answer_accuracy": accuracy,
            "evaluated_answers": evaluated,
            "correct_answers": correct,
            "failed_or_unparsed_answers": (
                len(questions) - evaluated
            ),
            "successful_generation_rate": (
                evaluated / len(questions)
                if questions
                else 0.0
            ),
            "average_retrieval_latency_ms": average_retrieval_latency,
            "median_retrieval_latency_ms": median_retrieval_latency,
            "average_generation_latency_ms": average_generation_latency,
            "median_generation_latency_ms": median_generation_latency,
            "average_total_latency_ms": (
                average_retrieval_latency
                + average_generation_latency
            ),
            "bm25_build_time_seconds": bm25_build_time,
            "rate_manager": rate_manager.status(),
        },
        "confusion_matrix": confusion,
        "checkpoint": not final,
        "detailed_results": detailed_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint():
    """Load only the checkpoint created by this rate-limited script."""

    if not CHECKPOINT_PATH.exists():
        return {}

    try:
        with open(
            CHECKPOINT_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        results = data.get("detailed_results", [])

        return {
            str(item.get("pmid")): item
            for item in results
            if item.get("pmid") is not None
        }

    except Exception as exc:
        print(
            f"Warning: could not load checkpoint: {exc}"
        )
        return {}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LANGCHAIN RAG EVALUATION - RATE LIMIT AWARE")
    print("BM25 + MedCPT + Groq")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Groq model: {GROQ_MODEL}")
    print(f"Candidate K: {CANDIDATE_K}")
    print(f"RAG Top K: {RAG_TOP_K}")
    print(f"BM25 weight: {BM25_WEIGHT}")
    print(f"MedCPT weight: {MEDCPT_WEIGHT}")
    print(f"Max questions: {MAX_QUESTIONS}")
    print(f"Request delay: {REQUEST_DELAY_SECONDS}s")
    print(f"Daily token budget used by script: {DAILY_TOKEN_BUDGET:,}")

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
        "\nGroq API key loaded successfully from .env."
    )

    # ========================================================
    # LOAD CORPUS
    # ========================================================

    print("\nLoading PubMed corpus...")
    documents = load_corpus()
    print(f"Corpus documents: {len(documents):,}")

    # ========================================================
    # BUILD BM25 - UNCHANGED
    # ========================================================

    print("\nBuilding BM25 index...")

    bm25_start = time.perf_counter()

    bm25 = build_bm25(
        documents
    )

    bm25_build_time = (
        time.perf_counter()
        - bm25_start
    )

    print(
        f"BM25 build time: {bm25_build_time:.2f} seconds"
    )

    # ========================================================
    # LOAD MEDCPT FAISS INDEX - UNCHANGED
    # ========================================================

    print("\nLoading MedCPT FAISS index...")

    if not MEDCPT_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"MedCPT index not found:\n{MEDCPT_INDEX_PATH}"
        )

    if not MEDCPT_PMIDS_PATH.exists():
        raise FileNotFoundError(
            f"MedCPT PMID mapping not found:\n{MEDCPT_PMIDS_PATH}"
        )

    medcpt_index = faiss.read_index(
        str(MEDCPT_INDEX_PATH)
    )

    with open(
        MEDCPT_PMIDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        medcpt_pmids = json.load(f)

    print(f"MedCPT vectors: {medcpt_index.ntotal:,}")
    print(f"MedCPT PMID mappings: {len(medcpt_pmids):,}")

    if medcpt_index.ntotal != len(medcpt_pmids):
        raise ValueError(
            "MedCPT FAISS index size does not match PMID mapping size."
        )

    # ========================================================
    # LOAD MEDCPT QUERY ENCODER - UNCHANGED
    # ========================================================

    print("\nLoading MedCPT query encoder...")

    medcpt_tokenizer = AutoTokenizer.from_pretrained(
        MEDCPT_QUERY_MODEL
    )

    medcpt_model = AutoModel.from_pretrained(
        MEDCPT_QUERY_MODEL
    )

    medcpt_model.to(DEVICE)
    medcpt_model.eval()

    print("MedCPT query encoder loaded.")

    # ========================================================
    # LOAD TEST QUESTIONS
    # ========================================================

    print("\nLoading PubMedQA test questions...")
    questions = load_questions()
    print(f"Questions to evaluate: {len(questions):,}")

    # ========================================================
    # LOAD THIS RUN'S CHECKPOINT
    # ========================================================

    checkpoint_by_pmid = load_checkpoint()

    if checkpoint_by_pmid:
        print(
            f"\nCheckpoint found: {len(checkpoint_by_pmid)} questions already processed."
        )
        print(
            "Successful predictions in checkpoint will be skipped; "
            "failed/unparsed questions will be retried."
        )
    else:
        print("\nNo previous rate-limited checkpoint found. Starting fresh.")

    # ========================================================
    # INITIALIZE LANGCHAIN + GROQ - UNCHANGED
    # ========================================================

    print("\nInitializing LangChain + Groq...")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_COMPLETION_TOKENS,
        api_key=groq_api_key,
    )

    rag_chain = PROMPT | llm

    print("LangChain RAG chain initialized.")

    # ========================================================
    # EVALUATION
    # ========================================================

    detailed_results = []
    rate_manager = RateLimitManager()

    print("\nRunning RAG evaluation...\n")

    stopped_by_daily_quota = False

    for item in tqdm(
        questions,
        desc="RAG evaluation",
    ):

        question = item["question"]
        gold_pmid = item["pmid"]
        gold_label = item["gold_label"]

        existing = checkpoint_by_pmid.get(
            str(gold_pmid)
        )

        # A successful checkpoint entry is already complete, so do not
        # spend another API request on it.
        if existing is not None and existing.get("predicted_label") in {
            "yes",
            "no",
            "maybe",
        }:
            detailed_results.append(existing)
            continue

        # ====================================================
        # RETRIEVAL - SAME CORE LOGIC
        # ====================================================

        retrieval_start = time.perf_counter()

        # Reuse saved top-5 retrieval results when this question was
        # already retrieved successfully in an earlier checkpoint.
        saved_retrieved = (
            existing.get("retrieved_documents")
            if existing is not None
            else None
        )

        if saved_retrieved:
            retrieved = saved_retrieved
            retrieval_error = existing.get("retrieval_error")
        else:
            try:
                retrieved = hybrid_retrieve(
                    query=question,
                    bm25=bm25,
                    documents=documents,
                    medcpt_index=medcpt_index,
                    medcpt_pmids=medcpt_pmids,
                    medcpt_tokenizer=medcpt_tokenizer,
                    medcpt_model=medcpt_model,
                )
                retrieval_error = None

            except Exception as exc:
                retrieved = []
                retrieval_error = str(exc)

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # ====================================================
        # CONTEXT - SAME CORE LOGIC
        # ====================================================

        lc_documents = create_langchain_documents(
            retrieved
        )

        context = format_context(
            lc_documents
        )

        # ====================================================
        # GENERATION - SAME REQUEST, RATE-LIMIT PROTECTED
        # ====================================================

        generation_start = time.perf_counter()

        generated_text = ""
        predicted_label = None
        generation_error = None
        usage = {}

        try:
            response, usage = safe_rag_invoke(
                rag_chain=rag_chain,
                question=question,
                context=context,
                rate_manager=rate_manager,
            )

            generated_text = response.content

            predicted_label = parse_answer(
                generated_text
            )

        except DailyQuotaReached as exc:

            generation_error = str(exc)
            stopped_by_daily_quota = True

        except Exception as exc:

            generation_error = str(exc)

        generation_latency_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        # ====================================================
        # EVALUATION - SAME CORE LOGIC
        # ====================================================

        is_correct = None

        if predicted_label is not None:
            is_correct = (
                predicted_label == gold_label
            )

        detailed_results.append(
            {
                "pmid": gold_pmid,
                "question": question,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "correct": is_correct,
                "generated_answer": generated_text,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "usage": usage,
                "retrieval_error": retrieval_error,
                "generation_error": generation_error,
                "retrieved_documents": retrieved,
            }
        )

        # ====================================================
        # CHECKPOINT
        # ====================================================

        if SAVE_AFTER_EVERY_QUESTION:
            save_results(
                detailed_results=detailed_results,
                questions=questions,
                corpus_size=len(documents),
                bm25_build_time=bm25_build_time,
                rate_manager=rate_manager,
                output_path=CHECKPOINT_PATH,
                final=False,
            )

        if stopped_by_daily_quota:
            print(
                "\nDaily quota protection triggered. "
                "Stopping safely after saving the current result."
            )
            break

    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_results(
        detailed_results=detailed_results,
        questions=questions,
        corpus_size=len(documents),
        bm25_build_time=bm25_build_time,
        rate_manager=rate_manager,
        output_path=OUTPUT_PATH,
        final=True,
    )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    evaluated = sum(
        1
        for result in detailed_results
        if result.get("predicted_label") in {
            "yes",
            "no",
            "maybe",
        }
    )

    correct = sum(
        1
        for result in detailed_results
        if result.get("correct") is True
    )

    accuracy = (
        correct / evaluated
        if evaluated > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("LANGCHAIN RAG RESULTS")
    print("=" * 70)
    print(f"Questions available: {len(questions)}")
    print(f"Questions processed this run: {len(detailed_results)}")
    print(f"Successfully evaluated: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Answer accuracy: {accuracy:.4f}")
    print(
        f"Generation success rate: "
        f"{evaluated / len(questions):.4f}"
        if questions
        else "Generation success rate: 0.0000"
    )
    print(
        f"Requests used this run: "
        f"{rate_manager.total_requests:,}/{RPD_LIMIT:,}"
    )
    print(
        f"Tokens used this run: "
        f"{rate_manager.total_tokens:,}/"
        f"{DAILY_TOKEN_BUDGET:,} local safety budget"
    )
    print("\nFinal results saved to:")
    print(OUTPUT_PATH)
    print("Checkpoint saved to:")
    print(CHECKPOINT_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
