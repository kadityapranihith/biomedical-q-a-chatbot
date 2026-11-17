"""
app.py  –  BioMed RAG Q&A
=========================
Streamlit interface for the hybrid BM25 + MedCPT retrieval-augmented
generation system over the PubMed corpus.

Run:
    streamlit run app.py
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

# ============================================================
# PAGE CONFIG  (must be FIRST streamlit call)
# ============================================================

st.set_page_config(
    page_title="BioMed RAG  ·  PubMed Q&A",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CORPUS_PATH        = BASE_DIR / "data" / "pubmed_retrieval_corpus.jsonl"
MEDCPT_INDEX_PATH  = BASE_DIR / "data" / "medcpt_index" / "medcpt_abstract.index"
MEDCPT_PMIDS_PATH  = BASE_DIR / "data" / "medcpt_index" / "pmids.json"

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_GROQ_MODEL     = "openai/gpt-oss-120b"
DEFAULT_CANDIDATE_K    = 100
DEFAULT_TOP_K          = 5
DEFAULT_BM25_WEIGHT    = 0.5
DEFAULT_MEDCPT_WEIGHT  = 0.5
MEDCPT_QUERY_MODEL     = "ncbi/MedCPT-Query-Encoder"
MEDCPT_MAX_LENGTH      = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ── Google Font ───────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Root palette ──────────────────────────────────────── */
    :root {
        --bg-dark:        #0D1117;
        --bg-card:        #161B22;
        --bg-card-hover:  #1C2330;
        --border:         #30363D;
        --accent-blue:    #4361EE;
        --accent-cyan:    #4CC9F0;
        --accent-green:   #2EA043;
        --accent-amber:   #F4A261;
        --accent-pink:    #F72585;
        --text-primary:   #E6EDF3;
        --text-secondary: #8B949E;
        --text-muted:     #484F58;
        --radius:         12px;
        --radius-sm:      8px;
    }

    /* ── Page background ───────────────────────────────────── */
    .stApp {
        background: var(--bg-dark);
        color: var(--text-primary);
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* ── Header hero ───────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0f1829 50%, #0D1117 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem 2.5rem 1.8rem;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(67,97,238,.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 160px; height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(76,201,240,.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4CC9F0, #4361EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 .4rem;
    }
    .hero-sub {
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
    }
    .badge-row {
        display: flex;
        gap: .5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: .3rem;
        padding: .22rem .7rem;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: .02em;
    }
    .badge-blue  { background: rgba(67,97,238,.18);  color: #7B9CFF; border: 1px solid rgba(67,97,238,.35); }
    .badge-cyan  { background: rgba(76,201,240,.15); color: #6EDCFF; border: 1px solid rgba(76,201,240,.3); }
    .badge-green { background: rgba(46,160,67,.15);  color: #57D975; border: 1px solid rgba(46,160,67,.3);  }
    .badge-amber { background: rgba(244,162,97,.15); color: #F9C784; border: 1px solid rgba(244,162,97,.3); }

    /* ── Answer card ───────────────────────────────────────── */
    .answer-card {
        background: linear-gradient(135deg, #161B22, #0f1829);
        border: 1px solid var(--accent-blue);
        border-radius: var(--radius);
        padding: 1.5rem 1.8rem;
        margin: 1rem 0 1.4rem;
        box-shadow: 0 0 24px rgba(67,97,238,.12);
        animation: fadeIn .4s ease;
    }
    .answer-label {
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: .6rem;
    }
    .answer-verdict {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: .5rem;
    }
    .verdict-yes   { color: #2EA043; }
    .verdict-no    { color: #F85149; }
    .verdict-maybe { color: #F4A261; }
    .verdict-open  { color: #4CC9F0; }
    .answer-explanation {
        color: var(--text-primary);
        font-size: .95rem;
        line-height: 1.65;
        border-top: 1px solid var(--border);
        padding-top: .8rem;
        margin-top: .8rem;
        white-space: pre-wrap;
    }
    .latency-row {
        display: flex;
        gap: 1.2rem;
        margin-top: .9rem;
        flex-wrap: wrap;
    }
    .latency-chip {
        font-size: .73rem;
        color: var(--text-secondary);
        background: rgba(255,255,255,.04);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: .15rem .65rem;
    }

    /* ── Evidence cards ────────────────────────────────────── */
    .evidence-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 1rem 1.2rem;
        margin-bottom: .8rem;
        transition: border-color .2s, background .2s;
    }
    .evidence-card:hover {
        border-color: var(--accent-blue);
        background: var(--bg-card-hover);
    }
    .evidence-rank {
        display: inline-block;
        width: 1.5rem; height: 1.5rem;
        border-radius: 50%;
        background: var(--accent-blue);
        color: white;
        font-size: .68rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.5rem;
        margin-right: .5rem;
        flex-shrink: 0;
    }
    .evidence-title {
        font-size: .9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: .25rem;
    }
    .evidence-meta {
        font-size: .72rem;
        color: var(--text-secondary);
        margin-bottom: .5rem;
    }
    .evidence-abstract {
        font-size: .82rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* ── Score pill ────────────────────────────────────────── */
    .score-pill {
        display: inline-block;
        background: rgba(76,201,240,.12);
        color: #6EDCFF;
        border: 1px solid rgba(76,201,240,.25);
        border-radius: 999px;
        font-size: .68rem;
        font-weight: 600;
        padding: .1rem .55rem;
        margin-left: .5rem;
        vertical-align: middle;
    }

    /* ── Input styling ─────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(67,97,238,.15) !important;
    }

    /* ── Button ────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #4361EE, #7B2FBE) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: .95rem !important;
        padding: .55rem 1.8rem !important;
        transition: opacity .2s, transform .1s !important;
        box-shadow: 0 4px 15px rgba(67,97,238,.25) !important;
    }
    .stButton > button:hover {
        opacity: .88 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Spinner / status ──────────────────────────────────── */
    .status-bar {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: .7rem 1rem;
        font-size: .85rem;
        color: var(--text-secondary);
        margin: .6rem 0;
    }

    /* ── Sliders & selects ─────────────────────────────────── */
    .stSlider > div > div { background: var(--border) !important; }
    .stSelectbox > div   { background: var(--bg-card) !important; }

    /* ── Animations ────────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: .5; }
    }

    /* ── Section header ────────────────────────────────────── */
    .section-header {
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border);
        padding-bottom: .4rem;
        margin: 1.2rem 0 .8rem;
    }

    /* ── Examples ──────────────────────────────────────────── */
    .example-btn {
        display: block;
        width: 100%;
        background: rgba(255,255,255,.04);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: .5rem .8rem;
        font-size: .8rem;
        color: var(--text-secondary);
        cursor: pointer;
        text-align: left;
        transition: background .2s, border-color .2s;
        margin-bottom: .4rem;
    }
    .example-btn:hover {
        background: rgba(67,97,238,.1);
        border-color: var(--accent-blue);
        color: var(--text-primary);
    }

    /* ── Divider ───────────────────────────────────────────── */
    hr { border-color: var(--border) !important; }

    /* ── Scrollbar ─────────────────────────────────────────── */
    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# RESOURCE LOADING  (cached – runs only once per session)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_all_resources():
    """Load corpus, BM25 index, FAISS index, and MedCPT encoder."""

    # ── Corpus ──────────────────────────────────────────────
    documents = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            abstract = rec.get("abstract", "").strip()
            if not abstract:
                continue
            documents.append({
                "pmid":     str(rec["pmid"]),
                "title":    rec.get("title", ""),
                "abstract": abstract,
            })

    # ── BM25 ────────────────────────────────────────────────
    tokenized = [_tokenize(d["abstract"]) for d in documents]
    bm25 = BM25Okapi(tokenized)

    # ── FAISS ───────────────────────────────────────────────
    medcpt_index = faiss.read_index(str(MEDCPT_INDEX_PATH))
    with open(MEDCPT_PMIDS_PATH, "r", encoding="utf-8") as f:
        medcpt_pmids = json.load(f)

    # ── MedCPT query encoder ────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
    model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL)
    model.to(DEVICE)
    model.eval()

    return documents, bm25, medcpt_index, medcpt_pmids, tokenizer, model


# ============================================================
# CORE FUNCTIONS
# ============================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _encode_query(query: str, tokenizer, model) -> np.ndarray:
    enc = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=MEDCPT_MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        emb = out.last_hidden_state[:, 0, :]
    return emb.cpu().numpy().astype(np.float32)


def _min_max(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def hybrid_retrieve(
    query: str,
    bm25,
    documents: list[dict],
    medcpt_index,
    medcpt_pmids: list,
    tokenizer,
    model,
    candidate_k: int,
    top_k: int,
    bm25_weight: float,
    medcpt_weight: float,
) -> list[dict]:

    docs_by_pmid = {d["pmid"]: d for d in documents}

    # ── BM25 ────────────────────────────────────────────────
    bm25_scores_all = bm25.get_scores(_tokenize(query))
    bm25_idx = np.argsort(bm25_scores_all)[::-1][:candidate_k]
    bm25_pmids  = [documents[int(i)]["pmid"] for i in bm25_idx]
    bm25_raw    = [float(bm25_scores_all[int(i)]) for i in bm25_idx]

    # ── MedCPT ──────────────────────────────────────────────
    emb = _encode_query(query, tokenizer, model)
    mc_scores, mc_idx = medcpt_index.search(emb, candidate_k)
    mc_scores, mc_idx = mc_scores[0], mc_idx[0]

    mc_pmids, mc_raw = [], []
    for sc, idx in zip(mc_scores, mc_idx):
        if int(idx) < 0:
            continue
        mc_pmids.append(str(medcpt_pmids[int(idx)]))
        mc_raw.append(float(sc))

    # ── Fusion ───────────────────────────────────────────────
    bm25_norm = _min_max(bm25_raw)
    mc_norm   = _min_max(mc_raw)

    fused: dict[str, float] = {}
    for pmid, sc in zip(bm25_pmids, bm25_norm):
        fused[pmid] = fused.get(pmid, 0.0) + bm25_weight * float(sc)
    for pmid, sc in zip(mc_pmids, mc_norm):
        fused[pmid] = fused.get(pmid, 0.0) + medcpt_weight * float(sc)

    ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]

    results = []
    for rank, pmid in enumerate(ranked, start=1):
        doc = docs_by_pmid.get(pmid)
        if doc is None:
            continue
        results.append({
            "rank":         rank,
            "pmid":         pmid,
            "title":        doc["title"],
            "abstract":     doc["abstract"],
            "hybrid_score": round(fused[pmid], 4),
            "bm25_score":   round(
                float(bm25_norm[bm25_pmids.index(pmid)])
                if pmid in bm25_pmids else 0.0, 4),
            "medcpt_score": round(
                float(mc_norm[mc_pmids.index(pmid)])
                if pmid in mc_pmids else 0.0, 4),
        })
    return results


def build_context(docs: list[dict]) -> str:
    parts = []
    for doc in docs:
        parts.append(
            f"[Evidence {doc['rank']}]\n"
            f"Title: {doc['title']}\n"
            f"PMID: {doc['pmid']}\n"
            f"Abstract: {doc['abstract']}"
        )
    return "\n\n".join(parts)


def parse_answer(text: str) -> tuple[str | None, str]:
    """Return (verdict, explanation)."""
    if not text:
        return None, ""

    m = re.search(
        r"Answer\s*:\s*(YES|NO|MAYBE)\b",
        text, flags=re.IGNORECASE
    )
    verdict = m.group(1).upper() if m else None

    # Extract explanation
    exp_m = re.search(
        r"Explanation\s*:\s*(.*)",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    explanation = exp_m.group(1).strip() if exp_m else text.strip()

    # If no structured format, try bare verdict
    if verdict is None:
        stripped = text.strip().upper()
        if stripped in {"YES", "NO", "MAYBE"}:
            verdict = stripped

    return verdict, explanation


# ============================================================
# LLM CHAIN (cached per model + api key combination)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, api_key: str, temperature: float):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=768,
        api_key=api_key,
    )


SYSTEM_PROMPT = """\
You are BioMed RAG, an expert biomedical question-answering assistant.

You have been given a set of PubMed abstracts as evidence.

Your task:
1. Read the PubMed evidence carefully.
2. Answer the question based ONLY on the provided evidence.
3. Do NOT use outside knowledge or web search.

For YES/NO/MAYBE questions (e.g., clinical, research questions):
  Return exactly:
      Answer: YES
      Explanation: <concise, evidence-grounded explanation>

  OR:
      Answer: NO
      Explanation: <concise, evidence-grounded explanation>

  OR:
      Answer: MAYBE
      Explanation: <concise, evidence-grounded explanation citing conflicting/insufficient evidence>

For open-ended questions: provide a clear, structured answer grounded in the abstracts.

Always cite which evidence items (e.g., Evidence 1, Evidence 3) support your reasoning.
If the evidence is insufficient, state that clearly.
"""

HUMAN_PROMPT = """\
Question:
{question}

PubMed Evidence:
{context}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])

EXAMPLE_QUESTIONS = [
    "Does aspirin reduce the risk of cardiovascular events in diabetic patients?",
    "Is cognitive behavioral therapy effective for treating depression?",
    "Can statins prevent dementia in elderly patients?",
    "Does sleep deprivation impair immune function?",
    "Is there a link between gut microbiome and mental health?",
    "Do beta-blockers improve survival in heart failure patients?",
]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        "<div style='padding:.5rem 0 1rem'>"
        "<span style='font-size:1.6rem'>🧬</span>"
        "<span style='font-size:1rem; font-weight:700; margin-left:.5rem; "
        "background:linear-gradient(90deg,#4CC9F0,#4361EE);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;'>BioMed RAG</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── API Key ──────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>🔑 API Configuration</div>",
        unsafe_allow_html=True,
    )
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        placeholder="gsk_...",
        help="Loaded from .env automatically if present.",
    )
    groq_api_key = api_key_input or env_key

    model_name = st.selectbox(
        "LLM Model",
        options=[
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0,
        value=0.0, step=0.05,
        help="0 = deterministic, higher = more creative"
    )

    st.markdown("---")

    # ── Retrieval Settings ───────────────────────────────────
    st.markdown(
        "<div class='section-header'>🔍 Retrieval Settings</div>",
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Top-K documents", min_value=1, max_value=10,
        value=DEFAULT_TOP_K, step=1,
        help="Number of abstracts passed to the LLM as context."
    )
    candidate_k = st.slider(
        "Candidate pool (BM25 + MedCPT each)", min_value=10, max_value=200,
        value=DEFAULT_CANDIDATE_K, step=10,
        help="How many candidates each retriever fetches before fusion."
    )
    bm25_weight = st.slider(
        "BM25 weight", min_value=0.0, max_value=1.0,
        value=DEFAULT_BM25_WEIGHT, step=0.05,
    )
    medcpt_weight = round(1.0 - bm25_weight, 2)
    st.markdown(
        f"<div style='font-size:.8rem; color:#8B949E; margin-top:-.5rem;'>"
        f"MedCPT weight auto-set to <b>{medcpt_weight}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── System info ──────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>⚙️ System Info</div>",
        unsafe_allow_html=True,
    )
    device_label = "🟢 CUDA (GPU)" if DEVICE.type == "cuda" else "🟡 CPU"
    st.markdown(
        f"<div style='font-size:.8rem; color:#8B949E;'>"
        f"Device: <b style='color:#E6EDF3;'>{device_label}</b><br>"
        f"Encoder: <b style='color:#E6EDF3;'>ncbi/MedCPT-Query-Encoder</b><br>"
        f"Vectorstore: <b style='color:#E6EDF3;'>medcpt_abstract.index</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN AREA
# ============================================================

# ── Hero ─────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <p class="hero-title">🧬 BioMed RAG  ·  PubMed Q&A</p>
        <p class="hero-sub">
            Ask any biomedical question. The system retrieves relevant PubMed
            abstracts using hybrid BM25 + MedCPT semantic search, then
            generates a grounded answer with <strong>GPT-OSS 120B</strong> via Groq.
        </p>
        <div class="badge-row">
            <span class="badge badge-blue">⚡ BM25 Keyword</span>
            <span class="badge badge-cyan">🧠 MedCPT Semantic</span>
            <span class="badge badge-green">🔗 Score Fusion</span>
            <span class="badge badge-amber">🤖 GPT-OSS 120B</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load resources ────────────────────────────────────────────
with st.spinner("⏳ Loading corpus, BM25 index & MedCPT encoder (first run only)…"):
    try:
        documents, bm25, medcpt_index, medcpt_pmids, medcpt_tokenizer, medcpt_model = (
            load_all_resources()
        )
        corpus_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load resources: {e}")
        corpus_loaded = False

if corpus_loaded:
    st.markdown(
        f"<div style='font-size:.78rem; color:#2EA043; margin-bottom:1rem;'>"
        f"✅ Corpus ready — {len(documents):,} PubMed abstracts indexed</div>",
        unsafe_allow_html=True,
    )

# ── Example questions ─────────────────────────────────────────
with st.expander("💡 Example questions — click to load", expanded=False):
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        col = cols[i % 2]
        if col.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["prefill_question"] = q

# ── Question input ────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")
question = st.text_area(
    "Your biomedical question",
    value=prefill,
    height=110,
    placeholder=(
        "e.g. Does aspirin reduce cardiovascular risk in diabetic patients? "
        "Or ask any open-ended biomedical question…"
    ),
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([1, 6])
with col_btn:
    ask_clicked = st.button("🔬 Ask", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state.pop("last_result", None)
        st.rerun()

st.markdown("---")

# ============================================================
# QUERY PIPELINE
# ============================================================

if ask_clicked:
    q = question.strip()

    if not q:
        st.warning("⚠️ Please enter a question.")
    elif not groq_api_key:
        st.error("❌ Groq API key is required. Add it to `.env` or the sidebar.")
    elif not corpus_loaded:
        st.error("❌ Corpus not loaded — cannot answer.")
    else:
        # ── Step 1: Retrieval ──────────────────────────────────
        with st.status("🔍 Retrieving relevant PubMed abstracts…", expanded=True) as status:
            st.write(f"Running hybrid BM25 (w={bm25_weight}) + MedCPT (w={medcpt_weight}) retrieval…")
            t0 = time.perf_counter()
            retrieved = hybrid_retrieve(
                query=q,
                bm25=bm25,
                documents=documents,
                medcpt_index=medcpt_index,
                medcpt_pmids=medcpt_pmids,
                tokenizer=medcpt_tokenizer,
                model=medcpt_model,
                candidate_k=candidate_k,
                top_k=top_k,
                bm25_weight=bm25_weight,
                medcpt_weight=medcpt_weight,
            )
            retrieval_ms = (time.perf_counter() - t0) * 1000
            st.write(f"✅ Retrieved {len(retrieved)} abstracts in {retrieval_ms:.0f} ms")

            # ── Step 2: Generation ─────────────────────────────
            st.write(f"🤖 Generating answer with {model_name}…")
            context = build_context(retrieved)
            llm = get_llm(model_name, groq_api_key, temperature)
            chain = CHAT_PROMPT | llm

            t1 = time.perf_counter()
            try:
                response = chain.invoke({"question": q, "context": context})
                generation_ms = (time.perf_counter() - t1) * 1000
                raw_answer = response.content

                usage = {}
                rm = getattr(response, "response_metadata", {})
                if isinstance(rm, dict):
                    usage = rm.get("token_usage", rm.get("usage", {}))

                verdict, explanation = parse_answer(raw_answer)
                status.update(
                    label="✅ Done!",
                    state="complete",
                    expanded=False,
                )

                # Store result
                st.session_state["last_result"] = {
                    "question":      q,
                    "retrieved":     retrieved,
                    "raw_answer":    raw_answer,
                    "verdict":       verdict,
                    "explanation":   explanation,
                    "retrieval_ms":  retrieval_ms,
                    "generation_ms": generation_ms,
                    "usage":         usage,
                    "bm25_weight":   bm25_weight,
                    "medcpt_weight": medcpt_weight,
                    "top_k":         top_k,
                    "model":         model_name,
                }

            except Exception as e:
                generation_ms = (time.perf_counter() - t1) * 1000
                status.update(label="❌ Generation failed", state="error")
                st.error(f"LLM error: {e}")
                st.session_state["last_result"] = None

# ============================================================
# DISPLAY RESULTS
# ============================================================

result = st.session_state.get("last_result")

if result:

    # ── Answer card ───────────────────────────────────────────
    verdict     = result["verdict"]
    explanation = result["explanation"]

    verdict_class = {
        "YES":   "verdict-yes",
        "NO":    "verdict-no",
        "MAYBE": "verdict-maybe",
    }.get(verdict, "verdict-open")

    verdict_icon = {
        "YES":   "✅",
        "NO":    "❌",
        "MAYBE": "⚠️",
    }.get(verdict, "💬")

    verdict_display = (
        f"{verdict_icon} {verdict}"
        if verdict
        else "💬 Open Answer"
    )

    lat_ret = result["retrieval_ms"]
    lat_gen = result["generation_ms"]
    lat_tot = lat_ret + lat_gen
    usage   = result.get("usage", {})
    tok_str = ""
    if usage:
        total_tok = usage.get("total_tokens", usage.get("completion_tokens", ""))
        if total_tok:
            tok_str = f" · {total_tok} tokens used"

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">Answer</div>
            <div class="answer-verdict {verdict_class}">{verdict_display}</div>
            <div class="answer-explanation">{explanation}</div>
            <div class="latency-row">
                <span class="latency-chip">⚡ Retrieval: {lat_ret:.0f} ms</span>
                <span class="latency-chip">🤖 Generation: {lat_gen:.0f} ms</span>
                <span class="latency-chip">⏱ Total: {lat_tot:.0f} ms</span>
                <span class="latency-chip">📄 Top-{result['top_k']} docs{tok_str}</span>
                <span class="latency-chip">🔵 BM25 {result['bm25_weight']} · 🟢 MedCPT {result['medcpt_weight']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Retrieved evidence ────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📚 Retrieved Evidence</div>",
        unsafe_allow_html=True,
    )

    for doc in result["retrieved"]:
        score_color = "#2EA043" if doc["hybrid_score"] > 0.7 else (
                      "#4CC9F0" if doc["hybrid_score"] > 0.4 else "#8B949E")
        st.markdown(
            f"""
            <div class="evidence-card">
                <div style="display:flex; align-items:center; margin-bottom:.4rem;">
                    <span class="evidence-rank">{doc['rank']}</span>
                    <span class="evidence-title">{doc['title'] or 'Untitled'}</span>
                    <span class="score-pill">⚡ {doc['hybrid_score']:.3f}</span>
                </div>
                <div class="evidence-meta">
                    PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/"
                             target="_blank"
                             style="color:#4CC9F0; text-decoration:none;">
                             {doc['pmid']}
                    </a>
                    &nbsp;·&nbsp;BM25: {doc.get('bm25_score', 0):.3f}
                    &nbsp;·&nbsp;MedCPT: {doc.get('medcpt_score', 0):.3f}
                </div>
                <div class="evidence-abstract">{doc['abstract'][:600]}{"…" if len(doc['abstract']) > 600 else ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Raw LLM output (collapsed) ────────────────────────────
    with st.expander("🔎 Raw LLM output", expanded=False):
        st.code(result["raw_answer"], language=None)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:.75rem; color:#484F58; padding:.5rem 0 1.5rem;'>"
    "BioMed RAG · Hybrid BM25 + MedCPT · PubMed Corpus (43,977 abstracts) · "
    "GPT-OSS 120B via Groq"
    "</div>",
    unsafe_allow_html=True,
)
