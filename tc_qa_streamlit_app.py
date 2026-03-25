"""
TC QA Model — Streamlit Web App
================================
Advanced QA system for time-course data with RAG, multi-modal document ingestion,
multi-provider LLM support, conversation memory, and export capabilities.

Requirements (install via pip):
    streamlit>=1.35
    openai>=1.0
    anthropic
    langchain-openai
    langchain-community
    langchain-text-splitters
    chromadb
    PyPDF2
    python-docx
    Pillow
    pytesseract          # optional: OCR for images
    markdown
    pandas

Run:
    streamlit run tc_qa_streamlit_app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import csv
import io
import re
import time
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Dict

import streamlit as st

# ── LLM clients ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from anthropic import Anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

# ── Document parsers ─────────────────────────────────────────────────────────
try:
    from PyPDF2 import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    import docx as python_docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    import markdown as md_lib
    MARKDOWN_OK = True
except ImportError:
    md_lib = None
    MARKDOWN_OK = False

try:
    from duckduckgo_search import DDGS
    SEARCH_OK = True
except ImportError:
    SEARCH_OK = False

# ── RAG stack ─────────────────────────────────────────────────────────────────
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    RAG_OK = True
except ImportError:
    RAG_OK = False

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TC·QA·Agent — Time-Course Q&A Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# :root {
#     --bg:        #0a0d14;
#     --bg2:       #111520;
#     --bg3:       #161c2a;
#     --border:    #1f2a40;
#     --accent:    #00d4aa;
#     --accent2:   #3b82f6;
#     --accent3:   #f59e0b;
#     --text:      #e2e8f0;
#     --muted:     #64748b;
#     --danger:    #ef4444;
#     --radius:    10px;
#     --font-mono: 'JetBrains Mono', monospace;
# }

###
# /* ── Chat messages ──────────────────────────────── */
# .msg-user {
#     background: linear-gradient(135deg, #1e293b, #1e3a5f);
#     border: 1px solid rgba(59, 130, 246, 0.35); /* softer accent2 */
#     border-radius: var(--radius);
#     padding: 14px 18px;
#     margin: 10px 0;
#     position: relative;
#     color: var(--text2);
# }

# .msg-user::before {
#     content: "YOU";
#     font-size: 0.65rem;
#     font-weight: 700;
#     letter-spacing: 0.1em;
#     color: #60a5fa; /* softer blue */
#     display: block;
#     margin-bottom: 6px;
# }

# .msg-ai {
#     background: linear-gradient(135deg, #0f172a, #0f2a2a);
#     border: 1px solid rgba(0, 212, 170, 0.35); /* softer accent */
#     border-radius: var(--radius);
#     padding: 14px 18px;
#     margin: 10px 0;
#     color: var(--text2);
# }

# .msg-ai::before {
#     content: "TC·QA·Agent";
#     font-size: 0.65rem;
#     font-weight: 700;
#     letter-spacing: 0.1em;
#     color: #2dd4bf; /* softer teal */
#     display: block;
#     margin-bottom: 6px;
# }
###

###
# .avatar-user { background: #1e3a5f; border: 1px solid #3b82f6; }
# .avatar-ai   { background: #0f2a1a; border: 1px solid #00d4aa; }
###

st.markdown("""
<style>
/* ── Force light mode regardless of system preference ── */
html, body, [data-testid="stAppViewContainer"], 
[data-testid="stHeader"], [data-testid="stToolbar"] {
    color-scheme: light only !important;
}             
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@700;900&display=swap');

/* ── Root variables ─────────────────────────────── */

:root {
    --bg:        #f0f4f8;
    --bg2:       #e2ecf5;
    --bg3:       #d4e4f0;
    --border:    #a8c4d8;
    --accent:    #0369a1;
    --accent2:   #0891b2;
    --accent3:   #059669;
    --text:      #0c1929;
    --text2:     #ffffff;
    --muted:     #4a7090;
    --danger:    #dc2626;
    --radius:    10px;
    --font-mono: 'JetBrains Mono', monospace;
}
            
/* ── Global reset ───────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Header ─────────────────────────────────────── */
.tc-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.tc-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.tc-subtitle {
    font-size: 0.9rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Cards ──────────────────────────────────────── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    margin-bottom: 14px;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}

/* ── Chat messages ──────────────────────────────── */
/* ── Chat layout wrappers ───────────────────────── */
.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 10px 0;
}
.msg-row.user-row {
    flex-direction: row-reverse;  /* avatar on right */
}

/* ── Avatars ─────────────────────────────────────── */
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 4px;
}
.avatar-user {
    background: #e5e7eb;
    border: 1px solid rgba(59, 130, 246, 0.6);
}

.avatar-ai {
    background: #e5e7eb;
    border: 1px solid rgba(0, 212, 170, 0.6);
}

/* ── Bubbles ─────────────────────────────────────── */
.msg-user {
    background: linear-gradient(135deg, #1e293b, #1e3a5f);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 16px 4px 16px 16px;  /* pointed top-right */
    padding: 14px 18px;
    max-width: 75%;
    color: var(--text2);
}
.msg-user::before {
    content: "YOU";
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #60a5fa;
    display: block;
    margin-bottom: 6px;
}
.msg-ai {
    background: linear-gradient(135deg, #0f172a, #0f2a2a);
    border: 1px solid rgba(0, 212, 170, 0.35);
    border-radius: 4px 16px 16px 16px;  /* pointed top-left */
    padding: 14px 18px;
    max-width: 75%;
    color: var(--text2);
}
.msg-ai::before {
    content: "TC·QA·Agent";
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #2dd4bf;
    display: block;
    margin-bottom: 6px;
}

/* ✅ KEY FIX: Remove pre-wrap, enable proper markdown flow */
.msg-text {
    line-height: 1.7;
    font-size: 0.95rem;
    white-space: normal;        /* ← was pre-wrap, caused giant gaps */
    word-break: break-word;
}

/* ✅ Fix bullet lists inside chat bubbles */
.msg-text ul, .msg-text ol {
    margin: 6px 0 6px 20px;
    padding-left: 4px;
}
.msg-text li {
    margin: 2px 0;              /* ← tight spacing, no giant gaps */
    line-height: 1.6;
}

/* ✅ Fix bold/italic inside chat bubbles */
.msg-text strong { font-weight: 700; color: inherit; }
.msg-text em     { font-style: italic; }

/* ✅ Fix code blocks inside chat bubbles */
.msg-text code {
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    padding: 1px 5px;
    font-family: var(--font-mono);
    font-size: 0.85em;
}
.msg-text pre {
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
    padding: 10px 14px;
    overflow-x: auto;
    margin: 8px 0;
}

/* ✅ Fix headings inside chat bubbles */
.msg-text h1, .msg-text h2, .msg-text h3 {
    margin: 10px 0 4px;
    font-weight: 700;
    line-height: 1.3;
}            



/* ── Badges ─────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.badge-green  { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
.badge-blue   { background: #1e3a5f; color: #93c5fd; border: 1px solid #1d4ed8; }
.badge-amber  { background: #451a03; color: #fcd34d; border: 1px solid #92400e; }
.badge-red    { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }

/* ── Sidebar overrides ───────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextArea textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* ── Inputs ─────────────────────────────────────── */
.stTextArea textarea, .stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus, .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.12) !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa22, #3b82f622) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    padding: 6px 18px !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    border-color: var(--accent) !important;
}

/* ── File uploader ───────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--bg3) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Metrics ─────────────────────────────────────── */
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }

/* ── Expander ────────────────────────────────────── */
details { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
summary { color: var(--text) !important; font-weight: 600 !important; }

/* ── Selectbox / Slider ───────────────────────────── */
.stSelectbox [data-baseweb="select"] > div { background: var(--bg3) !important; border-color: var(--border) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }

/* ── Spinner ─────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Code blocks ─────────────────────────────────── */
code { background: #1a2535 !important; color: #93ddca !important; border-radius: 4px; padding: 2px 5px; font-family: var(--font-mono) !important; }
pre  { background: #111927 !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── Misc ────────────────────────────────────────── */
hr  { border-color: var(--border) !important; }
.stAlert { border-radius: 8px !important; }

/* RAG source chip */
.src-chip {
    display: inline-block;
    background: #0f2027;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 99px;
    padding: 1px 10px;
    font-size: 0.7rem;
    font-family: var(--font-mono);
    margin: 2px;
}

/* Pulse animation for thinking */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.thinking { animation: pulse 1.4s ease-in-out infinite; color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":        [],      # [{role, content, sources}]
        "documents":       [],      # [{name, text, size_kb}]
        "rag_store":       None,    # Chroma instance
        "rag_texts":       [],      # raw chunk list for stats
        "history":         [],      # plain Q&A strings
        "export_buf":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Document extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from uploaded file bytes, detecting type by extension."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        if not PDF_OK:
            return "[PyPDF2 not installed — install with: pip install PyPDF2]"
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if ext == ".docx":
        if not DOCX_OK:
            return "[python-docx not installed — install with: pip install python-docx]"
        doc = python_docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext in (".txt", ".md", ".csv", ".tsv", ".log"):
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return file_bytes.decode("latin-1", errors="replace")

    if ext == ".json":
        try:
            obj = json.loads(file_bytes)
            return json.dumps(obj, indent=2)
        except Exception:
            return file_bytes.decode("utf-8", errors="replace")

    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"):
        if not PIL_OK:
            return "[Pillow not installed — install with: pip install Pillow]"
        img = Image.open(io.BytesIO(file_bytes))
        if OCR_OK:
            return pytesseract.image_to_string(img)
        # Describe via metadata if no OCR
        return f"[Image: {img.format} {img.size[0]}×{img.size[1]}px — pytesseract not installed for OCR]"

    return "[Unsupported file type]"


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    if not SEARCH_OK:
        return "[Web search unavailable — install duckduckgo-search]"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "[No results found]"
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Summary: {r.get('body', '')}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"[Search error: {e}]"


def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract readable text from a URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        return "\n".join(lines)[:max_chars]
    except Exception as e:
        return f"[Failed to fetch URL: {e}]"
    
def generate_image(api_key: str, prompt: str, size: str, quality: str) -> str:
    """Generate an image with DALL·E 3, return the URL."""
    if not OPENAI_OK:
        return None, "OpenAI not installed."
    try:
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        return response.data[0].url, response.data[0].revised_prompt
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_store(api_key: str, documents: list) -> Optional[object]:
    """Build (or rebuild) a Chroma vector store from all loaded documents."""
    if not RAG_OK:
        return None
    if not documents:
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        all_chunks, all_meta = [], []
        for doc in documents:
            chunks = splitter.split_text(doc["text"])
            all_chunks.extend(chunks)
            all_meta.extend([{"source": doc["name"]}] * len(chunks))
        st.session_state["rag_texts"] = all_chunks
        embeddings = OpenAIEmbeddings(api_key=api_key)
        tmpdir = tempfile.mkdtemp(prefix="tc_rag_")
        store = Chroma.from_texts(all_chunks, embeddings, metadatas=all_meta, persist_directory=tmpdir)
        return store
    except Exception as e:
        st.warning(f"RAG build failed: {e}")
        return None


def retrieve_rag_context(store, query: str, k: int = 4) -> tuple[str, list]:
    """Return (context_str, list_of_source_names)."""
    if store is None:
        return "", []
    try:
        docs = store.similarity_search(query, k=k)
        sources = list({d.metadata.get("source", "?") for d in docs})
        text = "\n\n".join(d.page_content for d in docs)
        return text, sources
    except Exception:
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────
TASK_SYSTEM_PROMPTS = {
    "General QA": (
        "You are TC·QA·Agent, an expert assistant specializing in time-course data interpretation, "
        "clinical data analysis, physiological measurements, and biomedical research. "
        "Answer concisely, precisely, and with scientific rigor."
    ),
    "Clinical Analyst": (
        "You are a clinical data analyst with deep expertise in ICU data, sepsis, shock, phenotyping, physiological waveforms, "
        "labs and vital-sign time-series. Interpret trends, flag anomalies, and suggest clinical insights."
    ),
    "Code Interpreter": (
        "You are an expert Python programmer and data scientist. Analyse, debug, explain, and improve "
        "code. When providing code, wrap it in triple-backtick python blocks."
    ),
    "Summarizer": (
        "You are a precise scientific summarizer. Condense long documents or data reports into concise, "
        "well-structured summaries preserving key findings and numerical results."
    ),
    "Data Analyst": (
        "You are a quantitative data analyst. Given data-related questions, structured data or time-course data visualizations, such as waveforms, charts, and graphs, produce Python code snippets, "
        "statistical summaries, or plain-language interpretations as requested."
    ),
    "RAG Assistant": (
        "You answer questions using ONLY the retrieved context provided. If the context is insufficient, "
        "say so clearly. Cite the source document when possible."
    ),
    "Resume Analyst": (
        "You are a resume analyst. Given a resume and a job description, identify key skills, experience, and "
        "qualifications. Provide a concise summary of the candidate's fit for the role, highlighting strengths and potential gaps."
        "Generate the updated ATS-friendly resume with improvements in a plain text format (use bold and italics, if needed)."
    ),
    "Image Generator": (
        "You are an expert at writing detailed, vivid DALL·E image generation prompts. "
        "When the user describes an image, enhance their description into a detailed prompt "
        "and confirm what image will be generated."
    ),
}



def call_llm(
    provider: str,
    api_key: str,
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int = 2048,
) -> str:
    """Unified LLM call supporting OpenAI and Anthropic (Claude)."""
    if provider == "OpenAI" and OPENAI_OK:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()

    if provider == "Anthropic (Claude)" and ANTHROPIC_OK:
        client = Anthropic(api_key=api_key)
        # Strip system message for Anthropic format
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs  = [m for m in messages if m["role"] != "system"]
        # Convert to anthropic format
        anthropic_msgs = [{"role": m["role"], "content": m["content"]} for m in user_msgs]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_msg,
            messages=anthropic_msgs,
            temperature=temperature,
        )
        return resp.content[0].text.strip()

    raise RuntimeError(f"Provider '{provider}' unavailable. Check installation and API key.")


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def export_conversation(messages: list, fmt: str) -> bytes:
    if fmt == "JSON":
        return json.dumps(messages, indent=2, ensure_ascii=False).encode("utf-8")
    if fmt == "TXT":
        lines = []
        for m in messages:
            role = "You" if m["role"] == "user" else "TC·QA·Agent"
            lines.append(f"[{role}]\n{m['content']}\n")
        return "\n".join(lines).encode("utf-8")
    if fmt == "CSV":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["role", "content"])
        w.writeheader()
        w.writerows(messages)
        return buf.getvalue().encode("utf-8")
    if fmt == "Markdown":
        lines = ["# TC·QA·Agent Conversation Export\n"]
        for m in messages:
            role = "**You**" if m["role"] == "user" else "**TC·QA·Agent**"
            lines.append(f"{role}\n\n{m['content']}\n\n---\n")
        return "\n".join(lines).encode("utf-8")
    return b""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
#background:linear-gradient(135deg,#00d4aa,#3b82f6);
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 12px;'>
        <div style='font-family:"Playfair Display",serif;font-size:2.6rem;font-weight:900;
                    background: linear-gradient(135deg, #f59e0b, #ef4444, #ec4899);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>TC·QA·Agent</div>
        <div style='font-size:0.7rem;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;'>
            Time-Course Q&A Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Provider & Model ──────────────────────────────────────────────────────
    st.markdown('<div class="card-label">🤖 Provider & Model</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "LLM Provider", ["OpenAI", "Anthropic (Claude)"], label_visibility="collapsed"
    )
    provider_models = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "dall-e-3"],
        "Anthropic (Claude)": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    }
    model = st.selectbox("Model", provider_models[provider], label_visibility="collapsed")
    api_key = st.text_input("API Key", type="password", placeholder="sk-… or sk-ant-…")

    # ── Task Mode ─────────────────────────────────────────────────────────────
    st.markdown('<div class="card-label" style="margin-top:14px;">⚙️ Task Mode</div>', unsafe_allow_html=True)
    task_mode = st.selectbox("Task", list(TASK_SYSTEM_PROMPTS.keys()), label_visibility="collapsed")

    # ── Temperature ───────────────────────────────────────────────────────────
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                            help="0 = precise/factual  ·  0.7+ = creative")

    # ── Max tokens ────────────────────────────────────────────────────────────
    max_tokens = st.slider("Max tokens", 256, 4096, 1024, 128)

    # # ── RAG ───────────────────────────────────────────────────────────────────
    # st.markdown("---")
    # st.markdown('<div class="card-label">📚 RAG Settings</div>', unsafe_allow_html=True)
    # use_rag = st.toggle("Enable RAG", value=False, disabled=not RAG_OK,
    #                     help="Retrieval-Augmented Generation using uploaded documents")
    # if not RAG_OK:
    #     st.caption("⚠️ Install langchain-openai, langchain-community, chromadb to enable RAG")

    # # ✅ FIX 1: Auto-build RAG index if toggled ON and docs exist but store is not yet built
    # if use_rag and RAG_OK and api_key and st.session_state["documents"] and st.session_state["rag_store"] is None:
    #     with st.spinner("Building RAG index from loaded documents…"):
    #         st.session_state["rag_store"] = build_rag_store(api_key, st.session_state["documents"])
    #     if st.session_state["rag_store"]:
    #         st.success("🔎 RAG index ready.")
    #     else:
    #         st.warning("RAG index build failed — check your OpenAI API key.")

    # rag_k = st.slider("Top-k chunks", 1, 8, 4, 1, disabled=not use_rag)



    ########
    # ── RAG ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">📚 RAG Settings</div>', unsafe_allow_html=True)
    use_rag = st.toggle("Enable RAG", value=False, disabled=not RAG_OK,
                        help="Retrieval-Augmented Generation using uploaded documents")
    if not RAG_OK:
        st.caption("⚠️ Install langchain-openai, langchain-community, chromadb to enable RAG")

    rag_k = st.slider("Top-k chunks", 1, 8, 4, 1, disabled=not use_rag)  # ← belongs here

    # ✅ Auto-build RAG index if toggled ON and docs exist but store not yet built
    if use_rag and RAG_OK and api_key and st.session_state["documents"] and st.session_state["rag_store"] is None:
        with st.spinner("Building RAG index from loaded documents…"):
            st.session_state["rag_store"] = build_rag_store(api_key, st.session_state["documents"])
        if st.session_state["rag_store"]:
            st.success("🔎 RAG index ready.")
        else:
            st.warning("RAG index build failed — check your OpenAI API key.")

    # ── Web Search ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">🌐 Web Search</div>', unsafe_allow_html=True)
    use_web_search = st.toggle("Enable Web Search", value=False, disabled=not SEARCH_OK,
                               help="Automatically search the web to answer questions")
    if not SEARCH_OK:
        st.caption("⚠️ Install duckduckgo-search to enable web search")
    web_search_k = st.slider("Max results", 1, 8, 4, 1, disabled=not use_web_search)

    # ── Image Generation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">🎨 Image Generation</div>', unsafe_allow_html=True)
    img_size = st.selectbox("Size", ["1024x1024", "1792x1024", "1024x1792"],
                            label_visibility="collapsed")
    img_quality = st.selectbox("Quality", ["standard", "hd"],
                            label_visibility="collapsed")
    #########

    # ← sidebar ends here (dedent out of `with st.sidebar:`)


    # ── Memory ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">🧠 Conversation Memory</div>', unsafe_allow_html=True)
    carry_history = st.toggle("Carry conversation history", value=True)
    max_history   = st.slider("Max history turns", 1, 20, 6, 1, disabled=not carry_history)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">💾 Export</div>', unsafe_allow_html=True)
    export_fmt = st.selectbox("Format", ["JSON", "TXT", "CSV", "Markdown"], label_visibility="collapsed")
    if st.button("⬇ Export Conversation"):
        if st.session_state["messages"]:
            data = export_conversation(st.session_state["messages"], export_fmt)
            st.download_button(
                label=f"Download .{export_fmt.lower()}",
                data=data,
                file_name=f"tc_qa_agent_export.{export_fmt.lower()}",
                mime="text/plain",
            )
        else:
            st.warning("No conversation to export yet.")

    # ── Clear ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑 Clear Conversation"):
        st.session_state["messages"] = []
        st.rerun()
    if st.button("🗑 Clear Documents & RAG"):
        st.session_state["documents"] = []
        st.session_state["rag_store"]  = None
        st.session_state["rag_texts"]  = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="tc-header">
#     <div class="tc-logo">TC·QA·Agent</div>
#     <div class="tc-subtitle">Time-Course Data Intelligence Platform</div>
# </div>
# """, unsafe_allow_html=True)

st.markdown("""
<div class="tc-header">
    <div class="tc-logo" style="font-family:'Playfair Display',serif;font-size:3.6rem;font-weight:900;">TC·QA·Agent</div>
    <div class="tc-subtitle" style="font-size:1.5rem;">Time-Course Data Intelligence Platform</div>
</div>
<p style="font-size:1.1rem; color:var(--muted); margin:-10px 0 20px 0; letter-spacing:0.01em;">
    A web-based tool with advanced Q&A systems for interpreting text, data, code, images and documents using LLMs 
    with 8 expert assistants accessible via <strong style="color:var(--accent);">Task Mode</strong>, plus RAG, multi-modal document ingestion, web search,
    multi-provider LLM support, conversation memory, and export capabilities.
</p>

<div style="margin-bottom:18px;">
    <div style="font-size:1.2rem; font-weight:700; margin-bottom:6px; color:var(--text);">Assistants</div>
    <div class="card" style="padding:10px;">
        <ul style="margin:0; padding-left:20px; color:var(--muted);">
            <li><strong>General QA — </strong> Ask anything — general knowledge, science, concepts, or open-ended questions.</li>
            <li><strong>Clinical Analyst — </strong> Interpret healthcare data, vital signs, lab trends, diseases, and physiological time-series.</li>
            <li><strong>Code Interpreter — </strong> Explain, debug, optimize, or rewrite code snippets in any language.</li>
            <li><strong>Summarizer — </strong> Condense long documents, papers, or reports into concise structured summaries.</li>
            <li><strong>Data Analyst — </strong> Analyze structured data, generate Python code, and extract statistical insights.</li>
            <li><strong>RAG Assistant — </strong> Answer questions grounded strictly in your uploaded documents.</li>
            <li><strong>Resume Analyst — </strong> Review, critique, and improve resumes for clarity, impact, and ATS compatibility.</li>
            <li><strong>Image Generator — </strong> Create detailed DALL·E prompts to generate vivid images from your descriptions.</li>
        </ul>
    </div>
</div>

""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_context, tab_stats = st.tabs(
    ["💬 Chat", "📂 Documents", "📝 Context", "📊 Stats"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # ── Status bar ───────────────────────────────────────────────────────────
    status_cols = st.columns([1, 1, 1, 2])
    with status_cols[0]:
        provider_ok = (provider == "OpenAI" and OPENAI_OK) or \
                      (provider == "Anthropic (Claude)" and ANTHROPIC_OK)
        badge = "badge-green" if (api_key and provider_ok) else "badge-red"
        label = "Connected" if (api_key and provider_ok) else "No API Key"
        st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)
    with status_cols[1]:
        has_docs = bool(st.session_state["documents"])
        rag_badge = (
            "badge-green" if (use_rag and st.session_state["rag_store"]) else
            "badge-amber" if (use_rag and has_docs and not st.session_state["rag_store"]) else
            "badge-amber" if (use_rag and not has_docs) else
            "badge-blue"
        )
        rag_label = (
            "RAG Active" if (use_rag and st.session_state["rag_store"]) else
            "RAG: No Index" if (use_rag and has_docs) else
            "RAG: No Docs"  if use_rag else
            "RAG Off"
        )
        st.markdown(f'<span class="badge {rag_badge}">{rag_label}</span>', unsafe_allow_html=True)
    with status_cols[2]:
        n_docs = len(st.session_state["documents"])
        st.markdown(f'<span class="badge badge-blue">{n_docs} doc{"s" if n_docs != 1 else ""}</span>',
                    unsafe_allow_html=True)
    with status_cols[3]:
        st.caption(f"Model: `{model}` · Mode: **{task_mode}** · T={temperature}")

    st.markdown("---")

    
    # # ── Chat history ─────────────────────────────────────────────────────────
    # chat_container = st.container()
    # with chat_container:
    #     for msg in st.session_state["messages"]:
    #         if msg["role"] == "user":
    #             with st.chat_message("user", avatar="👤"):
    #                 st.markdown(msg["content"])
    #         else:
    #             with st.chat_message("assistant", avatar="🤖"):
    #                 # Show RAG source chips if any
    #                 sources = msg.get("sources", [])
    #                 if sources:
    #                     src_html = " ".join(
    #                         f'<span class="src-chip">📄 {s}</span>' for s in sources
    #                     )
    #                     st.markdown(src_html, unsafe_allow_html=True)
    #                 st.markdown(msg["content"])
    
    # # ── Chat history ─────────────────────────────────────────────────────────
    # chat_container = st.container()
    # with chat_container:
    #     for msg in st.session_state["messages"]:
    #         if msg["role"] == "user":
    #             st.markdown(
    #                 f'<div class="msg-user"><div class="msg-text">{msg["content"]}</div></div>',
    #                 unsafe_allow_html=True,
    #             )
    #         else:
    #             src_html = ""
    #             for s in msg.get("sources", []):
    #                 src_html += f'<span class="src-chip">📄 {s}</span>'
    #             st.markdown(
    #                 f'<div class="msg-ai">{src_html + "<br>" if src_html else ""}'
    #                 f'<div class="msg-text">',
    #                 unsafe_allow_html=True,
    #             )
    #             st.markdown(msg["content"])   # ← native markdown: renders bullets, bold, code correctly
    #             st.markdown('</div></div>', unsafe_allow_html=True)
                
    #             # st.markdown(
    #             #     f'<div class="msg-ai">'
    #             #     f'{src_html + "<br>" if src_html else ""}'
    #             #     f'<div class="msg-text">{msg["content"]}</div>'
    #             #     f'</div>',
    #             #     unsafe_allow_html=True,
    #             # )

    # ── Chat history ─────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-row user-row">'
                    f'  <div class="avatar avatar-user">👤</div>'
                    f'  <div class="msg-user"><div class="msg-text">{msg["content"]}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                src_html = ""
                for s in msg.get("sources", []):
                    src_html += f'<span class="src-chip">📄 {s}</span>'
                st.markdown(
                    f'<div class="msg-row">'
                    f'  <div class="avatar avatar-ai">🧬</div>'
                    f'  <div class="msg-ai">'
                    f'    {src_html + "<br>" if src_html else ""}'
                    f'    <div class="msg-text">',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                st.markdown('</div></div></div>', unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown("")
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "Ask a question…",
            placeholder="e.g., What does the blood pressure reading indicate?\n"
                        "Or paste code to interpret, data to summarise, etc.",
            height=100,
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([1, 5])
        with col_send:
            submitted = st.form_submit_button("▶ Send", type="primary", use_container_width=True)

    if submitted and user_query.strip():
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
        else:
            # Append user message
            st.session_state["messages"].append({"role": "user", "content": user_query.strip()})

            with st.spinner(""):
                st.markdown('<p class="thinking">TC·QA·Agent is thinking…</p>', unsafe_allow_html=True)

                # ── Build system prompt ─────────────────────────────────────
                system_content = TASK_SYSTEM_PROMPTS[task_mode]

                # Prepend any manual context
                manual_ctx = st.session_state.get("manual_context", "").strip()
                if manual_ctx:
                    system_content += f"\n\n--- ADDITIONAL CONTEXT ---\n{manual_ctx}"

                # ✅ FIX 2a: When RAG is OFF, inject full document text directly into system prompt
                rag_sources = []
                if not use_rag and st.session_state["documents"]:
                    doc_blocks = []
                    for doc in st.session_state["documents"]:
                        # Truncate very large docs to avoid token overflow (~4000 chars each)
                        preview = doc["text"][:4000]
                        if len(doc["text"]) > 4000:
                            preview += "\n\n[... document truncated for length ...]"
                        doc_blocks.append(f"[Document: {doc['name']}]\n{preview}")
                    system_content += (
                        "\n\n--- UPLOADED DOCUMENTS (use these to answer the user's question) ---\n"
                        + "\n\n".join(doc_blocks)
                    )

                # ✅ FIX 2b: When RAG is ON, retrieve relevant chunks — also fallback-inject if retrieval returns nothing
                if use_rag:
                    if st.session_state["rag_store"]:
                        rag_ctx, rag_sources = retrieve_rag_context(
                            st.session_state["rag_store"], user_query.strip(), k=rag_k
                        )
                        if rag_ctx:
                            system_content += f"\n\n--- RETRIEVED CONTEXT (RAG) ---\n{rag_ctx}"
                        else:
                            # Fallback: RAG returned nothing, inject raw text instead
                            for doc in st.session_state["documents"]:
                                system_content += f"\n\n[Document: {doc['name']}]\n{doc['text'][:3000]}"
                    else:
                        # RAG enabled but index not built yet — inject raw docs as fallback
                        for doc in st.session_state["documents"]:
                            system_content += f"\n\n[Document: {doc['name']}]\n{doc['text'][:3000]}"
                        st.warning("⚠️ RAG index not built yet — using raw document text as fallback. "
                                   "Go to the Documents tab and click 'Rebuild RAG Index'.")

                
                # ── Web search and URL fetch ──────────────────────────────
                if use_web_search:
                    with st.spinner("🌐 Searching the web…"):
                        search_results = web_search(user_query.strip(), max_results=web_search_k)
                    system_content += f"\n\n--- WEB SEARCH RESULTS ---\n{search_results}"
                    urls = re.findall(r'https?://[^\s\)\"\']+', user_query)
                    for url in urls[:2]:
                        page_text = fetch_url(url)
                        system_content += f"\n\n--- FETCHED PAGE: {url} ---\n{page_text}"

                # ── Image generation (DALL·E 3) ───────────────────────────
                if model == "dall-e-3" or task_mode == "Image Generator":
                    with st.spinner("🎨 Generating image…"):
                        img_url, revised_prompt = generate_image(
                            api_key, user_query.strip(), img_size, img_quality
                        )
                    if img_url:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": f"**Revised prompt used:**\n_{revised_prompt}_\n\n![Generated Image]({img_url})\n\n[⬇ Download image]({img_url})",
                            "sources": [],
                            "image_url": img_url,
                        })
                    else:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": f"❌ Image generation failed: {revised_prompt}",
                            "sources": [],
                        })
                    st.session_state["history"].append(f"Q: {user_query.strip()}\nA: [Image generated]")

                else:  # ← normal LLM path, only runs if NOT image generation
                    # ── Build messages list ───────────────────────────────
                    chat_messages = [{"role": "system", "content": system_content}]

                    if carry_history:
                        history = st.session_state["messages"][-(max_history * 2 + 1):-1]
                        for h in history:
                            chat_messages.append({"role": h["role"], "content": h["content"]})

                    chat_messages.append({"role": "user", "content": user_query.strip()})

                    # ── Call LLM ──────────────────────────────────────────
                    try:
                        answer = call_llm(
                            provider=provider,
                            api_key=api_key,
                            model=model,
                            messages=chat_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "sources": rag_sources,
                        })
                        st.session_state["history"].append(
                            f"Q: {user_query.strip()}\nA: {answer}"
                        )
                    except Exception as e:
                        err = f"**Error calling {provider}:** {e}\n\n```\n{traceback.format_exc()}\n```"
                        st.session_state["messages"].append({
                            "role": "assistant", "content": err, "sources": []
                        })

            st.rerun()  # ← single rerun at the end covers both paths


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown('<div class="card-label">📂 Upload Documents for RAG & Context</div>',
                unsafe_allow_html=True)
    st.caption(
        "Supported: PDF · DOCX · TXT · MD · CSV · JSON · PNG / JPG (OCR) · "
        "TIFF · BMP · WEBP"
    )

    uploaded_files = st.file_uploader(
        "Drop files here or browse",
        type=["pdf", "docx", "txt", "md", "csv", "tsv", "json", "log",
              "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_names = {d["name"] for d in st.session_state["documents"]}
        added = 0
        for f in uploaded_files:
            if f.name in new_names:
                continue
            with st.spinner(f"Extracting text from **{f.name}**…"):
                raw = f.read()
                text = extract_text_from_file(raw, f.name)
            st.session_state["documents"].append({
                "name": f.name,
                "text": text,
                "size_kb": round(len(raw) / 1024, 1),
                "chars": len(text),
            })
            new_names.add(f.name)
            added += 1

        if added:
            st.success(f"✅ Added {added} document(s).")
            # ✅ FIX 3: Always rebuild RAG store on new upload when RAG is enabled and api_key is present
            if use_rag and RAG_OK and api_key:
                with st.spinner("Building vector index…"):
                    st.session_state["rag_store"] = build_rag_store(
                        api_key, st.session_state["documents"]
                    )
                if st.session_state["rag_store"]:
                    st.success("🔎 RAG index built successfully.")
                else:
                    st.warning("RAG index build failed — check OpenAI key and langchain install.")
            elif use_rag and not api_key:
                st.info("ℹ️ Documents loaded. Enter your API key in the sidebar to build the RAG index.")

    # ── Loaded documents table ────────────────────────────────────────────────
    if st.session_state["documents"]:
        st.markdown("---")
        st.markdown(f'<div class="card-label">Loaded Documents ({len(st.session_state["documents"])})</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame([
            {"Name": d["name"], "Size (KB)": d["size_kb"], "Extracted chars": d["chars"]}
            for d in st.session_state["documents"]
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Preview
        preview_name = st.selectbox(
            "Preview document text", [d["name"] for d in st.session_state["documents"]]
        )
        if preview_name:
            preview_doc = next(d for d in st.session_state["documents"] if d["name"] == preview_name)
            with st.expander("📄 Text Preview (first 3000 chars)", expanded=False):
                st.text(preview_doc["text"][:3000])

        # Manual RAG rebuild
        if use_rag and RAG_OK and api_key:
            if st.button("🔄 Rebuild RAG Index"):
                with st.spinner("Rebuilding vector index…"):
                    st.session_state["rag_store"] = build_rag_store(
                        api_key, st.session_state["documents"]
                    )
                st.success("RAG index rebuilt.")
    else:
        st.info("No documents loaded yet. Upload files above to enable document-grounded QA.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MANUAL CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab_context:
    st.markdown('<div class="card-label">📝 Manual Context / Paste Data</div>',
                unsafe_allow_html=True)
    st.caption(
        "Paste any text, time-series data, code, or clinical notes here. "
        "This will be included as system context in every message."
    )

    ctx_placeholder = (
        "Example — paste a time-course table:\n\n"
        "Time(h)  HR(bpm)  MAP(mmHg)  Lactate(mmol/L)  Temp(°C)\n"
        "0        88       72         1.2              37.1\n"
        "6        102      65         2.1              38.4\n"
        "12       118      58         3.8              39.1\n"
        "24       128      51         5.2              39.6\n\n"
        "Or paste code, reports, or any reference material…"
    )
    manual_context = st.text_area(
        "Context", placeholder=ctx_placeholder, height=320, label_visibility="collapsed"
    )
    st.session_state["manual_context"] = manual_context

    if manual_context.strip():
        st.markdown(
            f'<span class="badge badge-green">✓ {len(manual_context):,} characters of context ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge badge-amber">No manual context</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-label">💡 Quick Prompt Templates</div>', unsafe_allow_html=True)
    templates = {
        "Sepsis time-course": (
            "Patient ID: ICU-042\n"
            "Diagnosis: Septic shock (gram-negative)\n\n"
            "Time(h)  HR   MAP  Lactate  WBC    Temp\n"
            "0        88   72   1.2      12.4   37.1\n"
            "6        102  65   2.1      18.1   38.4\n"
            "12       118  58   3.8      22.3   39.1\n"
            "24       128  51   5.2      28.6   39.6\n"
        ),
        "Python code snippet": (
            "import pandas as pd\nimport numpy as np\n\n"
            "def compute_rolling_mean(series, window=5):\n"
            "    return pd.Series(series).rolling(window).mean().tolist()\n"
        ),
        "Lab results": (
            "Patient: John Doe, 67M, Admitted: 2024-01-15\n\n"
            "Test         Day1   Day3   Day7   RefRange\n"
            "Creatinine   1.1    2.4    3.8    0.7-1.2 mg/dL\n"
            "BUN          18     42     67     7-20 mg/dL\n"
            "eGFR         72     31     18     >60 mL/min\n"
            "Urine output  1200   650    280    >500 mL/d\n"
        ),
    }
    for tname, ttext in templates.items():
        if st.button(f"Use: {tname}"):
            st.session_state["manual_context"] = ttext
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown('<div class="card-label">📊 Session Statistics</div>', unsafe_allow_html=True)

    msgs = st.session_state["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    ai_msgs   = [m for m in msgs if m["role"] == "assistant"]
    docs      = st.session_state["documents"]
    chunks    = st.session_state["rag_texts"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turns", len(user_msgs))
    m2.metric("Documents", len(docs))
    m3.metric("RAG Chunks", len(chunks))
    total_chars = sum(d["chars"] for d in docs) if docs else 0
    m4.metric("Total chars indexed", f"{total_chars:,}")

    if msgs:
        st.markdown("---")
        st.markdown('<div class="card-label">💬 Full Conversation History</div>', unsafe_allow_html=True)
        history_df = pd.DataFrame([
            {"#": i + 1, "Role": m["role"], "Message (preview)": m["content"][:120] + "…"}
            for i, m in enumerate(msgs)
        ])
        st.dataframe(history_df, use_container_width=True, hide_index=True)

    if docs:
        st.markdown("---")
        st.markdown('<div class="card-label">📂 Document Breakdown</div>', unsafe_allow_html=True)
        df_docs = pd.DataFrame([
            {"Name": d["name"], "KB": d["size_kb"], "Chars": d["chars"]}
            for d in docs
        ])
        st.bar_chart(df_docs.set_index("Name")["Chars"])

    # ── Dependency status ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">🔧 Dependency Status</div>', unsafe_allow_html=True)
    dep_data = {
        "openai":              OPENAI_OK,
        "anthropic":           ANTHROPIC_OK,
        "PyPDF2":              PDF_OK,
        "python-docx":         DOCX_OK,
        "Pillow":              PIL_OK,
        "pytesseract (OCR)":   OCR_OK,
        "langchain + chromadb (RAG)": RAG_OK,
    }
    for dep, ok in dep_data.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} `{dep}`")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#334155;font-size:0.75rem;padding:8px 0;">'
    'Copyright © 2026 Tilendra Choudhary · TC·QA·Agent · Time-Course Intelligence Platform with 8 Assistants'
    '</div>',
    unsafe_allow_html=True,
)
