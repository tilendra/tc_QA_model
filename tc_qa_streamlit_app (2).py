"""
TC·QA·Agent — Streamlit Web App (Final Version)
================================================
Features:
  - 8 specialist task modes + MultiAgent orchestrator
  - RAG (Retrieval-Augmented Generation) with Chroma
  - Multi-modal document ingestion (PDF, DOCX, TXT, MD, CSV, JSON, images/OCR)
  - Web search via ddgs
  - DALL·E 3 image generation
  - Auto-state rules (Image Generator → dall-e-3, RAG Assistant → toggles ON)
  - Agent dialogue exchange panel (shows agents chatting each other)
  - Conversation memory, export, and session stats

Run:
    streamlit run tc_qa_streamlit_app.py
"""

import os, json, csv, io, re, time, tempfile, traceback
from pathlib import Path
from typing import List, Optional, Dict
import streamlit as st

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
    from ddgs import DDGS
    SEARCH_OK = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        SEARCH_OK = True
    except ImportError:
        SEARCH_OK = False

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    RAG_OK = True
except ImportError:
    RAG_OK = False

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TC·QA·Agent — Time-Course Q&A Agent",
    page_icon="🤖", layout="wide", initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] { color-scheme: light only !important; }
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@700;900&display=swap');
:root {
    --bg:#f0f4f8; --bg2:#e2ecf5; --bg3:#d4e4f0; --border:#a8c4d8;
    --accent:#0369a1; --accent2:#0891b2; --accent3:#059669;
    --text:#0c1929; --text2:#ffffff; --muted:#4a7090; --danger:#dc2626;
    --radius:10px; --font-mono:'JetBrains Mono',monospace;
}
html,body,[class*="css"] { background-color:var(--bg)!important; color:var(--text)!important; font-family:'Space Grotesk',sans-serif!important; }
::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-track{background:var(--bg)} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.tc-header{display:flex;align-items:baseline;gap:16px;padding:28px 0 20px;border-bottom:1px solid var(--border);margin-bottom:24px;}
.tc-logo{font-family:'Playfair Display',serif;font-size:3.6rem;font-weight:900;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
.tc-subtitle{font-size:1.5rem;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;margin-bottom:14px;}
.card-label{font-size:0.72rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);margin-bottom:8px;}
.msg-row{display:flex;align-items:flex-start;gap:10px;margin:10px 0;}
.msg-row.user-row{flex-direction:row-reverse;}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;margin-top:4px;}
.avatar-user{background:#e5e7eb;border:1px solid rgba(59,130,246,0.6);}
.avatar-ai{background:#e5e7eb;border:1px solid rgba(0,212,170,0.6);}
.msg-user{background:linear-gradient(135deg,#1e293b,#1e3a5f);border:1px solid rgba(59,130,246,0.35);border-radius:16px 4px 16px 16px;padding:14px 18px;max-width:75%;color:var(--text2);}
.msg-user::before{content:"YOU";font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#60a5fa;display:block;margin-bottom:6px;}
.msg-ai{background:linear-gradient(135deg,#0f172a,#0f2a2a);border:1px solid rgba(0,212,170,0.35);border-radius:4px 16px 16px 16px;padding:14px 18px;max-width:75%;color:var(--text2);}
.msg-ai::before{content:"TC·QA·Agent";font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#2dd4bf;display:block;margin-bottom:6px;}
.msg-text{line-height:1.7;font-size:0.95rem;white-space:normal;word-break:break-word;}
.msg-text ul,.msg-text ol{margin:6px 0 6px 20px;padding-left:4px;}
.msg-text li{margin:2px 0;line-height:1.6;}
.msg-text strong{font-weight:700;color:inherit;} .msg-text em{font-style:italic;}
.msg-text code{background:rgba(255,255,255,0.08);border-radius:4px;padding:1px 5px;font-family:var(--font-mono);font-size:0.85em;}
.msg-text pre{background:rgba(0,0,0,0.3);border-radius:6px;padding:10px 14px;overflow-x:auto;margin:8px 0;}
.msg-text h1,.msg-text h2,.msg-text h3{margin:10px 0 4px;font-weight:700;line-height:1.3;}
/* Agent dialogue panel */
.dialogue-panel{background:#0a0f1a;border:1px solid #1e3a5f;border-radius:12px;padding:16px;margin:12px 0;font-family:var(--font-mono);font-size:0.82rem;}
.dialogue-title{color:#2dd4bf;font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #1e3a5f;}
.d-msg{display:flex;gap:10px;margin:8px 0;align-items:flex-start;}
.d-msg.right{flex-direction:row-reverse;}
.d-avatar{width:32px;height:32px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:0.9rem;font-weight:700;}
.d-bubble{max-width:80%;padding:8px 12px;border-radius:10px;font-size:0.78rem;line-height:1.5;color:#e2e8f0;}
.d-name{font-size:0.62rem;font-weight:700;letter-spacing:0.08em;margin-bottom:3px;}
.d-arrow{text-align:center;color:#334155;font-size:0.7rem;padding:2px 0;letter-spacing:0.1em;}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:0.7rem;font-weight:600;letter-spacing:0.06em;}
.badge-green{background:#064e3b;color:#6ee7b7;border:1px solid #065f46;}
.badge-blue{background:#1e3a5f;color:#93c5fd;border:1px solid #1d4ed8;}
.badge-amber{background:#451a03;color:#fcd34d;border:1px solid #92400e;}
.badge-red{background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d;}
.badge-purple{background:#2e1065;color:#c4b5fd;border:1px solid #6d28d9;}
[data-testid="stSidebar"]{background:var(--bg)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .stTextInput>div>div>input,[data-testid="stSidebar"] .stSelectbox>div>div,[data-testid="stSidebar"] .stTextArea textarea{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-family:var(--font-mono)!important;font-size:0.82rem!important;}
.stTextArea textarea,.stTextInput>div>div>input{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-size:0.9rem!important;}
.stTextArea textarea:focus,.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(0,212,170,0.12)!important;}
.stButton>button{background:linear-gradient(135deg,#00d4aa22,#3b82f622)!important;border:1px solid var(--accent)!important;color:var(--accent)!important;border-radius:6px!important;font-weight:600!important;font-size:0.85rem!important;letter-spacing:0.04em!important;transition:all 0.2s!important;padding:6px 18px!important;}
.stButton>button:hover{background:var(--accent)!important;color:#000!important;transform:translateY(-1px)!important;}
.stButton>button[kind="primary"]{background:var(--accent)!important;color:#000!important;border-color:var(--accent)!important;}
[data-testid="stFileUploader"]{background:var(--bg3)!important;border:2px dashed var(--border)!important;border-radius:var(--radius)!important;padding:10px!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;border-bottom:1px solid var(--border)!important;gap:4px!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-weight:600!important;font-size:0.82rem!important;letter-spacing:0.06em!important;text-transform:uppercase!important;border-radius:6px 6px 0 0!important;padding:8px 16px!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;background:var(--bg3)!important;border-bottom:2px solid var(--accent)!important;}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-family:var(--font-mono)!important;font-size:1.6rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.75rem!important;}
details{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
summary{color:var(--text)!important;font-weight:600!important;}
.stSelectbox [data-baseweb="select"]>div{background:var(--bg3)!important;border-color:var(--border)!important;}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--accent)!important;}
.stSpinner>div{border-top-color:var(--accent)!important;}
code{background:#1a2535!important;color:#93ddca!important;border-radius:4px;padding:2px 5px;font-family:var(--font-mono)!important;}
pre{background:#111927!important;border:1px solid var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;} .stAlert{border-radius:8px!important;}
.src-chip{display:inline-block;background:#0f2027;border:1px solid var(--accent);color:var(--accent);border-radius:99px;padding:1px 10px;font-size:0.7rem;font-family:var(--font-mono);margin:2px;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.thinking{animation:pulse 1.4s ease-in-out infinite;color:var(--accent);}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [], "documents": [], "rag_store": None,
        "rag_texts": [], "history": [], "export_buf": None,
        "manual_context": "", "auto_model": None,
        "auto_rag": None, "auto_search": None, "agent_dialogues": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Agent colour palette
# ─────────────────────────────────────────────────────────────────────────────
AGENT_COLORS = {
    "Orchestrator":    {"bg": "#1e3a5f", "text": "#93c5fd", "emoji": "🎯"},
    "General QA":      {"bg": "#064e3b", "text": "#6ee7b7", "emoji": "💬"},
    "Clinical Analyst":{"bg": "#450a0a", "text": "#fca5a5", "emoji": "🩺"},
    "Code Interpreter":{"bg": "#1e1b4b", "text": "#c4b5fd", "emoji": "💻"},
    "Summarizer":      {"bg": "#3b1f00", "text": "#fcd34d", "emoji": "📋"},
    "Data Analyst":    {"bg": "#0c2340", "text": "#7dd3fc", "emoji": "📊"},
    "RAG Assistant":   {"bg": "#162032", "text": "#a5f3fc", "emoji": "📚"},
    "Resume Analyst":  {"bg": "#1a0533", "text": "#e9d5ff", "emoji": "📄"},
    "Image Generator": {"bg": "#1f1108", "text": "#fdba74", "emoji": "🎨"},
    "User":            {"bg": "#1e293b", "text": "#94a3b8", "emoji": "👤"},
    "Pipeline":        {"bg": "#0f172a", "text": "#64748b", "emoji": "⚙️"},
}

def agent_color(name: str) -> dict:
    return AGENT_COLORS.get(name, {"bg": "#1e293b", "text": "#e2e8f0", "emoji": "🤖"})

# ─────────────────────────────────────────────────────────────────────────────
# Document extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if not PDF_OK: return "[PyPDF2 not installed]"
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if ext == ".docx":
        if not DOCX_OK: return "[python-docx not installed]"
        doc = python_docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext in (".txt", ".md", ".csv", ".tsv", ".log"):
        try: return file_bytes.decode("utf-8", errors="replace")
        except: return file_bytes.decode("latin-1", errors="replace")
    if ext == ".json":
        try: return json.dumps(json.loads(file_bytes), indent=2)
        except: return file_bytes.decode("utf-8", errors="replace")
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"):
        if not PIL_OK: return "[Pillow not installed]"
        img = Image.open(io.BytesIO(file_bytes))
        if OCR_OK: return pytesseract.image_to_string(img)
        return f"[Image: {img.format} {img.size[0]}x{img.size[1]}px — no OCR]"
    return "[Unsupported file type]"

# ─────────────────────────────────────────────────────────────────────────────
# Web search & fetch
# ─────────────────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> str:
    if not SEARCH_OK: return "[Web search unavailable]"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results: return "[No results found]"
        return "\n\n".join(
            f"[{i}] {r.get('title','')}\nURL: {r.get('href','')}\nSummary: {r.get('body','')}"
            for i, r in enumerate(results, 1))
    except Exception as e:
        return f"[Search error: {e}]"

def fetch_url(url: str, max_chars: int = 4000) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]): tag.decompose()
        lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()]
        return "\n".join(lines)[:max_chars]
    except Exception as e:
        return f"[Failed to fetch URL: {e}]"

# ─────────────────────────────────────────────────────────────────────────────
# Image generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_image(api_key: str, prompt: str, size: str, quality: str):
    if not OPENAI_OK: return None, "OpenAI not installed."
    try:
        client = OpenAI(api_key=api_key)
        response = client.images.generate(model="dall-e-3", prompt=prompt,
                                          size=size, quality=quality, n=1)
        return response.data[0].url, response.data[0].revised_prompt
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_store(api_key: str, documents: list):
    if not RAG_OK or not documents: return None
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
        return Chroma.from_texts(all_chunks, embeddings, metadatas=all_meta, persist_directory=tmpdir)
    except Exception as e:
        st.warning(f"RAG build failed: {e}"); return None

def retrieve_rag_context(store, query: str, k: int = 4):
    if store is None: return "", []
    try:
        docs = store.similarity_search(query, k=k)
        sources = list({d.metadata.get("source","?") for d in docs})
        return "\n\n".join(d.page_content for d in docs), sources
    except: return "", []

# ─────────────────────────────────────────────────────────────────────────────
# Task prompts & focus instructions
# ─────────────────────────────────────────────────────────────────────────────
TASK_SYSTEM_PROMPTS = {
    "General QA": (
        "You are a general-purpose expert assistant with additional expertise in time-course data interpretation. "
        "Answer clearly and accurately across domains. If uncertain, state that information is unavailable."
    ),
    "Clinical Analyst": (
        "You are a clinical data analyst with deep expertise in ICU data, sepsis, shock, phenotyping, "
        "physiological waveforms, labs and vital-sign time-series. Interpret trends, flag anomalies, "
        "and suggest clinical insights."
    ),
    "Code Interpreter": (
        "You are an expert Python programmer and data scientist. Analyse, debug, explain, and improve "
        "code. Always wrap code in triple-backtick python blocks."
    ),
    "Summarizer": (
        "You are a precise scientific summarizer. Condense long documents or data reports into concise, "
        "well-structured summaries preserving key findings and numerical results."
    ),
    "Data Analyst": (
        "You are a quantitative data analyst. Given data-related questions, produce Python code snippets, "
        "statistical summaries, or plain-language interpretations as requested."
    ),
    "RAG Assistant": (
        "You answer questions using ONLY the retrieved context provided. If the context is insufficient, "
        "say so clearly. Cite the source document when possible."
    ),
    "Resume Analyst": (
        "You are an expert resume analyst and career coach specializing in ATS optimization.\n\n"
        "1. **Fit Analysis**: Identify key skills, experience, and qualifications.\n"
        "2. **ATS Score**: Estimate an ATS compatibility score (0-100) and explain what is hurting it.\n"
        "3. **Improvement Suggestions**: List specific, actionable improvements.\n"
        "4. **Updated Resume**: Rewrite in clean markdown using **bold** headers and *italic* dates."
    ),
    "Image Generator": (
        "You are an expert at writing detailed, vivid DALL-E image generation prompts. "
        "When the user describes an image, enhance their description and confirm what will be generated."
    ),
}

TASK_DESCRIPTIONS = {
    "General QA":         "Ask anything — general knowledge, science, concepts, or open-ended questions.",
    "Clinical Analyst":   "Interpret ICU vitals, lab trends, sepsis phenotypes, and physiological time-series.",
    "Code Interpreter":   "Explain, debug, optimize, or rewrite code snippets in any language.",
    "Summarizer":         "Condense long documents, papers, or reports into concise structured summaries.",
    "Data Analyst":       "Analyze structured data, generate Python code, and extract statistical insights.",
    "RAG Assistant":      "Answer questions grounded strictly in your uploaded documents.",
    "Resume Analyst":     "Review, critique, and improve resumes for clarity, impact, and ATS compatibility.",
    "Image Generator":    "Generate images from text descriptions using DALL-E 3.",
}

AGENT_FOCUS = {
    "General QA": (
        "Focus ONLY on: general knowledge, factual questions, geography, travel, science concepts. "
        "Answer your assigned portion concisely. Ignore clinical or code-specific parts."
    ),
    "Clinical Analyst": (
        "Focus ONLY on: clinical data, ICU metrics, vital signs, lab values, sepsis criteria, "
        "physiological interpretation. Ignore non-clinical parts."
    ),
    "Code Interpreter": (
        "Focus ONLY on: writing, explaining, debugging, or improving code. "
        "Provide complete runnable Python in triple-backtick blocks. Ignore non-code parts."
    ),
    "Summarizer": (
        "Focus ONLY on: summarizing and condensing content. Produce a concise structured summary. "
        "Do NOT repeat prior agent outputs verbatim — synthesize them."
    ),
    "Data Analyst": (
        "Focus ONLY on: statistical analysis, data interpretation, chart code, quantitative insights. "
        "Provide Python/pandas code where helpful. Ignore non-data parts."
    ),
    "RAG Assistant": (
        "Focus ONLY on: answering from the provided document context. "
        "Quote or reference specific sections. If context insufficient, say so explicitly."
    ),
    "Resume Analyst": (
        "Focus ONLY on: resume review, ATS scoring, skill gap analysis, and rewriting. "
        "Ignore non-resume parts."
    ),
}

MULTI_AGENT_ROSTER = [
    "General QA", "Clinical Analyst", "Code Interpreter",
    "Summarizer", "Data Analyst", "RAG Assistant", "Resume Analyst",
]

# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────
def call_llm(provider: str, api_key: str, model: str, messages: list,
             temperature: float, max_tokens: int = 2048) -> str:
    if provider == "OpenAI" and OPENAI_OK:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
    if provider == "Anthropic (Claude)" and ANTHROPIC_OK:
        client = Anthropic(api_key=api_key)
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        resp = client.messages.create(model=model, max_tokens=max_tokens,
                                      system=system_msg, messages=user_msgs, temperature=temperature)
        return resp.content[0].text.strip()
    raise RuntimeError(f"Provider '{provider}' unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
ORCHESTRATOR_PROMPT = """You are the orchestrator of TC·QA·Agent.

Read the user query and return a JSON array of agent names to handle it.

AGENT SPECIALIZATIONS:
- "General QA"        → general knowledge, geography, travel, facts, open-ended
- "Clinical Analyst"  → ICU vitals, sepsis, lab trends, medical interpretation
- "Code Interpreter"  → write/debug/explain code, Python scripts
- "Summarizer"        → condense documents, papers, long reports
- "Data Analyst"      → statistics, data analysis, chart/plot code
- "RAG Assistant"     → questions needing answers FROM uploaded documents
- "Resume Analyst"    → resume review, ATS optimization, career advice

RULES:
1. Select 1-3 agents MAXIMUM.
2. Each agent handles a DIFFERENT part — do NOT assign same task to two agents.
3. For simple single-topic queries, return exactly ONE agent.
4. Return ONLY a valid JSON array. No explanation, no markdown.

EXAMPLES:
"What is the distance from Atlanta to Durham?" → ["General QA"]
"Analyze patient lactate trend and write Python to plot it" → ["Clinical Analyst", "Code Interpreter"]
"Summarize my uploaded paper" → ["RAG Assistant", "Summarizer"]
"Atlanta tourist spots, sepsis criteria, distance to Durham" → ["General QA", "Clinical Analyst"]

USER QUERY: {query}
"""

def orchestrate(api_key: str, provider: str, model: str, query: str) -> list:
    try:
        raw = call_llm(
            provider=provider, api_key=api_key, model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a routing orchestrator. Reply ONLY with a valid JSON array of agent names from: "
                    + str(MULTI_AGENT_ROSTER) + ". No extra text.")},
                {"role": "user", "content": ORCHESTRATOR_PROMPT.format(query=query)},
            ],
            temperature=0.0, max_tokens=100,
        )
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            agents = json.loads(match.group())
            agents = [a for a in agents if a in MULTI_AGENT_ROSTER]
            seen, unique = set(), []
            for a in agents:
                if a not in seen:
                    seen.add(a); unique.append(a)
            return unique[:3] if unique else ["General QA"]
        return ["General QA"]
    except Exception:
        return ["General QA"]


def run_multi_agent_pipeline(
    api_key: str, provider: str, model: str, query: str,
    agents: list, doc_context: str, manual_context: str,
    temperature: float, max_tokens: int,
) -> tuple:
    """Run agents sequentially. Returns (final_answer, agent_outputs, dialogue_log)."""
    agent_outputs = []
    prior_summary = ""
    dialogue_log  = []

    dialogue_log.append({
        "from": "Orchestrator", "to": "Pipeline",
        "message": f"Routing query to: {' -> '.join(agents)}",
        "type": "dispatch",
    })

    for i, agent_name in enumerate(agents):
        base_prompt = TASK_SYSTEM_PROMPTS.get(agent_name, TASK_SYSTEM_PROMPTS["General QA"])
        focus_instr = AGENT_FOCUS.get(agent_name, "Answer the relevant part of the query.")
        sender = "Orchestrator" if i == 0 else agents[i - 1]

        handoff_msg = (
            f"Please handle your specialist portion of: \"{query[:120]}\""
            if i == 0 else
            f"Here is what {agents[i-1]} produced. Now apply your [{agent_name}] expertise."
        )
        dialogue_log.append({
            "from": sender, "to": agent_name,
            "message": handoff_msg, "type": "handoff",
        })

        system_parts = [
            base_prompt,
            f"\n\n--- YOUR ROLE ---",
            f"You are agent {i+1} of {len(agents)}: [{agent_name}].",
            focus_instr,
            "Be concise and specific. Answer ONLY your specialist portion.",
        ]
        if doc_context.strip():
            system_parts.append(f"\n\n--- DOCUMENT CONTEXT ---\n{doc_context}")
        if manual_context.strip():
            system_parts.append(f"\n\n--- ADDITIONAL CONTEXT ---\n{manual_context}")
        if prior_summary:
            system_parts.append(
                f"\n\n--- PRIOR AGENT OUTPUTS (background only — do NOT repeat) ---\n{prior_summary}"
            )

        try:
            output = call_llm(
                provider=provider, api_key=api_key, model=model,
                messages=[
                    {"role": "system", "content": "\n".join(system_parts)},
                    {"role": "user",   "content": query},
                ],
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            output = f"[{agent_name} error: {e}]"

        agent_outputs.append({"agent": agent_name, "output": output})
        prior_summary += f"\n[{agent_name}]: {output[:500]}{'...' if len(output) > 500 else ''}\n"

        dialogue_log.append({
            "from": agent_name,
            "to": agents[i + 1] if i + 1 < len(agents) else "User",
            "message": output[:200] + ("..." if len(output) > 200 else ""),
            "type": "response",
        })

        if i + 1 < len(agents):
            dialogue_log.append({
                "from": agent_name, "to": agents[i + 1],
                "message": f"Passing analysis to [{agents[i+1]}] for further processing.",
                "type": "baton",
            })

    dialogue_log.append({
        "from": agents[-1], "to": "User",
        "message": "Pipeline complete. Delivering final response.",
        "type": "done",
    })

    if len(agent_outputs) == 1:
        final = agent_outputs[0]["output"]
    else:
        sections = [f"### {agent_color(ao['agent'])['emoji']} {ao['agent']}\n\n{ao['output']}"
                    for ao in agent_outputs]
        final = "\n\n---\n\n".join(sections)

    return final, agent_outputs, dialogue_log


# ─────────────────────────────────────────────────────────────────────────────
# Dialogue panel renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_dialogue_panel(dialogue_log: list):
    if not dialogue_log: return
    with st.expander("🤖 Agent Dialogue Exchange", expanded=False):
        st.markdown('<div class="dialogue-panel">', unsafe_allow_html=True)
        st.markdown('<div class="dialogue-title">MULTI-AGENT PIPELINE — INTERNAL DIALOGUE</div>',
                    unsafe_allow_html=True)
        for entry in dialogue_log:
            frm = entry["from"]; to = entry["to"]
            msg = entry["message"]; typ = entry["type"]
            fc  = agent_color(frm); tc = agent_color(to)

            if typ in ("dispatch", "done"):
                st.markdown(
                    f'<div class="d-arrow">-- {fc["emoji"]} {frm} -> {tc["emoji"]} {to} --</div>'
                    f'<div style="text-align:center;color:#64748b;font-size:0.72rem;margin-bottom:6px;">{msg}</div>',
                    unsafe_allow_html=True,
                )
            elif typ == "baton":
                st.markdown(
                    f'<div class="d-arrow" style="color:#2dd4bf;">v {fc["emoji"]} {frm} passes baton to {tc["emoji"]} {to}</div>',
                    unsafe_allow_html=True,
                )
            else:
                align_class = "right" if typ == "response" else ""
                st.markdown(
                    f'<div class="d-msg {align_class}">'
                    f'<div class="d-avatar" style="background:{fc["bg"]};color:{fc["text"]};">{fc["emoji"]}</div>'
                    f'<div class="d-bubble" style="background:{fc["bg"]};">'
                    f'<div class="d-name" style="color:{fc["text"]};">{fc["emoji"]} {frm} -> {tc["emoji"]} {to}</div>'
                    f'{msg}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
def export_conversation(messages: list, fmt: str) -> bytes:
    if fmt == "JSON":
        return json.dumps(messages, indent=2, ensure_ascii=False).encode("utf-8")
    if fmt == "TXT":
        lines = [f"[{'You' if m['role']=='user' else 'TC·QA·Agent'}]\n{m['content']}\n" for m in messages]
        return "\n".join(lines).encode("utf-8")
    if fmt == "CSV":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["role", "content"])
        w.writeheader(); w.writerows(messages)
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
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 12px;'>
        <div style='font-family:"Playfair Display",serif;font-size:2.6rem;font-weight:900;
                    background:linear-gradient(135deg,#f59e0b,#ef4444,#ec4899);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>TC·QA·Agent</div>
        <div style='font-size:0.7rem;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;'>
            <strong style="color:var(--accent);font-size:0.9rem;">Time-Course Q&A Assistant</strong></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # Provider & Model
    st.markdown('<div class="card-label">🤖 Provider & Model</div>', unsafe_allow_html=True)
    provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic (Claude)"], label_visibility="collapsed")
    provider_models = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "dall-e-3"],
        "Anthropic (Claude)": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    }
    auto_model  = st.session_state.get("auto_model")
    model_list  = provider_models[provider]
    default_idx = model_list.index(auto_model) if (auto_model and auto_model in model_list) else 0
    model       = st.selectbox("Model", model_list, index=default_idx, label_visibility="collapsed")
    if auto_model and auto_model in model_list:
        st.caption(f"🤖 Auto-set to `{auto_model}` for Image Generator")
    api_key = st.text_input("API Key", type="password", placeholder="sk-... or sk-ant-...")

    # Task Mode & MultiAgent
    st.markdown('<div class="card-label" style="margin-top:14px;">⚙️ Task Mode</div>', unsafe_allow_html=True)
    multi_agent_mode = st.toggle("🧠 MultiAgent Mode", value=False,
        help="ON: Orchestrator auto-selects and chains specialist agents.\nOFF: Pick one agent manually.")

    if multi_agent_mode:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0f2a1a,#0f1a2a);'
            'border:1px solid #2dd4bf;border-radius:8px;padding:10px 14px;margin:6px 0;">'
            '<span style="color:#2dd4bf;font-size:0.75rem;font-weight:700;letter-spacing:0.08em;">'
            '🤖 ORCHESTRATOR ACTIVE</span><br>'
            '<span style="color:#94a3b8;font-size:0.75rem;">Auto-selects &amp; chains agents sequentially</span>'
            '</div>', unsafe_allow_html=True,
        )
        pinned_agents = st.multiselect(
            "Pin agents (optional — leave empty for auto)",
            options=MULTI_AGENT_ROSTER, default=[],
            help="Force specific agents. Leave empty to let orchestrator decide.",
        )
        task_mode = "General QA"
        if "Image Generator" in pinned_agents and provider == "OpenAI":
            st.session_state["auto_model"]  = "dall-e-3"
        else:
            st.session_state["auto_model"]  = None
        if "RAG Assistant" in pinned_agents:
            st.session_state["auto_rag"]    = True
            st.session_state["auto_search"] = True
        else:
            st.session_state["auto_rag"]    = None
            st.session_state["auto_search"] = None
    else:
        pinned_agents = []
        task_mode = st.selectbox("Task", list(TASK_SYSTEM_PROMPTS.keys()), label_visibility="collapsed")
        st.caption(f"💡 {TASK_DESCRIPTIONS[task_mode]}")
        if task_mode == "Image Generator" and provider == "OpenAI":
            st.session_state["auto_model"]  = "dall-e-3"
        else:
            st.session_state["auto_model"]  = None
        if task_mode == "RAG Assistant":
            st.session_state["auto_rag"]    = True
            st.session_state["auto_search"] = True
        else:
            st.session_state["auto_rag"]    = None
            st.session_state["auto_search"] = None

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05, help="0=precise · 0.7+=creative")
    max_tokens  = st.slider("Max tokens", 256, 4096, 1024, 128)

    # RAG
    st.markdown("---")
    st.markdown('<div class="card-label">📚 RAG Settings</div>', unsafe_allow_html=True)
    _rag_default = bool(st.session_state.get("auto_rag"))
    use_rag = st.toggle("Enable RAG", value=_rag_default, disabled=not RAG_OK,
                        help="Retrieval-Augmented Generation using uploaded documents")
    if st.session_state.get("auto_rag"):
        st.caption("🤖 Auto-enabled for RAG Assistant")
    if not RAG_OK:
        st.caption("⚠️ Install langchain-openai, langchain-community, chromadb to enable RAG")
    rag_k = st.slider("Top-k chunks", 1, 8, 4, 1, disabled=not use_rag)
    if use_rag and RAG_OK and api_key and st.session_state["documents"] and st.session_state["rag_store"] is None:
        with st.spinner("Building RAG index..."):
            st.session_state["rag_store"] = build_rag_store(api_key, st.session_state["documents"])
        if st.session_state["rag_store"]:
            st.success("🔎 RAG index ready.")
        else:
            st.warning("RAG index build failed — check OpenAI API key.")

    # Web Search
    st.markdown("---")
    st.markdown('<div class="card-label">🌐 Web Search</div>', unsafe_allow_html=True)
    _search_default = bool(st.session_state.get("auto_search"))
    use_web_search = st.toggle("Enable Web Search", value=_search_default, disabled=not SEARCH_OK,
                               help="Automatically search the web to answer questions")
    if st.session_state.get("auto_search"):
        st.caption("🤖 Auto-enabled for RAG Assistant")
    if not SEARCH_OK:
        st.caption("⚠️ Install ddgs to enable web search")
    web_search_k = st.slider("Max results", 1, 8, 4, 1, disabled=not use_web_search)

    # Image Generation
    st.markdown("---")
    st.markdown('<div class="card-label">🎨 Image Generation</div>', unsafe_allow_html=True)
    img_size    = st.selectbox("Size",    ["1024x1024","1792x1024","1024x1792"], label_visibility="collapsed")
    img_quality = st.selectbox("Quality", ["standard","hd"],                    label_visibility="collapsed")

    # Memory
    st.markdown("---")
    st.markdown('<div class="card-label">🧠 Conversation Memory</div>', unsafe_allow_html=True)
    carry_history = st.toggle("Carry conversation history", value=True)
    max_history   = st.slider("Max history turns", 1, 20, 6, 1, disabled=not carry_history)

    # Export
    st.markdown("---")
    st.markdown('<div class="card-label">💾 Export</div>', unsafe_allow_html=True)
    export_fmt = st.selectbox("Format", ["JSON","TXT","CSV","Markdown"], label_visibility="collapsed")
    if st.button("⬇ Export Conversation"):
        if st.session_state["messages"]:
            data = export_conversation(st.session_state["messages"], export_fmt)
            st.download_button(label=f"Download .{export_fmt.lower()}", data=data,
                               file_name=f"tc_qa_agent_export.{export_fmt.lower()}", mime="text/plain")
        else:
            st.warning("No conversation to export yet.")

    # Clear
    st.markdown("---")
    if st.button("🗑 Clear Conversation"):
        st.session_state["messages"] = []
        st.session_state["agent_dialogues"] = []
        st.rerun()
    if st.button("🗑 Clear Documents & RAG"):
        st.session_state["documents"] = []
        st.session_state["rag_store"] = None
        st.session_state["rag_texts"] = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tc-header">
    <div class="tc-logo">TC·QA·Agent</div>
    <div class="tc-subtitle">Time-Course Data Intelligence Platform</div>
</div>
<p style="font-size:1.1rem;color:var(--text);margin:-10px 0 20px 0;letter-spacing:0.01em;">
    A web-based tool with advanced Q&A systems for interpreting text, data, code, images and documents using LLMs
    with <strong style="color:var(--accent);">8 expert assistants</strong>, RAG, multi-modal document ingestion,
    web search, multi-provider LLM support, conversation memory, and export capabilities.
</p>
<p style="font-size:1rem;color:var(--text);margin-bottom:24px;padding:12px 16px;background:var(--bg3);border-radius:8px;">
    <strong style="font-size:1.3rem;">Instructions</strong>: Put your
    <strong style="color:var(--accent);">API Key</strong> in the sidebar.
    Select a <strong style="color:var(--accent);">TASK MODE</strong> or enable
    <strong style="color:var(--accent);">🧠 MultiAgent Mode</strong>, then ask questions in the
    <strong style="color:var(--accent);">CHAT tab</strong>.
    Upload documents in the <strong style="color:var(--accent);">DOCUMENTS tab</strong>.
    Use toggles to enable <strong style="color:var(--accent);">RAG</strong> and
    <strong style="color:var(--accent);">Web Search</strong>.
</p>
<div style="margin-bottom:18px;">
    <div style="font-size:1.2rem;font-weight:700;margin-bottom:6px;color:var(--text);">Assistants</div>
    <div class="card" style="padding:10px;">
        <ul style="margin:0;padding-left:20px;color:var(--text);">
            <li><strong>General QA</strong> — Ask anything across domains.</li>
            <li><strong>Clinical Analyst</strong> — ICU data, sepsis, vital-sign time-series.</li>
            <li><strong>Code Interpreter</strong> — Debug, explain, optimize code.</li>
            <li><strong>Summarizer</strong> — Condense documents and reports.</li>
            <li><strong>Data Analyst</strong> — Statistics, Python code, quantitative insights.</li>
            <li><strong>RAG Assistant</strong> — Answers grounded in uploaded documents. <em>(auto-enables RAG + Web Search)</em></li>
            <li><strong>Resume Analyst</strong> — ATS scoring, resume rewriting.</li>
            <li><strong>Image Generator</strong> — DALL-E 3 image generation. <em>(auto-selects dall-e-3 model)</em></li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_context, tab_stats = st.tabs(
    ["💬 Chat", "📂 Documents", "📝 Context", "📊 Stats"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    status_cols = st.columns([1, 1, 1, 1, 2])
    with status_cols[0]:
        provider_ok = (provider == "OpenAI" and OPENAI_OK) or (provider == "Anthropic (Claude)" and ANTHROPIC_OK)
        badge = "badge-green" if (api_key and provider_ok) else "badge-red"
        st.markdown(f'<span class="badge {badge}">{"Connected" if (api_key and provider_ok) else "No API Key"}</span>', unsafe_allow_html=True)
    with status_cols[1]:
        has_docs = bool(st.session_state["documents"])
        rag_badge = ("badge-green" if (use_rag and st.session_state["rag_store"]) else
                     "badge-amber" if use_rag else "badge-blue")
        rag_label = ("RAG Active" if (use_rag and st.session_state["rag_store"]) else
                     "RAG: No Index" if (use_rag and has_docs) else
                     "RAG: No Docs" if use_rag else "RAG Off")
        st.markdown(f'<span class="badge {rag_badge}">{rag_label}</span>', unsafe_allow_html=True)
    with status_cols[2]:
        n_docs = len(st.session_state["documents"])
        st.markdown(f'<span class="badge badge-blue">{n_docs} doc{"s" if n_docs!=1 else ""}</span>', unsafe_allow_html=True)
    with status_cols[3]:
        if multi_agent_mode:
            st.markdown('<span class="badge badge-purple">🧠 MultiAgent</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge badge-blue">{task_mode}</span>', unsafe_allow_html=True)
    with status_cols[4]:
        mode_label = "🧠 MultiAgent" if multi_agent_mode else task_mode
        st.caption(f"Model: `{model}` · Mode: **{mode_label}** · T={temperature}")

    st.markdown("---")

    # Chat history
    chat_container = st.container()
    with chat_container:
        dialogue_map = {d["turn"]: d["log"] for d in st.session_state.get("agent_dialogues", [])}
        turn_idx = 0
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-row user-row">'
                    f'<div class="avatar avatar-user">👤</div>'
                    f'<div class="msg-user"><div class="msg-text">{msg["content"]}</div></div>'
                    f'</div>', unsafe_allow_html=True,
                )
                turn_idx += 1
            else:
                # Render dialogue panel for this turn
                if turn_idx in dialogue_map:
                    render_dialogue_panel(dialogue_map[turn_idx])

                src_html   = "".join(f'<span class="src-chip">📄 {s}</span>' for s in msg.get("sources", []))
                agents_used = msg.get("agents", [])
                agent_html  = "".join(
                    f'<span class="src-chip" style="background:{agent_color(a)["bg"]};'
                    f'border-color:{agent_color(a)["text"]};color:{agent_color(a)["text"]};">'
                    f'{agent_color(a)["emoji"]} {a}</span>'
                    for a in agents_used
                )
                st.markdown(
                    f'<div class="msg-row">'
                    f'<div class="avatar avatar-ai">🧬</div>'
                    f'<div class="msg-ai">'
                    f'{agent_html + "<br>" if agent_html else ""}'
                    f'{src_html + "<br>" if src_html else ""}'
                    f'<div class="msg-text">',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                st.markdown('</div></div></div>', unsafe_allow_html=True)

    # Input
    st.markdown("")
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "Ask a question...",
            placeholder="e.g., What does the lactate trend suggest about sepsis progression?\n"
                        "Or paste code to interpret, data to summarise, etc.",
            height=100, label_visibility="collapsed",
        )
        col_send, _ = st.columns([1, 5])
        with col_send:
            submitted = st.form_submit_button("▶ Send", type="primary", use_container_width=True)

    if submitted and user_query.strip():
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
        else:
            st.session_state["messages"].append({"role": "user", "content": user_query.strip()})
            current_turn = len([m for m in st.session_state["messages"] if m["role"] == "user"])

            with st.spinner(""):
                st.markdown('<p class="thinking">TC·QA·Agent is thinking...</p>', unsafe_allow_html=True)

                # Collect contexts
                manual_ctx  = st.session_state.get("manual_context", "").strip()
                doc_context = ""
                rag_sources = []

                if not use_rag and st.session_state["documents"]:
                    doc_blocks = []
                    for doc in st.session_state["documents"]:
                        preview = doc["text"][:4000]
                        if len(doc["text"]) > 4000: preview += "\n\n[... truncated ...]"
                        doc_blocks.append(f"[Document: {doc['name']}]\n{preview}")
                    doc_context = "\n\n".join(doc_blocks)

                if use_rag:
                    if st.session_state["rag_store"]:
                        rag_ctx, rag_sources = retrieve_rag_context(
                            st.session_state["rag_store"], user_query.strip(), k=rag_k)
                        doc_context = rag_ctx if rag_ctx else "\n".join(
                            f"[Document: {d['name']}]\n{d['text'][:3000]}"
                            for d in st.session_state["documents"])
                    else:
                        doc_context = "\n".join(
                            f"[Document: {d['name']}]\n{d['text'][:3000]}"
                            for d in st.session_state["documents"])
                        if st.session_state["documents"]:
                            st.warning("⚠️ RAG index not built — using raw document text.")

                if use_web_search:
                    with st.spinner("🌐 Searching the web..."):
                        search_results = web_search(user_query.strip(), max_results=web_search_k)
                    doc_context += f"\n\n--- WEB SEARCH RESULTS ---\n{search_results}"
                    for url in re.findall(r'https?://[^\s\)\"\']+', user_query)[:2]:
                        doc_context += f"\n\n--- FETCHED PAGE: {url} ---\n{fetch_url(url)}"

                # Path 1: Image generation
                if model == "dall-e-3" or task_mode == "Image Generator":
                    with st.spinner("🎨 Generating image..."):
                        img_url, revised_prompt = generate_image(
                            api_key, user_query.strip(), img_size, img_quality)
                    if img_url:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": (f"**Revised prompt used:**\n_{revised_prompt}_\n\n"
                                        f"![Generated Image]({img_url})\n\n[Download image]({img_url})"),
                            "sources": [], "image_url": img_url,
                        })
                    else:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": f"Image generation failed: {revised_prompt}",
                            "sources": [],
                        })
                    st.session_state["history"].append(f"Q: {user_query.strip()}\nA: [Image generated]")

                # Path 2: MultiAgent
                elif multi_agent_mode:
                    if pinned_agents:
                        selected_agents   = pinned_agents
                        orchestrator_note = f"📌 Pinned agents: {' -> '.join(selected_agents)}"
                    else:
                        with st.spinner("🧠 Orchestrator routing query to specialist agents..."):
                            selected_agents = orchestrate(
                                api_key=api_key, provider=provider,
                                model=model, query=user_query.strip())
                        orchestrator_note = f"🤖 Orchestrator selected: {' -> '.join(selected_agents)}"

                    with st.spinner(f"⚙️ Running {len(selected_agents)} agent(s): {', '.join(selected_agents)}..."):
                        final_answer, agent_outputs, dialogue_log = run_multi_agent_pipeline(
                            api_key=api_key, provider=provider, model=model,
                            query=user_query.strip(), agents=selected_agents,
                            doc_context=doc_context, manual_context=manual_ctx,
                            temperature=temperature, max_tokens=max_tokens,
                        )

                    st.session_state["agent_dialogues"].append({
                        "turn": current_turn, "log": dialogue_log,
                    })

                    agent_badges = " -> ".join(f"**[{ao['agent']}]**" for ao in agent_outputs)
                    full_response = (
                        f"> {orchestrator_note}\n\n"
                        f"**Pipeline:** {agent_badges}\n\n---\n\n"
                        f"{final_answer}"
                    )
                    st.session_state["messages"].append({
                        "role": "assistant", "content": full_response,
                        "sources": rag_sources, "agents": selected_agents,
                    })
                    st.session_state["history"].append(
                        f"Q: {user_query.strip()}\nA: [MultiAgent: {', '.join(selected_agents)}]")

                # Path 3: Single-agent
                else:
                    system_content = TASK_SYSTEM_PROMPTS[task_mode]
                    if manual_ctx:
                        system_content += f"\n\n--- ADDITIONAL CONTEXT ---\n{manual_ctx}"
                    if doc_context:
                        system_content += f"\n\n--- UPLOADED DOCUMENTS ---\n{doc_context}"

                    chat_messages = [{"role": "system", "content": system_content}]
                    if carry_history:
                        for h in st.session_state["messages"][-(max_history * 2 + 1):-1]:
                            chat_messages.append({"role": h["role"], "content": h["content"]})
                    chat_messages.append({"role": "user", "content": user_query.strip()})

                    try:
                        answer = call_llm(provider=provider, api_key=api_key, model=model,
                                          messages=chat_messages, temperature=temperature,
                                          max_tokens=max_tokens)
                        st.session_state["messages"].append({
                            "role": "assistant", "content": answer, "sources": rag_sources,
                        })
                        st.session_state["history"].append(f"Q: {user_query.strip()}\nA: {answer}")
                    except Exception as e:
                        err = f"**Error calling {provider}:** {e}\n\n```\n{traceback.format_exc()}\n```"
                        st.session_state["messages"].append({
                            "role": "assistant", "content": err, "sources": []
                        })

            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown('<div class="card-label">📂 Upload Documents for RAG & Context</div>', unsafe_allow_html=True)
    st.caption("Supported: PDF · DOCX · TXT · MD · CSV · JSON · PNG / JPG (OCR) · TIFF · BMP · WEBP")

    uploaded_files = st.file_uploader(
        "Drop files here or browse",
        type=["pdf","docx","txt","md","csv","tsv","json","log","png","jpg","jpeg","tiff","bmp","gif","webp"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded_files:
        new_names = {d["name"] for d in st.session_state["documents"]}
        added = 0
        for f in uploaded_files:
            if f.name in new_names: continue
            with st.spinner(f"Extracting text from **{f.name}**..."):
                raw = f.read(); text = extract_text_from_file(raw, f.name)
            st.session_state["documents"].append({
                "name": f.name, "text": text,
                "size_kb": round(len(raw)/1024,1), "chars": len(text),
            })
            new_names.add(f.name); added += 1
        if added:
            st.success(f"✅ Added {added} document(s).")
            if use_rag and RAG_OK and api_key:
                with st.spinner("Building vector index..."):
                    st.session_state["rag_store"] = build_rag_store(api_key, st.session_state["documents"])
                if st.session_state["rag_store"]:
                    st.success("🔎 RAG index built.")
                else:
                    st.warning("RAG index build failed.")
            elif use_rag and not api_key:
                st.info("ℹ️ Enter your API key to build the RAG index.")

    if st.session_state["documents"]:
        st.markdown("---")
        st.markdown(f'<div class="card-label">Loaded Documents ({len(st.session_state["documents"])})</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Name": d["name"], "Size (KB)": d["size_kb"], "Extracted chars": d["chars"]}
            for d in st.session_state["documents"]
        ]), use_container_width=True, hide_index=True)
        preview_name = st.selectbox("Preview document text", [d["name"] for d in st.session_state["documents"]])
        if preview_name:
            preview_doc = next(d for d in st.session_state["documents"] if d["name"] == preview_name)
            with st.expander("📄 Text Preview (first 3000 chars)", expanded=False):
                st.text(preview_doc["text"][:3000])
        if use_rag and RAG_OK and api_key:
            if st.button("🔄 Rebuild RAG Index"):
                with st.spinner("Rebuilding..."):
                    st.session_state["rag_store"] = build_rag_store(api_key, st.session_state["documents"])
                st.success("RAG index rebuilt.")
    else:
        st.info("No documents loaded yet.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab_context:
    st.markdown('<div class="card-label">📝 Manual Context / Paste Data</div>', unsafe_allow_html=True)
    st.caption("Paste any text, time-series data, code, or clinical notes here.")

    ctx_placeholder = (
        "Example — paste a time-course table:\n\n"
        "Time(h)  HR(bpm)  MAP(mmHg)  Lactate(mmol/L)  Temp(C)\n"
        "0        88       72         1.2              37.1\n"
        "6        102      65         2.1              38.4\n"
        "12       118      58         3.8              39.1\n"
        "24       128      51         5.2              39.6\n\n"
        "Or paste code, reports, or any reference material..."
    )
    manual_context = st.text_area("Context", placeholder=ctx_placeholder, height=320, label_visibility="collapsed")
    st.session_state["manual_context"] = manual_context

    if manual_context.strip():
        st.markdown(f'<span class="badge badge-green">✓ {len(manual_context):,} characters ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-amber">No manual context</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-label">💡 Quick Prompt Templates</div>', unsafe_allow_html=True)
    templates = {
        "Sepsis time-course": (
            "Patient ID: ICU-042\nDiagnosis: Septic shock (gram-negative)\n\n"
            "Time(h)  HR   MAP  Lactate  WBC    Temp\n"
            "0        88   72   1.2      12.4   37.1\n6        102  65   2.1      18.1   38.4\n"
            "12       118  58   3.8      22.3   39.1\n24       128  51   5.2      28.6   39.6\n"
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
            "Urine output 1200   650    280    >500 mL/d\n"
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
    msgs   = st.session_state["messages"]
    docs   = st.session_state["documents"]
    chunks = st.session_state["rag_texts"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turns",              len([m for m in msgs if m["role"] == "user"]))
    m2.metric("Documents",          len(docs))
    m3.metric("RAG Chunks",         len(chunks))
    m4.metric("Total chars indexed", f"{sum(d['chars'] for d in docs):,}" if docs else "0")

    if msgs:
        st.markdown("---")
        st.markdown('<div class="card-label">💬 Conversation History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"#": i+1, "Role": m["role"],
             "Agents": ", ".join(m.get("agents", [])) or "—",
             "Message (preview)": m["content"][:120] + "..."}
            for i, m in enumerate(msgs)
        ]), use_container_width=True, hide_index=True)

    if docs:
        st.markdown("---")
        st.markdown('<div class="card-label">📂 Document Breakdown</div>', unsafe_allow_html=True)
        df_docs = pd.DataFrame([{"Name": d["name"], "KB": d["size_kb"], "Chars": d["chars"]} for d in docs])
        st.bar_chart(df_docs.set_index("Name")["Chars"])

    st.markdown("---")
    st.markdown('<div class="card-label">🔧 Dependency Status</div>', unsafe_allow_html=True)
    for dep, ok in {
        "openai": OPENAI_OK, "anthropic": ANTHROPIC_OK, "PyPDF2": PDF_OK,
        "python-docx": DOCX_OK, "Pillow": PIL_OK, "pytesseract (OCR)": OCR_OK,
        "ddgs (web search)": SEARCH_OK, "langchain + chromadb (RAG)": RAG_OK,
    }.items():
        st.markdown(f"{'✅' if ok else '❌'} `{dep}`")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#334155;font-size:0.75rem;padding:8px 0;">'
    'Copyright © 2026 Tilendra Choudhary · TC·QA·Agent · Time-Course Intelligence Platform with 8 Assistants'
    '</div>', unsafe_allow_html=True,
)
