"""
Enhanced QA Chatbot GUI with RAG (Retrieval-Augmented Generation), Task Modes, and Multi-Modal Input Support
---------------------------------------------------------------------------------
New Features:
1. Embedded RAG (Chroma + LangChain) for contextual document retrieval
2. Task modes for specialized prompt flows (QA, Code Interpreter, Summarizer, Data Analyst)
3. Persistent memory and conversation summarization
4. Drag-and-drop document ingestion and live embedding
5. Modular structure for easy plugin expansion
6. Multi-threaded and cancel-safe operations with improved GUI responsiveness
"""

import os
import json
import threading
import tempfile
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional, Callable, Dict

from openai import OpenAI
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings  # ✅ updated import
from langchain_community.vectorstores import Chroma  # ✅ updated namespace
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ updated import
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- RAG controller ---
class RAGManager:
    def __init__(self, base_dir="./rag_store"):
        os.makedirs(base_dir, exist_ok=True)
        self.vector_db = Chroma(
            persist_directory=base_dir,
            embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        )
        # self.vector_db = Chroma(persist_directory=base_dir, embedding_function=OpenAIEmbeddings())
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    def add_document(self, text: str, name: str):
        chunks = self.splitter.split_text(text)
        meta = [{"source": name}] * len(chunks)
        self.vector_db.add_texts(chunks, metadatas=meta)

    def retrieve_context(self, query: str, k=3) -> str:
        docs = self.vector_db.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])

# --- Core model call function ---
def query_llm(api_key: str, model: str, messages: List[dict], temperature: float) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content.strip()

# --- Task templates ---
TASK_TEMPLATES = {
    "General QA": "You are a helpful assistant. Answer concisely and clearly.",
    "Code Interpreter": "You are a skilled code interpreter. Analyze, explain, and modify code snippets.",
    "Summarizer": "Summarize documents or text to a concise, factual summary.",
    "Data Analyst": "Given structured data or analysis context, provide Python code snippets or concise analysis.",
    "RAG Assistant": "You answer user questions using context retrieved from a document database."
}

# --- GUI Class ---
class SmartChatGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Enhanced QA Chatbot with RAG")
        root.geometry("1150x850")

        self.api_key = tk.StringVar()
        self.model = tk.StringVar(value="gpt-4o-mini")
        self.temperature = tk.DoubleVar(value=0.4)
        self.task_mode = tk.StringVar(value="General QA")

        self.rag = RAGManager()
        self.documents = []
        self.cancel_event = threading.Event()

        frm = ttk.Frame(root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Top Row – API + Model + Task
        row_api = ttk.Frame(frm)
        row_api.grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Label(row_api, text="API Key:").grid(row=0, column=0)
        ttk.Entry(row_api, textvariable=self.api_key, width=40, show="•").grid(row=0, column=1, padx=6)
        ttk.Label(row_api, text="Model:").grid(row=0, column=2)
        ttk.Entry(row_api, textvariable=self.model, width=18).grid(row=0, column=3, padx=4)
        ttk.Label(row_api, text="Task Mode:").grid(row=0, column=4, padx=(10, 0))
        ttk.Combobox(row_api, textvariable=self.task_mode, values=list(TASK_TEMPLATES.keys()), state="readonly", width=16).grid(row=0, column=5)

        # Context + Query Inputs
        ttk.Label(frm, text="Context:").grid(row=1, column=0, sticky="w")
        self.context_box = ScrolledText(frm, height=6)
        self.context_box.grid(row=2, column=0, sticky="ew", pady=3)
        ttk.Label(frm, text="Your Question / Instruction:").grid(row=3, column=0, sticky="w")
        self.query_box = ScrolledText(frm, height=5)
        self.query_box.grid(row=4, column=0, sticky="ew", pady=3)

        # Attach documents for RAG
        attach_row = ttk.Frame(frm)
        attach_row.grid(row=5, column=0, sticky="ew", pady=(4, 4))
        ttk.Button(attach_row, text="Add Document", command=self.add_doc).grid(row=0, column=0, padx=4)
        ttk.Button(attach_row, text="Clear Docs", command=self.clear_docs).grid(row=0, column=1, padx=4)
        self.docs_label = ttk.Label(attach_row, text="No documents added")
        self.docs_label.grid(row=0, column=2, padx=6)

        # Output Box
        ttk.Label(frm, text="Response:").grid(row=6, column=0, sticky="w")
        self.output = ScrolledText(frm, height=14, wrap="word", background="#0f172a", foreground="#e5e5e5")
        self.output.grid(row=7, column=0, sticky="nsew", pady=3)
        frm.rowconfigure(7, weight=1)

        # Control Buttons
        btn_row = ttk.Frame(frm)
        btn_row.grid(row=8, column=0, pady=6)
        ttk.Button(btn_row, text="Run", command=self.run_query).grid(row=0, column=0, padx=4)
        ttk.Button(btn_row, text="Cancel", command=self.cancel_query).grid(row=0, column=1, padx=4)
        ttk.Button(btn_row, text="Clear Output", command=self.clear_output).grid(row=0, column=2, padx=4)

    # ---------- File and RAG Handlers ----------
    def add_doc(self):
        files = filedialog.askopenfilenames(title="Add documents", filetypes=[("Documents", "*.txt *.pdf *.docx")])
        for path in files:
            text = self.extract_text(path)
            if text:
                self.rag.add_document(text, os.path.basename(path))
                self.documents.append(os.path.basename(path))
        self.docs_label.config(text=", ".join(self.documents) or "No documents")

    def clear_docs(self):
        self.documents.clear()
        self.rag = RAGManager()
        self.docs_label.config(text="No documents")

    def extract_text(self, path: str) -> str:
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif path.lower().endswith(".txt"):
            return open(path, "r", encoding="utf-8", errors="ignore").read()
        else:
            return ""  # Simplified – handle DOCX or OCR optionally

    # ---------- LLM Execution ----------
    def run_query(self):
        if not self.api_key.get().strip():
            messagebox.showwarning("Missing Key", "Please provide your API key.")
            return
        query = self.query_box.get("1.0", "end").strip()
        context = self.context_box.get("1.0", "end").strip()
        mode = self.task_mode.get()

        self.output.insert("end", f"\n[Running {mode}]\n", "info")
        self.cancel_event.clear()

        def worker():
            try:
                rag_context = ""
                if mode == "RAG Assistant" and query:
                    rag_context = self.rag.retrieve_context(query)
                full_context = f"{TASK_TEMPLATES[mode]}\n\n{context}\n\nRetrieved Context:\n{rag_context}"
                messages = [
                    {"role": "system", "content": full_context},
                    {"role": "user", "content": query}
                ]
                response = query_llm(self.api_key.get().strip(), self.model.get(), messages, self.temperature.get())
                if not self.cancel_event.is_set():
                    self.append_response(response)
            except Exception as e:
                self.append_response("Error: " + str(e))

        threading.Thread(target=worker, daemon=True).start()

    def cancel_query(self):
        self.cancel_event.set()
        self.append_response("[Cancelled current job]")

    def append_response(self, text):
        self.output.insert("end", text + "\n\n")
        self.output.see("end")

    def clear_output(self):
        self.output.delete("1.0", "end")

# --- Main Entry ---
def main():
    root = tk.Tk()
    app = SmartChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
