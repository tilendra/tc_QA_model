import threading
import time
import json
import csv
import os
from typing import List, Optional, Callable, Dict

try:
    # Update import to whichever OpenAI client you're using
    from openai import OpenAI
except Exception:
    # Placeholder - user must install openai package or adapt to their client lib.
    OpenAI = None

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
except Exception as e:
    raise RuntimeError("Tkinter is required to run this GUI script.") from e


def ask_questions_with_context(
    api_key: str,
    questions: List[str],
    context: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "json",      # "json", "txt", or "csv"
    colorize: bool = True,
    show_progress: bool = True,
    append_to_file: bool = False,
    max_retries: int = 2,
    retry_delay: float = 2.0,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    provider: str = "openai"        # "openai", "mistral", "llama"
) -> List[str]:
    """
    Ask multiple questions to Chat APIs (OpenAI supported inline). This function:
      - Appends prior Q&A as context for subsequent questions.
      - Optionally saves results to a file in json/txt/csv formats.
      - Reports progress via progress_callback(type, text) where type is 'q', 'a', or 'info'.
      - provider: select provider; currently only OpenAI client usage is implemented inline.
        For other providers, the API key is exported to an environment variable so external SDKs
        or runtimes can pick it up (user must integrate provider SDK calls if desired).
    """
    def cb(t: str, msg: str):
        if progress_callback:
            try:
                progress_callback(t, msg)
            except Exception:
                # Swallow UI callback errors so they don't break the worker
                pass

    # Normalize provider
    provider_norm = (provider or "openai").strip().lower()

    # Export API key into provider-specific env var so third-party libs can find it.
    if api_key:
        if provider_norm == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider_norm == "mistral":
            os.environ["MISTRAL_API_KEY"] = api_key
        elif provider_norm == "llama":
            os.environ["LLAMA_API_KEY"] = api_key
        else:
            # default fallback
            os.environ["API_KEY"] = api_key

    if provider_norm != "openai":
        # We deliberately allow environment variable supply for other providers, but we don't
        # attempt to call their SDKs here (SDKs / local runtimes differ widely).
        cb("info", f"Provider '{provider_norm}' selected. API key exported to environment; to use this provider natively, integrate its SDK/client and call it where OpenAI client is used.")
        # For safety, continue only if OpenAI client exists and provider is openai.
        # If user selected a non-openai provider but still has OpenAI client and wishes to use it,
        # we won't block them here; the code below will still try to use OpenAI if available.
        # Otherwise, we will raise helpful message below when OpenAI client is not available.
    if provider_norm == "openai":
        if OpenAI is None:
            cb("info", "Error: OpenAI client not available. Install and configure the openai package.")
            raise RuntimeError("OpenAI client not available")

        client = OpenAI(api_key=api_key)
    else:
        # Non-OpenAI providers: we do not implement provider SDK calls inline by default.
        # If the user has a custom client integration, they can modify this section to
        # instantiate and use that client. For now, if OpenAI client is present, allow fallback.
        if OpenAI is not None:
            cb("info", f"Provider '{provider_norm}' selected but falling back to OpenAI client (if you intended to use the other provider, integrate its SDK).")
            client = OpenAI(api_key=api_key)
        else:
            cb("info", f"Provider '{provider_norm}' selected but no supported client found locally. Please install and configure the provider SDK or use OpenAI.")
            raise RuntimeError(f"No client available for provider '{provider_norm}'")

    answers: List[str] = []

    running_context = (context.strip() if context else "").strip()
    prior_qa = ""

    def is_code_like(text: str) -> bool:
        if not text:
            return False
        return (
            "def " in text
            or "class " in text
            or any(sym in text for sym in [";", "{", "}", "#include", "import ", "console.log", "function "])
        )

    def _save_results(save_path_local: str, save_format_local: str, records: List[dict], append_flag: bool):
        fmt = save_format_local.lower()
        if fmt == "json":
            if append_flag and os.path.exists(save_path_local):
                with open(save_path_local, "r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                        if not isinstance(existing, list):
                            existing = [existing]
                    except Exception:
                        existing = []
                merged = existing + records
                with open(save_path_local, "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
            else:
                with open(save_path_local, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
        elif fmt == "txt":
            mode = "a" if append_flag else "w"
            with open(save_path_local, mode, encoding="utf-8") as f:
                for r in records:
                    f.write(f"Q: {r.get('question')}\nA:\n{r.get('answer')}\n\n---\n\n")
        elif fmt == "csv":
            mode = "a" if append_flag else "w"
            file_exists = os.path.exists(save_path_local)
            with open(save_path_local, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "question", "answer"])
                if not file_exists or not append_flag:
                    writer.writeheader()
                for r in records:
                    writer.writerow({
                        "index": r.get("index"),
                        "question": r.get("question"),
                        "answer": r.get("answer")
                    })
        else:
            raise ValueError(f"Unsupported save_format: {save_format_local}")

    for idx, question in enumerate(questions, start=1):
        combined_context_parts = []
        if running_context:
            combined_context_parts.append(running_context)
        if prior_qa:
            combined_context_parts.append("Previous Q&A:\n" + prior_qa.strip())
        combined_context = "\n\n".join(combined_context_parts).strip()

        code_like = is_code_like(combined_context)

        if combined_context:
            if code_like:
                user_content = (
                    f"The following contains code and prior discussion/context:\n\n"
                    f"{combined_context}\n\n"
                    f"Question: {question}\n\n"
                    "Please respond with two sections:\n"
                    "1. Explanation: Explain what the code does in simple terms.\n"
                    "2. Modified Code: Provide updated code (only the code block) or improvements based on the question.\n"
                )
            else:
                user_content = (
                    "Read the following context and previous Q&A, then answer the question.\n\n"
                    f"Context:\n{combined_context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer:"
                )
        else:
            user_content = (
                "Answer the following question concisely.\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant and skilled code interpreter."},
            {"role": "user", "content": user_content}
        ]

        # Notify UI of question about to be sent
        if show_progress:
            cb("q", f"Q{idx}: {question}")

        response_text = None
        for attempt in range(max_retries + 1):
            try:
                # Note: This code uses the OpenAI client's chat completions signature implemented earlier.
                # If you want to call Mistral / LLaMA SDKs, modify this block to switch on `provider`.
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                response_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < max_retries:
                    cb("info", f"Transient error calling API (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    cb("info", f"Error calling API: {e}")
                    raise

        answer = response_text or ""
        answers.append(answer)

        prior_qa += f"Q: {question}\nA: {answer}\n\n"

        if show_progress:
            cb("a", f"A{idx}: {answer}")

    # Auto-save if requested
    if save_path:
        records = []
        for i, (q, a) in enumerate(zip(questions, answers), start=1):
            records.append({"index": i, "question": q, "answer": a})
        try:
            _save_results(save_path, save_format, records, append_to_file)
            if show_progress:
                cb("info", f"Results saved to {save_path} (format={save_format})")
        except Exception as e:
            cb("info", f"Warning: Failed to save results to {save_path}: {e}")

    return answers


# --- Tkinter GUI ---

class QAGui:
    # popular model options - can be edited/extended
    MODELS = [
        "gpt-5", "gpt-5-mini",
        "gpt-4o", "gpt-4o-mini", "gpt-4o-realtime-preview", "gpt-4o-code",
        "gpt-4o-small", "gpt-3.5-turbo",
        # Popular Mistral / LLaMA-style models (names as commonly referenced)
        "mistral-1", "mistral-1-small",
        "llama-2-70b-chat", "llama-2-13b-chat", "llama-2-7b-chat"
    ]
    LOCKED_TEMP_MODELS = {"gpt-5", "gpt-5-mini"}

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Tilendra's QA Chatbot")
        # Colorful palette
        self.bg = "#1f2b38"
        self.panel_bg = "#253241"
        self.accent = "#2dd4bf"  # teal
        self.info_bg = "#2b2f3a"

        root.configure(bg=self.bg)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.panel_bg)
        style.configure("TLabel", background=self.panel_bg, foreground="white")
        style.configure("TButton", background=self.accent, foreground="black")
        style.configure("TEntry", foreground="black")
        style.configure("TCombobox", foreground="black")
        style.configure("TCheckbutton", background=self.panel_bg, foreground="white")

        frm = ttk.Frame(root, padding="8", style="TFrame")
        frm.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Row 0: Provider, API key, model (dropdown), temperature (slider)
        top_row = ttk.Frame(frm, style="TFrame")
        top_row.grid(row=0, column=0, sticky="ew", pady=(0,8))
        ttk.Label(top_row, text="Provider:").grid(row=0, column=0, sticky="w")
        self.provider_var = tk.StringVar(value="openai")
        self.provider_box = ttk.Combobox(top_row, textvariable=self.provider_var, values=["openai", "mistral", "llama"], width=12, state="readonly")
        self.provider_box.grid(row=0, column=1, sticky="w", padx=(4,8))
        self.provider_box.bind("<<ComboboxSelected>>", lambda e: self.on_provider_change())

        ttk.Label(top_row, text="API Key:").grid(row=0, column=2, sticky="w")
        self.api_key_var = tk.StringVar()
        # show key with bullets
        self.api_key_entry = ttk.Entry(top_row, textvariable=self.api_key_var, width=36, show="•")
        self.api_key_entry.grid(row=0, column=3, sticky="ew", padx=(4,8))

        ttk.Label(top_row, text="Model:").grid(row=0, column=4, sticky="w")
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.model_box = ttk.Combobox(top_row, textvariable=self.model_var, values=self.MODELS, width=22, state="readonly")
        self.model_box.grid(row=0, column=5, sticky="w", padx=(4,0))
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.on_model_change())

        ttk.Label(top_row, text="Temp:").grid(row=0, column=6, sticky="w", padx=(8,0))
        # temperature slider
        self.temp_var = tk.DoubleVar(value=0.3)
        self.temp_scale = tk.Scale(top_row, variable=self.temp_var, from_=0.0, to=1.0, resolution=0.05,
                                   orient="horizontal", length=140, showvalue=False, bg=self.panel_bg, troughcolor="#3a4754",
                                   highlightthickness=0)
        self.temp_scale.grid(row=0, column=7, sticky="w", padx=(4,0))
        self.temp_label = ttk.Label(top_row, text=f"{self.temp_var.get():.2f}")
        self.temp_label.grid(row=0, column=8, sticky="w", padx=(6,0))
        # Update label when slider moves
        self.temp_var.trace_add("write", lambda *args: self.temp_label.configure(text=f"{self.temp_var.get():.2f}"))

        # Row 1: Context (multi-line)
        ttk.Label(frm, text="Context (text or code) - Optional:").grid(row=1, column=0, sticky="w")
        self.context_text = ScrolledText(frm, height=8, background="#0f1720", foreground="#d1d5db", insertbackground="white")
        self.context_text.grid(row=2, column=0, sticky="nsew", pady=(0,8))
        frm.rowconfigure(2, weight=0)

        # Row 3: Questions (one per line)
        ttk.Label(frm, text="Questions (one per line):").grid(row=3, column=0, sticky="w")
        self.questions_text = ScrolledText(frm, height=6, background="#0f1720", foreground="#d1d5db", insertbackground="white")
        self.questions_text.grid(row=4, column=0, sticky="nsew", pady=(0,8))
        frm.rowconfigure(4, weight=0)

        # Row 5: Save options - with slider switch (0 = No Save, 1 = Save)
        save_row = ttk.Frame(frm, style="TFrame")
        save_row.grid(row=5, column=0, sticky="ew", pady=(0,8))
        ttk.Label(save_row, text="Save option:").grid(row=0, column=0, sticky="w")
        self.save_enable_var = tk.IntVar(value=0)
        # Slider switch - 0 or 1
        self.save_slider = tk.Scale(save_row, variable=self.save_enable_var, from_=0, to=1, orient="horizontal", length=80,
                                    showvalue=False, resolution=1, bg=self.panel_bg, troughcolor="#3a4754",
                                    highlightthickness=0, command=lambda v: self.on_save_toggle())
        self.save_slider.grid(row=0, column=1, sticky="w", padx=(4,4))
        self.save_state_label = ttk.Label(save_row, text="No Save")
        self.save_state_label.grid(row=0, column=2, sticky="w", padx=(6,8))

        ttk.Label(save_row, text="Save to file:").grid(row=0, column=3, sticky="w")
        self.save_path_var = tk.StringVar()
        self.save_entry = ttk.Entry(save_row, textvariable=self.save_path_var, width=36)
        self.save_entry.grid(row=0, column=4, sticky="w", padx=(4,4))
        self.browse_btn = ttk.Button(save_row, text="Browse...", command=self.browse_save)
        self.browse_btn.grid(row=0, column=5, sticky="w")

        ttk.Label(save_row, text="Format:").grid(row=0, column=6, sticky="w", padx=(8,0))
        self.save_format_var = tk.StringVar(value="json")
        self.save_format_box = ttk.Combobox(save_row, textvariable=self.save_format_var, values=["json", "txt", "csv"], width=6, state="readonly")
        self.save_format_box.grid(row=0, column=7, sticky="w", padx=(4,0))

        self.append_var = tk.BooleanVar(value=False)
        self.append_check = ttk.Checkbutton(save_row, text="Append", variable=self.append_var)
        self.append_check.grid(row=0, column=8, sticky="w", padx=(8,0))

        # Initially disable save controls (No Save)
        self.set_save_controls_enabled(False)

        # Row 6: Buttons
        btn_row = ttk.Frame(frm, style="TFrame")
        btn_row.grid(row=6, column=0, sticky="ew", pady=(0,8))
        self.run_btn = ttk.Button(btn_row, text="Run", command=self.on_run)
        self.run_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(btn_row, text="Clear Output", command=self.clear_output).grid(row=0, column=1, sticky="w", padx=(8,0))
        ttk.Button(btn_row, text="Copy Output", command=self.copy_output).grid(row=0, column=2, sticky="w", padx=(8,0))
        ttk.Button(btn_row, text="Exit", command=root.quit).grid(row=0, column=3, sticky="w", padx=(8,0))

        # Row 7: Output text (color via tags)
        ttk.Label(frm, text="Output:").grid(row=7, column=0, sticky="w")
        self.output = ScrolledText(frm, height=18, state="normal", background="#071026", foreground="#dbeafe", insertbackground="white")
        self.output.grid(row=8, column=0, sticky="nsew")
        frm.rowconfigure(8, weight=1)

        # Text tags for colorizing
        self.output.tag_config("q", foreground="#7dd3fc", font=("TkDefaultFont", 10, "bold"))
        self.output.tag_config("a", foreground="#86efac", font=("TkDefaultFont", 10))
        self.output.tag_config("info", foreground="#ffd980", font=("TkDefaultFont", 9, "italic"))
        self.output.tag_config("error", foreground="#ff7b7b", font=("TkDefaultFont", 10, "bold"))

        # Worker thread handle
        self.worker_thread: Optional[threading.Thread] = None

        # Ensure temperature locked appropriately for initial model
        self.on_model_change()
        # Ensure provider label updated if needed
        self.on_provider_change()

    def browse_save(self):
        default_ext = f".{self.save_format_var.get()}"
        f = filedialog.asksaveasfilename(defaultextension=default_ext,
                                         filetypes=[("All files", "*.*"), ("JSON", "*.json"), ("Text", "*.txt"), ("CSV", "*.csv")])
        if f:
            self.save_path_var.set(f)

    def set_save_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        # Entry and browse and format and append
        try:
            self.save_entry.configure(state=state)
            self.browse_btn.configure(state=state)
            self.save_format_box.configure(state="readonly" if enabled else "disabled")
            # Checkbutton doesn't have state in ttk before some versions; configure via state map
            if enabled:
                self.append_check.state(["!disabled"])
            else:
                self.append_check.state(["disabled"])
        except Exception:
            pass

    def on_save_toggle(self):
        val = int(self.save_enable_var.get())
        if val:
            self.save_state_label.configure(text="Save to File", foreground=self.accent)
            self.set_save_controls_enabled(True)
        else:
            self.save_state_label.configure(text="No Save", foreground="white")
            self.set_save_controls_enabled(False)

    def on_model_change(self):
        model = (self.model_var.get() or "").strip()
        if model in self.LOCKED_TEMP_MODELS:
            # Lock to 1.0 and disable slider
            self.temp_var.set(1.0)
            try:
                self.temp_scale.configure(state="disabled")
            except Exception:
                pass
        else:
            # Default to 0.3 if the current value was previously 1.0 due to a locked model
            if self.temp_var.get() == 1.0:
                self.temp_var.set(0.3)
            try:
                self.temp_scale.configure(state="normal")
            except Exception:
                pass

    def on_provider_change(self):
        provider = (self.provider_var.get() or "openai").strip().lower()
        # Update API Key label hint depending on provider
        label_text = "API Key:"
        if provider == "openai":
            label_text = "API Key (OpenAI):"
        elif provider == "mistral":
            label_text = "API Key (Mistral):"
        elif provider == "llama":
            label_text = "API Key (LLaMA/local):"
        # Find the label widget and update it (label has fixed grid position row=0 col=2)
        try:
            for child in self.root.children.values():
                pass
        except Exception:
            pass
        # Instead of searching, just update a small tooltip by setting entry's label via near widget
        # For simplicity, place a small text variable in the entry's widget tooltip is not available; so show an info in output.
        self.append_output("info", f"Provider set to '{provider}'. Place the provider API key in the API Key field; it will be exported to the environment variable for that provider when you run.")

    def append_output(self, typ: str, text: str):
        # Insert into text widget; runs in main/UI thread.
        # typ can be 'q', 'a', 'info', 'error'
        if typ not in ("q", "a", "info", "error"):
            typ = "info"
        self.output.configure(state="normal")
        self.output.insert("end", text + "\n\n", typ)
        self.output.see("end")
        self.output.configure(state="disabled")

    def clear_output(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def copy_output(self):
        # Copy the entire output to the clipboard
        try:
            # Temporarily make widget normal to get text reliably
            prev_state = self.output.cget("state")
            self.output.configure(state="normal")
            content = self.output.get("1.0", "end").strip()
            self.output.configure(state=prev_state)
            if content:
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                self.append_output("info", "Output copied to clipboard.")
            else:
                self.append_output("info", "Output is empty; nothing copied.")
        except Exception as e:
            self.append_output("error", f"Failed to copy output: {e}")

    def on_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Running", "A job is already running. Please wait for it to finish.")
            return

        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Missing API Key", "Please enter your API key.")
            return

        questions_raw = self.questions_text.get("1.0", "end").strip()
        if not questions_raw:
            messagebox.showwarning("No Questions", "Please enter at least one question (one per line).")
            return

        questions = [q.strip() for q in questions_raw.splitlines() if q.strip()]
        context = self.context_text.get("1.0", "end").strip()
        model = self.model_var.get().strip() or "gpt-4o"
        provider = (self.provider_var.get() or "openai").strip().lower()
        try:
            temperature = float(self.temp_var.get())
        except Exception:
            temperature = 0.3

        # If save slider indicates save enabled, use path; otherwise no save
        save_enabled = bool(self.save_enable_var.get())
        save_path = self.save_path_var.get().strip() or None
        if not save_enabled:
            save_path = None

        save_format = self.save_format_var.get().strip().lower() or "json"
        append_to_file = bool(self.append_var.get())

        # Disable run button while working
        self.run_btn.configure(state="disabled")
        self.append_output("info", "Starting job...")

        def progress_cb(typ: str, msg: str):
            # This may be called from worker thread; marshal to main thread using after
            self.root.after(0, self.append_output, typ if typ in ("q", "a", "info") else "info", msg)

        def worker():
            try:
                # Export provider API key to environment (ask_questions_with_context will do the export too)
                if api_key:
                    if provider == "openai":
                        os.environ["OPENAI_API_KEY"] = api_key
                    elif provider == "mistral":
                        os.environ["MISTRAL_API_KEY"] = api_key
                    elif provider == "llama":
                        os.environ["LLAMA_API_KEY"] = api_key
                    else:
                        os.environ["API_KEY"] = api_key

                # Call ask_questions_with_context (runs API calls)
                ask_questions_with_context(
                    api_key=api_key,
                    questions=questions,
                    context=context,
                    model=model,
                    temperature=temperature,
                    save_path=save_path,
                    save_format=save_format,
                    colorize=False,    # colorized output handled by GUI, not colorama
                    show_progress=True,
                    append_to_file=append_to_file,
                    max_retries=2,
                    retry_delay=2.0,
                    progress_callback=progress_cb,
                    provider=provider
                )
                self.root.after(0, self.append_output, "info", "Job completed.")
            except Exception as e:
                self.root.after(0, self.append_output, "error", f"Error during run: {e}")
            finally:
                # Re-enable run button in main thread
                self.root.after(0, lambda: self.run_btn.configure(state="normal"))

        # Start worker thread
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()


def main():
    root = tk.Tk()
    app = QAGui(root)
    root.geometry("1150x820")
    root.mainloop()


if __name__ == "__main__":
    main()