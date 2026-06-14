"""
Multimodal Document RAG — public demo (Hugging Face Space, Gradio).

⚠️  COST SAFETY. The OpenAI key lives in this Space as a secret, so every
guardrail that protects spend MUST live here on the server — a browser/front-end
limit is trivially bypassed by calling the Space directly. Defence in depth:

  1. Curated samples are PRE-INDEXED (built by build_samples.py). Serving them
     makes ZERO ingestion/captioning calls — only one cheap answer call.
  2. Uploads are strictly capped: size, page count, and number of images
     captioned, and are limited per session and globally per hour.
  3. Per-session and global rolling-window rate limits on questions + uploads.
  4. Model is gpt-4o-mini only, image detail "low", small max_tokens, ≤2 images
     attached per answer.
  5. DEMO_DISABLED env var is a kill switch. And you MUST set a hard monthly
     budget cap in the OpenAI dashboard as the final backstop (see DEPLOY.md).

The portfolio website calls the api_name="ask" endpoint, which only serves the
curated samples (no upload path), under the global + per-token limits.
"""

import os, json, time, pickle, glob, uuid
from collections import deque

import gradio as gr
import rag_core
from rag_core import (
    make_config, ingest_document, chunk_elements, HybridIndex,
    get_embed_model, get_clip_model, get_reranker,
)

# ── Tunables (env-overridable) ────────────────────────────────────────────────
def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

MODEL                   = os.environ.get("DEMO_MODEL", "gpt-4o-mini")
DEMO_DISABLED           = os.environ.get("DEMO_DISABLED", "").lower() in ("1", "true", "yes")
MAX_QUESTION_CHARS      = _int("MAX_QUESTION_CHARS", 500)
SESSION_MAX_Q           = _int("SESSION_MAX_Q", 20)
SESSION_MAX_UPLOADS     = _int("SESSION_MAX_UPLOADS", 2)
GLOBAL_MAX_Q_PER_HOUR   = _int("GLOBAL_MAX_Q_PER_HOUR", 150)
GLOBAL_MAX_Q_PER_DAY    = _int("GLOBAL_MAX_Q_PER_DAY", 800)
GLOBAL_MAX_UP_PER_HOUR  = _int("GLOBAL_MAX_UP_PER_HOUR", 25)
UPLOAD_MAX_MB           = _int("UPLOAD_MAX_MB", 5)
UPLOAD_MAX_PAGES        = _int("UPLOAD_MAX_PAGES", 8)
UPLOAD_MAX_CAPTIONS     = _int("UPLOAD_MAX_CAPTIONS", 6)
API_MAX_Q_PER_TOKEN_HR  = _int("API_MAX_Q_PER_TOKEN_HR", 30)

API_KEY = os.environ.get("OPENAI_API_KEY", "")

CFG = make_config(
    llm_model=MODEL, caption_model=MODEL,
    use_image_in_answer=True, use_clip_retrieval=True,
    max_answer_images=2, image_detail="low", max_tokens=600,
    top_k_retrieval=20, top_k_rerank=6,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(HERE, "samples")
SAMPLE_DOCS_DIR = os.path.join(HERE, "sample_docs")

# ── Models + sample indexes (built once at startup) ───────────────────────────
EM = get_embed_model(CFG["embedding_model"])
CLIP = get_clip_model(CFG["clip_model"]) if CFG["use_clip_retrieval"] else None
RERANKER = get_reranker(CFG["reranker_model"]) if CFG["use_reranker"] else None

SAMPLE_INDEXES = {}     # sid -> HybridIndex
SAMPLE_MANIFEST = []     # [{id, title, ...}]


def _build_index(chunks):
    idx = HybridIndex(EM, clip_model=CLIP)
    idx.build(chunks)
    return idx


def load_samples():
    """Prefer pre-built pickles; otherwise build text samples at boot (free)."""
    manifest_path = os.path.join(SAMPLES_DIR, "manifest.json")
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
        for m in manifest:
            pkl = os.path.join(SAMPLES_DIR, f"{m['id']}.pkl")
            if os.path.isfile(pkl):
                with open(pkl, "rb") as f:
                    chunks = pickle.load(f)
                SAMPLE_INDEXES[m["id"]] = _build_index(chunks)
                SAMPLE_MANIFEST.append(m)
        if SAMPLE_INDEXES:
            print(f"Loaded {len(SAMPLE_INDEXES)} pre-built sample index(es).")
            return
    # Fallback: ingest text/markdown samples at startup (no API needed).
    for fp in sorted(glob.glob(os.path.join(SAMPLE_DOCS_DIR, "*"))):
        ext = os.path.splitext(fp)[1].lower()
        if ext not in {".txt", ".md", ".pdf", ".docx"}:
            continue
        sid = os.path.splitext(os.path.basename(fp))[0]
        # Only caption images here if a key is present (text samples need none).
        elements, meta = ingest_document(fp, api_key=API_KEY if ext in {".pdf", ".docx"} else "",
                                         extract_images=ext in {".pdf", ".docx"},
                                         caption_model=MODEL, max_caption_images=UPLOAD_MAX_CAPTIONS)
        chunks = chunk_elements(elements, CFG["chunk_size"], CFG["chunk_overlap"])
        SAMPLE_INDEXES[sid] = _build_index(chunks)
        SAMPLE_MANIFEST.append({"id": sid, "title": sid.replace("_", " ").title(),
                                "file_type": meta["file_type"], "pages": meta.get("total_pages", 0),
                                "chunks": len(chunks), "images": meta.get("images_extracted", 0)})
    print(f"Built {len(SAMPLE_INDEXES)} sample index(es) at startup.")


load_samples()

# ── Rate limiting (rolling windows, in-process / shared across sessions) ───────
_GLOBAL_Q_HOUR, _GLOBAL_Q_DAY, _GLOBAL_UP_HOUR = deque(), deque(), deque()
_TOKEN_HITS = {}


def _prune(dq, window):
    cut = time.time() - window
    while dq and dq[0] < cut:
        dq.popleft()


def _global_question_ok():
    _prune(_GLOBAL_Q_HOUR, 3600); _prune(_GLOBAL_Q_DAY, 86400)
    if len(_GLOBAL_Q_HOUR) >= GLOBAL_MAX_Q_PER_HOUR:
        return False, "The demo has hit its hourly question limit. Please try again later."
    if len(_GLOBAL_Q_DAY) >= GLOBAL_MAX_Q_PER_DAY:
        return False, "The demo has hit its daily question limit. Please try again tomorrow."
    return True, ""


def _record_question():
    now = time.time(); _GLOBAL_Q_HOUR.append(now); _GLOBAL_Q_DAY.append(now)


def _global_upload_ok():
    _prune(_GLOBAL_UP_HOUR, 3600)
    if len(_GLOBAL_UP_HOUR) >= GLOBAL_MAX_UP_PER_HOUR:
        return False, "The demo has hit its hourly upload limit. Please use the sample documents."
    return True, ""


def _token_ok(token):
    if not token:
        return True
    dq = _TOKEN_HITS.setdefault(token, deque())
    _prune(dq, 3600)
    return len(dq) < API_MAX_Q_PER_TOKEN_HR


# ── Core query + formatting ───────────────────────────────────────────────────
def _precheck_question(question, session):
    if DEMO_DISABLED:
        return "This demo is temporarily disabled."
    if not API_KEY:
        return "The demo is not configured with an API key right now."
    q = (question or "").strip()
    if not q:
        return "Please enter a question."
    if len(q) > MAX_QUESTION_CHARS:
        return f"Question too long (max {MAX_QUESTION_CHARS} characters)."
    if session.get("q", 0) >= SESSION_MAX_Q:
        return f"You've reached this session's limit of {SESSION_MAX_Q} questions."
    ok, msg = _global_question_ok()
    if not ok:
        return msg
    return None


def _answer(index, question):
    result = rag_core.ask(question, index, CFG, reranker=RERANKER, api_key=API_KEY)
    return result


def _format_sources_md(sources):
    out = ["### Sources"]
    for s in sources:
        img = " 🖼️" if s["has_image"] else ""
        out.append(f"**[Source {s['source_num']}]** {s['citation']}{img}  \n"
                   f"<small>{s['text']}</small>")
    return "\n\n".join(out)


# ── UI handlers ───────────────────────────────────────────────────────────────
def _doc_choices(session):
    choices = [(f"📄 {m['title']}", f"sample:{m['id']}") for m in SAMPLE_MANIFEST]
    for key, meta in session.get("docs", {}).items():
        choices.append((f"⬆️ {meta['title']} (your upload)", key))
    return choices


def on_ask(selected, question, session):
    err = _precheck_question(question, session)
    if err:
        return err, "", session
    if not selected:
        return "Please choose a document first.", "", session
    if selected.startswith("sample:"):
        index = SAMPLE_INDEXES.get(selected.split(":", 1)[1])
    else:
        meta = session.get("docs", {}).get(selected)
        index = meta["index"] if meta else None
    if index is None:
        return "That document is no longer available — pick another.", "", session
    try:
        _record_question()
        session["q"] = session.get("q", 0) + 1
        r = _answer(index, question.strip())
        return r["answer"], _format_sources_md(r["sources"]), session
    except Exception as e:
        return f"Something went wrong answering that: {e}", "", session


def on_upload(file, session):
    if file is None:
        return gr.update(), "", session
    if DEMO_DISABLED:
        return gr.update(), "⚠️ This demo is temporarily disabled.", session
    if not API_KEY:
        return gr.update(), "⚠️ Uploads aren’t available right now (demo not configured).", session
    if session.get("up", 0) >= SESSION_MAX_UPLOADS:
        return gr.update(), f"⚠️ Upload limit for this session is {SESSION_MAX_UPLOADS}.", session
    ok, msg = _global_upload_ok()
    if not ok:
        return gr.update(), f"⚠️ {msg}", session

    path = file if isinstance(file, str) else file.name
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".pdf", ".docx", ".txt", ".md"}:
        return gr.update(), "⚠️ Unsupported file type.", session
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > UPLOAD_MAX_MB:
        return gr.update(), f"⚠️ File is {size_mb:.1f} MB; the demo cap is {UPLOAD_MAX_MB} MB.", session

    # Page-count guard (bounds how many images can be captioned).
    if ext == ".pdf":
        try:
            import fitz
            with fitz.open(path) as d:
                if d.page_count > UPLOAD_MAX_PAGES:
                    return (gr.update(),
                            f"⚠️ Document has {d.page_count} pages; the demo cap is "
                            f"{UPLOAD_MAX_PAGES}. Try a shorter excerpt or use a sample.", session)
        except Exception:
            pass

    try:
        _GLOBAL_UP_HOUR.append(time.time())
        elements, meta = ingest_document(
            path, api_key=API_KEY, extract_images=ext in {".pdf", ".docx"},
            caption_model=MODEL, max_caption_images=UPLOAD_MAX_CAPTIONS)
        chunks = chunk_elements(elements, CFG["chunk_size"], CFG["chunk_overlap"])
        index = _build_index(chunks)
        key = f"upload:{uuid.uuid4().hex[:8]}"
        title = os.path.basename(path)
        session.setdefault("docs", {})[key] = {"title": title, "index": index, "meta": meta}
        session["up"] = session.get("up", 0) + 1
        note = (f"✅ Indexed **{title}** — {meta.get('total_pages',0)} pages, {len(chunks)} chunks, "
                f"{meta.get('images_extracted',0)} image(s) captioned. Select it below and ask.")
        return gr.update(choices=_doc_choices(session), value=key), note, session
    except Exception as e:
        return gr.update(), f"⚠️ Couldn't process that file: {e}", session


# ── Portfolio API: samples only, no uploads ───────────────────────────────────
def api_ask(sample_id, question, token):
    if DEMO_DISABLED or not API_KEY:
        return json.dumps({"error": "Demo unavailable."})
    q = (question or "").strip()
    if not q:
        return json.dumps({"error": "Empty question."})
    if len(q) > MAX_QUESTION_CHARS:
        return json.dumps({"error": f"Question too long (max {MAX_QUESTION_CHARS} chars)."})
    if not _token_ok(token):
        return json.dumps({"error": "Rate limit reached. Please slow down and try again later."})
    ok, msg = _global_question_ok()
    if not ok:
        return json.dumps({"error": msg})
    index = SAMPLE_INDEXES.get((sample_id or "").replace("sample:", ""))
    if index is None:
        return json.dumps({"error": "Unknown sample document."})
    try:
        if token:
            _TOKEN_HITS.setdefault(token, deque()).append(time.time())
        _record_question()
        r = _answer(index, q)
        return json.dumps({
            "answer": r["answer"],
            "sources": [{"source_num": s["source_num"], "citation": s["citation"],
                         "text": s["text"], "has_image": s["has_image"]} for s in r["sources"]],
        })
    except Exception as e:
        return json.dumps({"error": f"Generation failed: {e}"})


def api_samples():
    return json.dumps(SAMPLE_MANIFEST)


# ── UI ────────────────────────────────────────────────────────────────────────
INTRO = """# 📄🔎 Multimodal Document RAG
Ask questions over a document. The system uses **hybrid retrieval** (BGE dense +
numeric-aware BM25 + **CLIP visual** search), a **cross-encoder re-ranker**, and
feeds the **actual chart/table images** to the model at answer time — not just
their text captions. Answers are grounded and cited; out-of-document questions
are refused.

Pick a **sample** below, or upload a short document of your own (capped for cost).
"""

with gr.Blocks(title="Multimodal Document RAG", theme=gr.themes.Soft()) as demo:
    session = gr.State({"q": 0, "up": 0, "docs": {}})
    gr.Markdown(INTRO)

    with gr.Row():
        doc_dd = gr.Dropdown(label="Document", choices=_doc_choices({}),
                             value=(f"sample:{SAMPLE_MANIFEST[0]['id']}" if SAMPLE_MANIFEST else None))
        with gr.Column(scale=0):
            upload = gr.File(label=f"Upload (≤{UPLOAD_MAX_MB}MB, ≤{UPLOAD_MAX_PAGES}p)",
                             file_types=[".pdf", ".docx", ".txt", ".md"], type="filepath")
    upload_note = gr.Markdown()

    question = gr.Textbox(label="Question", placeholder="e.g. What was revenue and how much did it grow?",
                          max_lines=3)
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    sources = gr.Markdown()

    gr.Markdown(
        f"<small>Demo limits: {SESSION_MAX_Q} questions & {SESSION_MAX_UPLOADS} uploads per session, "
        f"model {MODEL}. Built by Salman Alfarisi.</small>")

    ask_btn.click(on_ask, [doc_dd, question, session], [answer, sources, session])
    question.submit(on_ask, [doc_dd, question, session], [answer, sources, session])
    upload.upload(on_upload, [upload, session], [doc_dd, upload_note, session])

    # Hidden endpoints for the portfolio website (programmatic access).
    with gr.Row(visible=False):
        api_sid = gr.Textbox(); api_q = gr.Textbox(); api_tok = gr.Textbox(); api_out = gr.Textbox()
        gr.Button().click(api_ask, [api_sid, api_q, api_tok], api_out, api_name="ask")
        samples_out = gr.Textbox()
        gr.Button().click(lambda: api_samples(), None, samples_out, api_name="samples")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=24).launch()
