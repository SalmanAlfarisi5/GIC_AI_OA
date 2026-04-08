"""
Document Q&A Web Application
Multimodal RAG — supports PDF, DOCX, TXT, and Markdown files.
PDFs: extracts text, tables, and images (captioned via vision model).
"""

import os
import re
import io
import json
import base64
import hashlib
import time
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
import tiktoken

# Conditional imports — not all formats need all libraries
try:
    import fitz  # PyMuPDF — for PDF
except ImportError:
    fitz = None

try:
    import docx  # python-docx — for DOCX
except ImportError:
    docx = None


# ══════════════════════════════════════════════════════════════
# Supported Formats
# ══════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_file_type(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    return None


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════

@dataclass
class PageElement:
    text: str
    page_num: int               # page/section number (1-indexed)
    element_type: str           # 'heading', 'paragraph', 'table', 'image'
    section_title: str = ""
    image_b64: str = ""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_numbers: List[int]
    section_title: str
    token_count: int
    chunk_index: int
    has_image: bool = False

    def citation(self) -> str:
        pages = sorted(set(self.page_numbers))
        if len(pages) == 1:
            p = f"p. {pages[0]}"
        elif len(pages) <= 3:
            p = f"pp. {', '.join(map(str, pages))}"
        else:
            p = f"pp. {pages[0]}-{pages[-1]}"
        sec = f' — "{self.section_title}"' if self.section_title else ""
        img = " [image]" if self.has_image else ""
        return f"[{p}{sec}{img}]"


# ══════════════════════════════════════════════════════════════
# Image Captioning (PDF only)
# ══════════════════════════════════════════════════════════════

IMAGE_CAPTION_PROMPT = (
    "Describe this image from a document in detail. "
    "If it is a chart, graph, or table, describe the data, trends, labels, axes, and key values. "
    "If it is a diagram, describe its structure and relationships. "
    "Be specific about any numbers, percentages, or text visible. "
    "Start with the type of visual (chart, graph, diagram, photo, etc)."
)


def caption_image(image_bytes, api_key, page_num, model="gpt-4o-mini"):
    if not api_key or len(image_bytes) < 5000:
        return None
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ]}],
        )
        return f"[Image on page {page_num}]: {resp.choices[0].message.content}"
    except Exception as e:
        print(f"  ⚠️  Image caption failed p.{page_num}: {e}")
        return None


def extract_page_images(doc, page, page_num, api_key, min_w=100, min_h=100):
    elements = []
    for img_info in page.get_images(full=True):
        try:
            base_image = doc.extract_image(img_info[0])
            if not base_image:
                continue
            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if w < min_w or h < min_h:
                continue
            caption = caption_image(base_image["image"], api_key, page_num)
            if caption:
                ext = base_image.get("ext", "png")
                b64 = base64.b64encode(base_image["image"]).decode("utf-8")
                elements.append(PageElement(
                    text=caption, page_num=page_num, element_type="image",
                    image_b64=f"data:image/{ext};base64,{b64}"))
        except Exception:
            continue
    return elements


# ══════════════════════════════════════════════════════════════
# PDF Ingestion
# ══════════════════════════════════════════════════════════════

def _get_median_font_size(page):
    sizes = []
    for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if b.get("type", 1) != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if len(t) > 2:
                    sizes.extend([span["size"]] * len(t))
    return float(np.median(sizes)) if sizes else 12.0


def _detect_heading_pdf(block, median_size):
    lines = block.get("lines", [])
    if not lines:
        return False, 0
    font_sizes, is_bold = [], False
    for line in lines:
        for span in line.get("spans", []):
            font_sizes.append(span["size"])
            if "bold" in span.get("font", "").lower():
                is_bold = True
    if not font_sizes:
        return False, 0
    ratio = max(font_sizes) / median_size if median_size > 0 else 1.0
    full_text = " ".join(
        span["text"] for line in lines for span in line.get("spans", [])).strip()
    if len(full_text) > 200 or len(full_text) < 2:
        return False, 0
    if ratio >= 1.5 or (ratio >= 1.3 and is_bold):
        return True, 1
    elif ratio >= 1.2 or (is_bold and len(full_text) < 80):
        return True, 2
    return False, 0


def ingest_pdf(filepath, api_key="", extract_images=True):
    doc = fitz.open(filepath)
    elements, current_section, image_count = [], "", 0
    metadata = {
        "title": doc.metadata.get("title", ""),
        "total_pages": len(doc),
        "file_name": os.path.basename(filepath),
        "file_type": "pdf",
    }
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        median_fs = _get_median_font_size(page)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # Tables
        try:
            for table in page.find_tables():
                df = table.to_pandas()
                if not df.empty:
                    elements.append(PageElement(text=df.to_markdown(index=False),
                        page_num=page_num, element_type="table",
                        section_title=current_section))
        except Exception:
            pass

        # Text blocks
        for block in blocks:
            if block.get("type", 1) != 0:
                continue
            text = ""
            for line in block.get("lines", []):
                text += "".join(s["text"] for s in line.get("spans", [])) + "\n"
            text = text.strip()
            if not text or len(text) < 3:
                continue
            is_heading, _ = _detect_heading_pdf(block, median_fs)
            if is_heading:
                current_section = text.strip()
                elements.append(PageElement(text=text, page_num=page_num,
                    element_type="heading", section_title=current_section))
            else:
                elements.append(PageElement(text=text, page_num=page_num,
                    element_type="paragraph", section_title=current_section))

        # Images
        if extract_images and api_key:
            for img_elem in extract_page_images(doc, page, page_num, api_key):
                img_elem.section_title = current_section
                elements.append(img_elem)
                image_count += 1

    doc.close()
    metadata["images_extracted"] = image_count
    return elements, metadata


# ══════════════════════════════════════════════════════════════
# DOCX Ingestion
# ══════════════════════════════════════════════════════════════

def ingest_docx(filepath):
    doc = docx.Document(filepath)
    elements = []
    current_section = ""
    page_estimate = 1

    metadata = {
        "title": doc.core_properties.title or "",
        "total_pages": 0,  # DOCX doesn't have real pages
        "file_name": os.path.basename(filepath),
        "file_type": "docx",
        "images_extracted": 0,
    }

    line_count = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Estimate page number (~45 lines per page)
        line_count += max(1, len(text) // 80)
        page_estimate = (line_count // 45) + 1

        # Detect headings from Word's built-in styles
        style = para.style.name.lower() if para.style else ""
        is_heading = "heading" in style

        if is_heading:
            current_section = text
            elements.append(PageElement(text=text, page_num=page_estimate,
                element_type="heading", section_title=current_section))
        else:
            elements.append(PageElement(text=text, page_num=page_estimate,
                element_type="paragraph", section_title=current_section))

    # Extract tables
    for table in doc.tables:
        rows = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        if rows and len(rows) > 1:
            # Format as markdown table
            header = "| " + " | ".join(rows[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
            md_table = f"{header}\n{separator}\n{body}"
            elements.append(PageElement(text=md_table, page_num=page_estimate,
                element_type="table", section_title=current_section))

    metadata["total_pages"] = page_estimate
    return elements, metadata


# ══════════════════════════════════════════════════════════════
# TXT / Markdown Ingestion
# ══════════════════════════════════════════════════════════════

# Regex patterns for detecting headings in plain text and markdown
MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')                    # # Heading
UNDERLINE_H1_RE = re.compile(r'^={3,}\s*$')                         # ====== (after text)
UNDERLINE_H2_RE = re.compile(r'^-{3,}\s*$')                         # ------ (after text)
CAPS_HEADING_RE = re.compile(r'^[A-Z][A-Z\s\d:]{5,80}$')            # ALL CAPS LINE
NUMBERED_HEADING_RE = re.compile(r'^\d+(\.\d+)*\.?\s+[A-Z].{3,80}$')  # 1.2 Some Heading


def ingest_text(filepath, is_markdown=False):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    lines = raw_text.split("\n")
    elements = []
    current_section = ""
    page_estimate = 1
    line_count = 0
    buf = []

    metadata = {
        "title": "",
        "total_pages": 0,
        "file_name": os.path.basename(filepath),
        "file_type": "md" if is_markdown else "txt",
        "images_extracted": 0,
    }

    def flush_buf():
        nonlocal buf
        if buf:
            text = "\n".join(buf).strip()
            if text:
                elements.append(PageElement(text=text, page_num=page_estimate,
                    element_type="paragraph", section_title=current_section))
            buf = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        line_count += 1
        page_estimate = (line_count // 50) + 1

        # Skip empty lines — flush paragraph buffer
        if not stripped:
            flush_buf()
            continue

        is_heading = False

        # Markdown heading: # Heading
        if is_markdown:
            m = MD_HEADING_RE.match(stripped)
            if m:
                flush_buf()
                current_section = m.group(2).strip()
                elements.append(PageElement(text=current_section,
                    page_num=page_estimate, element_type="heading",
                    section_title=current_section))
                is_heading = True

        # Underline-style heading (works for both TXT and MD)
        if not is_heading and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if UNDERLINE_H1_RE.match(next_line) or UNDERLINE_H2_RE.match(next_line):
                flush_buf()
                current_section = stripped
                elements.append(PageElement(text=current_section,
                    page_num=page_estimate, element_type="heading",
                    section_title=current_section))
                is_heading = True

        # ALL CAPS heading (for plain text)
        if not is_heading and not is_markdown:
            if CAPS_HEADING_RE.match(stripped) or NUMBERED_HEADING_RE.match(stripped):
                flush_buf()
                current_section = stripped
                elements.append(PageElement(text=current_section,
                    page_num=page_estimate, element_type="heading",
                    section_title=current_section))
                is_heading = True

        if not is_heading:
            buf.append(line)

    flush_buf()
    metadata["total_pages"] = page_estimate
    return elements, metadata


# ══════════════════════════════════════════════════════════════
# Unified Ingestion Router
# ══════════════════════════════════════════════════════════════

def ingest_document(filepath, api_key="", extract_images=True):
    """
    Route to the correct parser based on file extension.
    Returns (elements, metadata) regardless of format.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        if fitz is None:
            raise ImportError("PyMuPDF (pymupdf) is required for PDF files. pip install pymupdf")
        return ingest_pdf(filepath, api_key=api_key, extract_images=extract_images)

    elif ext == ".docx":
        if docx is None:
            raise ImportError("python-docx is required for DOCX files. pip install python-docx")
        return ingest_docx(filepath)

    elif ext == ".md":
        return ingest_text(filepath, is_markdown=True)

    elif ext == ".txt":
        return ingest_text(filepath, is_markdown=False)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")


# ══════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(tokenizer.encode(text))

def _make_chunk(texts, pages, section_title, idx, has_image=False):
    combined = "\n\n".join(texts)
    cid = hashlib.md5(f"{idx}:{combined[:50]}".encode()).hexdigest()[:10]
    return Chunk(chunk_id=cid, text=combined, page_numbers=list(pages),
                 section_title=section_title, token_count=count_tokens(combined),
                 chunk_index=idx, has_image=has_image)

def chunk_elements(elements, chunk_size=512, overlap=64):
    chunks, chunk_idx = [], 0
    sections, cur_title, cur_elems = [], "", []
    for elem in elements:
        if elem.element_type == "heading":
            if cur_elems: sections.append((cur_title, cur_elems))
            cur_title = elem.text.strip(); cur_elems = [elem]
        else: cur_elems.append(elem)
    if cur_elems: sections.append((cur_title, cur_elems))

    for sec_title, sec_elems in sections:
        buf_t, buf_p, buf_tok, buf_img = [], [], 0, False
        for elem in sec_elems:
            et = count_tokens(elem.text); is_img = elem.element_type == "image"
            if et > chunk_size:
                if buf_t:
                    chunks.append(_make_chunk(buf_t, buf_p, sec_title, chunk_idx, buf_img))
                    chunk_idx += 1; buf_t, buf_p, buf_tok, buf_img = [], [], 0, False
                tokens = tokenizer.encode(elem.text)
                for i in range(0, len(tokens), chunk_size):
                    chunks.append(_make_chunk([tokenizer.decode(tokens[i:i+chunk_size])],
                        [elem.page_num], sec_title, chunk_idx, is_img)); chunk_idx += 1
                continue
            if buf_tok + et > chunk_size and buf_t:
                chunks.append(_make_chunk(buf_t, buf_p, sec_title, chunk_idx, buf_img))
                chunk_idx += 1
                keep_t, keep_p, keep_tok = [], [], 0
                for t, p in zip(reversed(buf_t), reversed(buf_p)):
                    tt = count_tokens(t)
                    if keep_tok + tt > overlap: break
                    keep_t.insert(0, t); keep_p.insert(0, p); keep_tok += tt
                buf_t, buf_p, buf_tok, buf_img = keep_t, keep_p, keep_tok, False
            buf_t.append(elem.text); buf_p.append(elem.page_num); buf_tok += et
            if is_img: buf_img = True
        if buf_t:
            chunks.append(_make_chunk(buf_t, buf_p, sec_title, chunk_idx, buf_img)); chunk_idx += 1
    return [c for c in chunks if c.token_count >= 30]


# ══════════════════════════════════════════════════════════════
# Hybrid Index & Generation
# ══════════════════════════════════════════════════════════════

class HybridIndex:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.chunks, self.faiss_index, self.bm25 = [], None, None

    def build(self, chunks):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embs = self.embed_model.encode(texts, show_progress_bar=False,
                                       batch_size=32, normalize_embeddings=True)
        self.faiss_index = faiss.IndexFlatIP(embs.shape[1])
        self.faiss_index.add(embs.astype(np.float32))
        self.bm25 = BM25Okapi([self._tok(t) for t in texts])

    @staticmethod
    def _tok(text):
        return [t for t in re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split() if len(t) > 1]

    def search(self, query, top_k=15, rrf_k=60):
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        _, ids = self.faiss_index.search(q_emb, top_k)
        dense = [(int(i), r) for r, i in enumerate(ids[0]) if i >= 0]
        scores = self.bm25.get_scores(self._tok(query))
        sparse = [(int(i), r) for r, i in enumerate(np.argsort(scores)[::-1][:top_k]) if scores[i] > 0]
        rrf = {}
        for idx, r in dense: rrf[idx] = rrf.get(idx, 0) + 1.0/(rrf_k+r+1)
        for idx, r in sparse: rrf[idx] = rrf.get(idx, 0) + 1.0/(rrf_k+r+1)
        return [(self.chunks[i], s) for i, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]


SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer questions based
ONLY on the provided source passages.

RULES:
1. Use ONLY information from the provided sources. No external knowledge.
2. Cite sources using [Source N] notation.
3. If multiple sources support a claim, cite all: [Source 1, Source 3].
4. If the sources don't contain enough information, say:
   "The provided document sections do not contain sufficient information to answer this question."
5. Do not speculate beyond what the sources explicitly state.
6. For numbers, quote exact figures from the sources.
7. Be concise but complete.
8. Some sources contain descriptions of images, charts, or graphs from the document.
   Treat these descriptions as factual content and cite them normally."""


def generate_answer(query, passages, api_key, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)
    ctx = "\n\n---\n\n".join(
        f"[Source {i+1}] {c.citation()}\n{c.text}" for i, (c, _) in enumerate(passages))
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE PASSAGES:\n{ctx}\n\nQUESTION: {query}\n\nProvide a precise, well-cited answer."},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{
            "source_num": i+1, "citation": c.citation(),
            "text": c.text[:300]+("..." if len(c.text)>300 else ""),
            "score": round(s, 4), "has_image": c.has_image,
        } for i, (c, s) in enumerate(passages)],
        "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                  "completion_tokens": resp.usage.completion_tokens},
    }


# ══════════════════════════════════════════════════════════════
# Flask App
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

state = {"embed_model": None, "index": None, "metadata": None, "num_chunks": 0, "ready": False}
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "docqa_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_embed_model():
    if state["embed_model"] is None:
        print("Loading embedding model...")
        state["embed_model"] = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("✅ Loaded.")
    return state["embed_model"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    file_type = get_file_type(file.filename)
    if not file_type:
        return jsonify({"error": f"Unsupported format. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"}), 400

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)
    api_key = os.environ.get("OPENAI_API_KEY", "")

    try:
        t0 = time.time()
        elements, metadata = ingest_document(filepath, api_key=api_key, extract_images=True)
        chunks = chunk_elements(elements, chunk_size=512, overlap=64)
        index = HybridIndex(get_embed_model())
        index.build(chunks)
        state.update({"index": index, "metadata": metadata, "num_chunks": len(chunks), "ready": True})

        types = {}
        for e in elements: types[e.element_type] = types.get(e.element_type, 0) + 1

        return jsonify({
            "success": True, "filename": file.filename, "file_type": file_type,
            "pages": metadata["total_pages"], "elements": len(elements),
            "chunks": len(chunks), "images": metadata.get("images_extracted", 0),
            "element_types": types, "time": round(time.time()-t0, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    if not state["ready"]:
        return jsonify({"error": "No document loaded. Upload a file first."}), 400
    query = request.get_json().get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500
    try:
        t0 = time.time()
        passages = state["index"].search(query, top_k=15)[:6]
        result = generate_answer(query, passages, api_key)
        result.update({"latency": round(time.time()-t0, 2), "question": query})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    if state["ready"]:
        return jsonify({"ready": True, "filename": state["metadata"]["file_name"],
                        "pages": state["metadata"]["total_pages"],
                        "chunks": state["num_chunks"],
                        "images": state["metadata"].get("images_extracted", 0),
                        "file_type": state["metadata"].get("file_type", "")})
    return jsonify({"ready": False})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Document Q&A — Multimodal")
    print("  Supports: PDF, DOCX, TXT, Markdown")
    print("="*50)
    print(f"  http://localhost:5000")
    print(f"  API key: {'✅' if os.environ.get('OPENAI_API_KEY') else '❌'}")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)