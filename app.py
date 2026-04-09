"""
Document Q&A Web Application
Multimodal RAG — supports PDF, DOCX, TXT, and Markdown.
Images extracted from PDF and DOCX, captioned via GPT-4o-mini vision.
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

try:
    import fitz
except ImportError:
    fitz = None

try:
    import docx
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
except ImportError:
    docx = None


# ══════════════════════════════════════════════════════════════
# Supported Formats
# ══════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

def get_file_type(filename):
    ext = Path(filename).suffix.lower()
    return ext if ext in SUPPORTED_EXTENSIONS else None


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════

@dataclass
class PageElement:
    text: str
    page_num: int
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

    def citation(self):
        pages = sorted(set(self.page_numbers))
        p = f"p. {pages[0]}" if len(pages) == 1 else f"pp. {pages[0]}-{pages[-1]}"
        sec = f' — "{self.section_title}"' if self.section_title else ""
        img = " [image]" if self.has_image else ""
        return f"[{p}{sec}{img}]"


# ══════════════════════════════════════════════════════════════
# Image Captioning (shared by PDF and DOCX)
# ══════════════════════════════════════════════════════════════

IMAGE_CAPTION_PROMPT = (
    "Describe this image from a document in detail. "
    "If it is a chart, graph, or table, describe the data, trends, labels, axes, and key values. "
    "If it is a diagram, describe its structure and relationships. "
    "Be specific about any numbers, percentages, or text visible. "
    "Start with the type of visual (chart, graph, diagram, photo, etc)."
)

# Common image MIME types
MIME_MAP = {
    "png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
    "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff",
    "webp": "image/webp", "emf": "image/emf", "wmf": "image/wmf",
}


def caption_image(image_bytes, api_key, page_num, model="gpt-4o-mini",
                  mime_type="image/png"):
    """Send image to GPT-4o-mini vision. Shared by PDF and DOCX pipelines."""
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
                    "url": f"data:{mime_type};base64,{b64}", "detail": "high"}},
            ]}],
        )
        return f"[Image on page {page_num}]: {resp.choices[0].message.content}"
    except Exception as e:
        print(f"  ⚠️  Caption failed p.{page_num}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# PDF Ingestion + Images
# ══════════════════════════════════════════════════════════════

def _extract_pdf_images(doc, page, page_num, api_key, min_w=100, min_h=100):
    elements = []
    for img_info in page.get_images(full=True):
        try:
            bi = doc.extract_image(img_info[0])
            if not bi:
                continue
            if bi.get("width", 0) < min_w or bi.get("height", 0) < min_h:
                continue
            ext = bi.get("ext", "png")
            mime = MIME_MAP.get(ext, "image/png")
            cap = caption_image(bi["image"], api_key, page_num, mime_type=mime)
            if cap:
                b64 = base64.b64encode(bi["image"]).decode("utf-8")
                elements.append(PageElement(
                    text=cap, page_num=page_num, element_type="image",
                    image_b64=f"data:{mime};base64,{b64}"))
        except Exception:
            continue
    return elements


def _median_fs(page):
    sizes = []
    for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if b.get("type", 1) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s.get("text", "").strip()
                if len(t) > 2:
                    sizes.extend([s["size"]] * len(t))
    return float(np.median(sizes)) if sizes else 12.0


def _is_heading_pdf(block, med):
    lines = block.get("lines", [])
    if not lines:
        return False
    fs, bold = [], False
    for l in lines:
        for s in l.get("spans", []):
            fs.append(s["size"])
            if "bold" in s.get("font", "").lower():
                bold = True
    if not fs:
        return False
    r = max(fs) / med if med > 0 else 1.0
    txt = " ".join(s["text"] for l in lines for s in l.get("spans", [])).strip()
    if len(txt) > 200 or len(txt) < 2:
        return False
    return r >= 1.5 or (r >= 1.3 and bold) or r >= 1.2 or (bold and len(txt) < 80)


def ingest_pdf(fp, api_key="", extract_images=True):
    doc = fitz.open(fp)
    elems, sec, ic = [], "", 0
    meta = {"title": doc.metadata.get("title", ""), "total_pages": len(doc),
            "file_name": os.path.basename(fp), "file_type": "pdf"}
    for pi in range(len(doc)):
        page = doc[pi]; pn = pi + 1; med = _median_fs(page)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        try:
            for t in page.find_tables():
                df = t.to_pandas()
                if not df.empty:
                    elems.append(PageElement(df.to_markdown(index=False), pn, "table", sec))
        except Exception:
            pass
        for b in blocks:
            if b.get("type", 1) != 0:
                continue
            txt = ""
            for l in b.get("lines", []):
                txt += "".join(s["text"] for s in l.get("spans", [])) + "\n"
            txt = txt.strip()
            if not txt or len(txt) < 3:
                continue
            if _is_heading_pdf(b, med):
                sec = txt.strip()
                elems.append(PageElement(txt, pn, "heading", sec))
            else:
                elems.append(PageElement(txt, pn, "paragraph", sec))
        if extract_images and api_key:
            for ie in _extract_pdf_images(doc, page, pn, api_key):
                ie.section_title = sec; elems.append(ie); ic += 1
    doc.close()
    meta["images_extracted"] = ic
    return elems, meta


# ══════════════════════════════════════════════════════════════
# DOCX Ingestion + Images
# ══════════════════════════════════════════════════════════════

def _extract_docx_images(document, api_key, page_estimate):
    """
    Extract images from a DOCX file and caption them.

    DOCX stores images as relationships in the OPC package.
    We access them via document.part.rels, filtering for image relationships.
    """
    elements = []
    if not api_key:
        return elements

    try:
        for rel_id, rel in document.part.rels.items():
            # Filter for image relationships
            if "image" not in rel.reltype:
                continue

            try:
                image_part = rel.target_part
                image_bytes = image_part.blob
                content_type = image_part.content_type  # e.g., 'image/png'

                # Skip tiny images (icons, bullets)
                if len(image_bytes) < 5000:
                    continue

                # Determine MIME type
                mime = content_type if content_type else "image/png"

                cap = caption_image(image_bytes, api_key, page_estimate,
                                    mime_type=mime)
                if cap:
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    elements.append(PageElement(
                        text=cap, page_num=page_estimate,
                        element_type="image",
                        image_b64=f"data:{mime};base64,{b64}"))

            except Exception:
                continue
    except Exception:
        pass

    return elements


def ingest_docx(fp, api_key="", extract_images=True):
    d = docx.Document(fp)
    elems, sec, lc, ic = [], "", 0, 0
    meta = {"title": d.core_properties.title or "", "total_pages": 0,
            "file_name": os.path.basename(fp), "file_type": "docx"}

    for p in d.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        lc += max(1, len(txt) // 80)
        pn = lc // 45 + 1
        sty = p.style.name.lower() if p.style else ""
        if "heading" in sty:
            sec = txt
            elems.append(PageElement(txt, pn, "heading", sec))
        else:
            elems.append(PageElement(txt, pn, "paragraph", sec))

    # Tables
    for tbl in d.tables:
        rows = [[c.text.strip() for c in r.cells] for r in tbl.rows]
        if len(rows) > 1:
            hdr = "| " + " | ".join(rows[0]) + " |"
            sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
            elems.append(PageElement(f"{hdr}\n{sep}\n{body}",
                                     lc // 45 + 1, "table", sec))

    # Images
    if extract_images and api_key:
        page_est = lc // 45 + 1
        img_elems = _extract_docx_images(d, api_key, page_est)
        for ie in img_elems:
            ie.section_title = sec
            elems.append(ie)
            ic += 1

    meta["total_pages"] = lc // 45 + 1
    meta["images_extracted"] = ic
    return elems, meta


# ══════════════════════════════════════════════════════════════
# TXT / Markdown Ingestion
# ══════════════════════════════════════════════════════════════

MD_H = re.compile(r'^(#{1,6})\s+(.+)$')
CAPS_H = re.compile(r'^[A-Z][A-Z\s\d:]{5,80}$')
NUM_H = re.compile(r'^\d+(\.\d+)*\.?\s+[A-Z].{3,80}$')
UL_H1 = re.compile(r'^={3,}\s*$')
UL_H2 = re.compile(r'^-{3,}\s*$')


def ingest_text(fp, is_md=False):
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines = raw.split("\n")
    elems, sec, lc, buf = [], "", 0, []
    meta = {"title": "", "total_pages": 0, "file_name": os.path.basename(fp),
            "file_type": "md" if is_md else "txt", "images_extracted": 0}

    def flush():
        nonlocal buf
        if buf:
            t = "\n".join(buf).strip()
            if t:
                elems.append(PageElement(t, lc // 50 + 1, "paragraph", sec))
            buf = []

    for i, line in enumerate(lines):
        s = line.strip(); lc += 1; heading = False
        if not s:
            flush(); continue
        if is_md:
            m = MD_H.match(s)
            if m:
                flush(); sec = m.group(2).strip()
                elems.append(PageElement(sec, lc // 50 + 1, "heading", sec))
                heading = True
        if not heading and i + 1 < len(lines):
            nx = lines[i + 1].strip()
            if UL_H1.match(nx) or UL_H2.match(nx):
                flush(); sec = s
                elems.append(PageElement(sec, lc // 50 + 1, "heading", sec))
                heading = True
        if not heading and not is_md:
            if CAPS_H.match(s) or NUM_H.match(s):
                flush(); sec = s
                elems.append(PageElement(sec, lc // 50 + 1, "heading", sec))
                heading = True
        if not heading:
            buf.append(line)
    flush()
    meta["total_pages"] = lc // 50 + 1
    return elems, meta


# ══════════════════════════════════════════════════════════════
# Unified Router
# ══════════════════════════════════════════════════════════════

def ingest_document(filepath, api_key="", extract_images=True):
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        if fitz is None:
            raise ImportError("pymupdf required for PDF. pip install pymupdf")
        return ingest_pdf(filepath, api_key, extract_images)
    elif ext == ".docx":
        if docx is None:
            raise ImportError("python-docx required for DOCX. pip install python-docx")
        return ingest_docx(filepath, api_key, extract_images)
    elif ext == ".md":
        return ingest_text(filepath, is_md=True)
    elif ext == ".txt":
        return ingest_text(filepath, is_md=False)
    else:
        raise ValueError(f"Unsupported: {ext}. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}")


# ══════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(tokenizer.encode(text))

def _make_chunk(texts, pages, sec, idx, has_img=False):
    combined = "\n\n".join(texts)
    cid = hashlib.md5(f"{idx}:{combined[:50]}".encode()).hexdigest()[:10]
    return Chunk(cid, combined, list(pages), sec, count_tokens(combined), idx, has_img)

def chunk_elements(elements, cs=512, ov=64):
    chunks, ci = [], 0
    secs, ct, ce = [], "", []
    for e in elements:
        if e.element_type == "heading":
            if ce: secs.append((ct, ce))
            ct = e.text.strip(); ce = [e]
        else:
            ce.append(e)
    if ce:
        secs.append((ct, ce))
    for st, se in secs:
        bt, bp, bk, bi = [], [], 0, False
        for e in se:
            et = count_tokens(e.text); ii = e.element_type == "image"
            if et > cs:
                if bt:
                    chunks.append(_make_chunk(bt, bp, st, ci, bi)); ci += 1
                    bt, bp, bk, bi = [], [], 0, False
                toks = tokenizer.encode(e.text)
                for i in range(0, len(toks), cs):
                    chunks.append(_make_chunk([tokenizer.decode(toks[i:i+cs])],
                                              [e.page_num], st, ci, ii)); ci += 1
                continue
            if bk + et > cs and bt:
                chunks.append(_make_chunk(bt, bp, st, ci, bi)); ci += 1
                kt, kp, kk = [], [], 0
                for t, p in zip(reversed(bt), reversed(bp)):
                    tt = count_tokens(t)
                    if kk + tt > ov: break
                    kt.insert(0, t); kp.insert(0, p); kk += tt
                bt, bp, bk, bi = kt, kp, kk, False
            bt.append(e.text); bp.append(e.page_num); bk += et
            if ii: bi = True
        if bt:
            chunks.append(_make_chunk(bt, bp, st, ci, bi)); ci += 1
    return [c for c in chunks if c.token_count >= 30]


# ══════════════════════════════════════════════════════════════
# Hybrid Index & Generation
# ══════════════════════════════════════════════════════════════

class HybridIndex:
    def __init__(self, em):
        self.em = em; self.chunks = []; self.fi = None; self.bm = None

    def build(self, chunks):
        self.chunks = chunks; texts = [c.text for c in chunks]
        embs = self.em.encode(texts, show_progress_bar=False,
                              batch_size=32, normalize_embeddings=True)
        self.fi = faiss.IndexFlatIP(embs.shape[1])
        self.fi.add(embs.astype(np.float32))
        self.bm = BM25Okapi([self._t(t) for t in texts])

    @staticmethod
    def _t(text):
        return [t for t in re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split() if len(t) > 1]

    def search(self, q, top_k=15, rrf_k=60):
        qe = self.em.encode([q], normalize_embeddings=True).astype(np.float32)
        _, ids = self.fi.search(qe, top_k)
        dense = [(int(i), r) for r, i in enumerate(ids[0]) if i >= 0]
        sc = self.bm.get_scores(self._t(q))
        sparse = [(int(i), r) for r, i in enumerate(np.argsort(sc)[::-1][:top_k]) if sc[i] > 0]
        rrf = {}
        for i, r in dense: rrf[i] = rrf.get(i, 0) + 1.0 / (rrf_k + r + 1)
        for i, r in sparse: rrf[i] = rrf.get(i, 0) + 1.0 / (rrf_k + r + 1)
        return [(self.chunks[i], s) for i, s in
                sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]


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
8. Some sources contain descriptions of images, charts, or graphs.
   Treat these as factual content and cite them normally."""


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
            "source_num": i + 1, "citation": c.citation(),
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
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

state = {"embed_model": None, "index": None, "metadata": None,
         "num_chunks": 0, "ready": False}
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
        exts = ", ".join(SUPPORTED_EXTENSIONS)
        return jsonify({"error": f"Unsupported format. Accepted: {exts}"}), 400

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)
    api_key = os.environ.get("OPENAI_API_KEY", "")

    try:
        t0 = time.time()
        elements, metadata = ingest_document(filepath, api_key=api_key,
                                              extract_images=True)
        chunks = chunk_elements(elements, cs=512, ov=64)
        index = HybridIndex(get_embed_model())
        index.build(chunks)
        state.update({"index": index, "metadata": metadata,
                      "num_chunks": len(chunks), "ready": True})

        types = {}
        for e in elements:
            types[e.element_type] = types.get(e.element_type, 0) + 1

        return jsonify({
            "success": True, "filename": file.filename,
            "file_type": file_type,
            "pages": metadata["total_pages"],
            "elements": len(elements),
            "chunks": len(chunks),
            "images": metadata.get("images_extracted", 0),
            "element_types": types,
            "time": round(time.time() - t0, 1),
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
        result.update({"latency": round(time.time() - t0, 2), "question": query})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    if state["ready"]:
        return jsonify({
            "ready": True, "filename": state["metadata"]["file_name"],
            "pages": state["metadata"]["total_pages"],
            "chunks": state["num_chunks"],
            "images": state["metadata"].get("images_extracted", 0),
            "file_type": state["metadata"].get("file_type", ""),
        })
    return jsonify({"ready": False})

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Document Q&A — Multimodal")
    print("  Supports: PDF, DOCX, TXT, Markdown")
    print("  Images: PDF + DOCX")
    print("=" * 50)
    print(f"  http://localhost:5000")
    print(f"  API key: {'✅' if os.environ.get('OPENAI_API_KEY') else '❌'}")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)