"""
Document Q&A Web Application
Flask app with multimodal PDF processing — extracts text, tables, AND images.
Images are captioned using GPT-4o-mini vision and indexed alongside text.
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

from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
import tiktoken

# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════

@dataclass
class PageElement:
    text: str
    page_num: int
    element_type: str           # 'heading', 'paragraph', 'table', 'image'
    section_title: str = ""
    image_b64: str = ""         # base64 data URI (only for type='image')


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
# Image Captioning via Vision Model
# ══════════════════════════════════════════════════════════════

IMAGE_CAPTION_PROMPT = (
    "Describe this image from a document in detail. "
    "If it is a chart, graph, or table, describe the data, trends, labels, axes, and key values. "
    "If it is a diagram, describe its structure and relationships. "
    "Be specific about any numbers, percentages, or text visible. "
    "Start with the type of visual (chart, graph, diagram, photo, etc)."
)


def caption_image(image_bytes: bytes, api_key: str, page_num: int,
                  model: str = "gpt-4o-mini") -> Optional[str]:
    """Send an image to GPT-4o-mini vision for a detailed description."""
    if not api_key or len(image_bytes) < 5000:
        return None  # skip tiny images (icons, bullets)

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}", "detail": "high",
                    }},
                ],
            }],
        )
        caption = response.choices[0].message.content
        return f"[Image on page {page_num}]: {caption}"
    except Exception as e:
        print(f"  ⚠️  Image captioning failed p.{page_num}: {e}")
        return None


def extract_page_images(doc, page, page_num, api_key,
                        min_width=100, min_height=100) -> List[PageElement]:
    """Extract images from a page, caption them, return as PageElements."""
    elements = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            if not base_image:
                continue
            image_bytes = base_image["image"]
            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if w < min_width or h < min_height:
                continue

            caption = caption_image(image_bytes, api_key, page_num)
            if caption:
                ext = base_image.get("ext", "png")
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                elements.append(PageElement(
                    text=caption, page_num=page_num, element_type="image",
                    image_b64=f"data:image/{ext};base64,{b64}",
                ))
        except Exception:
            continue
    return elements


# ══════════════════════════════════════════════════════════════
# PDF Ingestion (text + tables + images)
# ══════════════════════════════════════════════════════════════

def get_median_font_size(page) -> float:
    sizes = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for b in blocks:
        if b.get("type", 1) != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if len(t) > 2:
                    sizes.extend([span["size"]] * len(t))
    return float(np.median(sizes)) if sizes else 12.0


def detect_heading(block, median_size):
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
        span["text"] for line in lines for span in line.get("spans", [])
    ).strip()
    if len(full_text) > 200 or len(full_text) < 2:
        return False, 0
    if ratio >= 1.5 or (ratio >= 1.3 and is_bold):
        return True, 1
    elif ratio >= 1.2 or (is_bold and len(full_text) < 80):
        return True, 2
    return False, 0


def extract_block_text(block):
    text = ""
    for line in block.get("lines", []):
        text += "".join(span["text"] for span in line.get("spans", [])) + "\n"
    return text.strip()


def ingest_pdf(pdf_path, api_key="", extract_images_flag=True):
    """Parse PDF → text + tables + images (captioned via vision model)."""
    doc = fitz.open(pdf_path)
    elements, current_section, image_count = [], "", 0
    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "total_pages": len(doc),
        "file_name": os.path.basename(pdf_path),
    }

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        median_fs = get_median_font_size(page)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # Tables
        try:
            for table in page.find_tables():
                df = table.to_pandas()
                if not df.empty:
                    elements.append(PageElement(
                        text=df.to_markdown(index=False), page_num=page_num,
                        element_type="table", section_title=current_section,
                    ))
        except Exception:
            pass

        # Text blocks
        for block in blocks:
            if block.get("type", 1) != 0:
                continue
            text = extract_block_text(block)
            if not text or len(text) < 3:
                continue
            is_heading, level = detect_heading(block, median_fs)
            if is_heading:
                current_section = text.strip()
                elements.append(PageElement(text=text, page_num=page_num,
                    element_type="heading", section_title=current_section))
            else:
                elements.append(PageElement(text=text, page_num=page_num,
                    element_type="paragraph", section_title=current_section))

        # Images
        if extract_images_flag and api_key:
            for img_elem in extract_page_images(doc, page, page_num, api_key):
                img_elem.section_title = current_section
                elements.append(img_elem)
                image_count += 1

    doc.close()
    metadata["images_extracted"] = image_count
    return elements, metadata


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
    sections, current_title, current_elems = [], "", []
    for elem in elements:
        if elem.element_type == "heading":
            if current_elems:
                sections.append((current_title, current_elems))
            current_title = elem.text.strip()
            current_elems = [elem]
        else:
            current_elems.append(elem)
    if current_elems:
        sections.append((current_title, current_elems))

    for section_title, section_elems in sections:
        buf_texts, buf_pages, buf_tokens, buf_img = [], [], 0, False
        for elem in section_elems:
            et = count_tokens(elem.text)
            is_img = elem.element_type == "image"
            if et > chunk_size:
                if buf_texts:
                    chunks.append(_make_chunk(buf_texts, buf_pages, section_title, chunk_idx, buf_img))
                    chunk_idx += 1
                    buf_texts, buf_pages, buf_tokens, buf_img = [], [], 0, False
                tokens = tokenizer.encode(elem.text)
                for i in range(0, len(tokens), chunk_size):
                    chunks.append(_make_chunk(
                        [tokenizer.decode(tokens[i:i+chunk_size])],
                        [elem.page_num], section_title, chunk_idx, is_img))
                    chunk_idx += 1
                continue
            if buf_tokens + et > chunk_size and buf_texts:
                chunks.append(_make_chunk(buf_texts, buf_pages, section_title, chunk_idx, buf_img))
                chunk_idx += 1
                keep_t, keep_p, keep_tok = [], [], 0
                for t, p in zip(reversed(buf_texts), reversed(buf_pages)):
                    tt = count_tokens(t)
                    if keep_tok + tt > overlap:
                        break
                    keep_t.insert(0, t); keep_p.insert(0, p); keep_tok += tt
                buf_texts, buf_pages, buf_tokens, buf_img = keep_t, keep_p, keep_tok, False
            buf_texts.append(elem.text); buf_pages.append(elem.page_num)
            buf_tokens += et
            if is_img: buf_img = True
        if buf_texts:
            chunks.append(_make_chunk(buf_texts, buf_pages, section_title, chunk_idx, buf_img))
            chunk_idx += 1
    return [c for c in chunks if c.token_count >= 30]


# ══════════════════════════════════════════════════════════════
# Hybrid Index & Answer Generation
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
        self.bm25 = BM25Okapi([self._tokenize(t) for t in texts])

    @staticmethod
    def _tokenize(text):
        return [t for t in re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split() if len(t) > 1]

    def search(self, query, top_k=15, rrf_k=60):
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        _, ids = self.faiss_index.search(q_emb, top_k)
        dense = [(int(i), r) for r, i in enumerate(ids[0]) if i >= 0]
        scores = self.bm25.get_scores(self._tokenize(query))
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
        f"[Source {i+1}] {c.citation()}\n{c.text}" for i, (c, _) in enumerate(passages)
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE PASSAGES:\n{ctx}\n\nQUESTION: {query}\n\nProvide a precise, well-cited answer using only the sources above."},
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
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        t0 = time.time()
        elements, metadata = ingest_pdf(filepath, api_key=api_key, extract_images_flag=True)
        chunks = chunk_elements(elements, chunk_size=512, overlap=64)
        index = HybridIndex(get_embed_model())
        index.build(chunks)
        state.update({"index": index, "metadata": metadata, "num_chunks": len(chunks), "ready": True})
        types = {}
        for e in elements: types[e.element_type] = types.get(e.element_type, 0) + 1
        return jsonify({
            "success": True, "filename": file.filename,
            "pages": metadata["total_pages"], "elements": len(elements),
            "chunks": len(chunks), "images": metadata.get("images_extracted", 0),
            "element_types": types, "time": round(time.time()-t0, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    if not state["ready"]:
        return jsonify({"error": "No document loaded. Upload a PDF first."}), 400
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
                        "images": state["metadata"].get("images_extracted", 0)})
    return jsonify({"ready": False})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Document Q&A — Multimodal (text + images)")
    print("="*50)
    print(f"  http://localhost:5000")
    print(f"  API key: {'✅' if os.environ.get('OPENAI_API_KEY') else '❌'}")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)