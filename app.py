"""
Document Q&A Web Application
Flask app that provides a web interface for the document QA pipeline.
Upload a PDF, ask questions, get grounded answers with citations.
"""

import os
import re
import json
import hashlib
import time
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Any

from flask import Flask, render_template, request, jsonify, session
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
    element_type: str
    section_title: str = ""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_numbers: List[int]
    section_title: str
    token_count: int
    chunk_index: int

    def citation(self) -> str:
        pages = sorted(set(self.page_numbers))
        if len(pages) == 1:
            p = f"p. {pages[0]}"
        elif len(pages) <= 3:
            p = f"pp. {', '.join(map(str, pages))}"
        else:
            p = f"pp. {pages[0]}-{pages[-1]}"
        sec = f' — "{self.section_title}"' if self.section_title else ""
        return f"[{p}{sec}]"


# ══════════════════════════════════════════════════════════════
# PDF Ingestion
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


def detect_heading(block: dict, median_size: float) -> Tuple[bool, int]:
    lines = block.get("lines", [])
    if not lines:
        return False, 0
    font_sizes = []
    is_bold = False
    for line in lines:
        for span in line.get("spans", []):
            font_sizes.append(span["size"])
            if "bold" in span.get("font", "").lower():
                is_bold = True
    if not font_sizes:
        return False, 0
    block_size = max(font_sizes)
    ratio = block_size / median_size if median_size > 0 else 1.0
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


def extract_block_text(block: dict) -> str:
    text = ""
    for line in block.get("lines", []):
        line_text = "".join(span["text"] for span in line.get("spans", []))
        text += line_text + "\n"
    return text.strip()


def ingest_pdf(pdf_path: str) -> Tuple[List[PageElement], dict]:
    doc = fitz.open(pdf_path)
    elements = []
    current_section = ""
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
        try:
            tables = page.find_tables()
            for table in tables:
                df = table.to_pandas()
                if not df.empty:
                    elements.append(PageElement(
                        text=df.to_markdown(index=False),
                        page_num=page_num,
                        element_type="table",
                        section_title=current_section,
                    ))
        except Exception:
            pass
        for block in blocks:
            if block.get("type", 1) != 0:
                continue
            text = extract_block_text(block)
            if not text or len(text) < 3:
                continue
            is_heading, level = detect_heading(block, median_fs)
            if is_heading:
                current_section = text.strip()
                elements.append(PageElement(
                    text=text, page_num=page_num,
                    element_type="heading", section_title=current_section,
                ))
            else:
                elements.append(PageElement(
                    text=text, page_num=page_num,
                    element_type="paragraph", section_title=current_section,
                ))
    doc.close()
    return elements, metadata


# ══════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def _make_chunk(texts, pages, section_title, idx):
    combined = "\n\n".join(texts)
    cid = hashlib.md5(f"{idx}:{combined[:50]}".encode()).hexdigest()[:10]
    return Chunk(
        chunk_id=cid, text=combined, page_numbers=list(pages),
        section_title=section_title, token_count=count_tokens(combined),
        chunk_index=idx,
    )


def chunk_elements(elements, chunk_size=512, overlap=64):
    chunks = []
    chunk_idx = 0
    sections = []
    current_title = ""
    current_elems = []
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
        buf_texts, buf_pages, buf_tokens = [], [], 0
        for elem in section_elems:
            elem_tokens = count_tokens(elem.text)
            if elem_tokens > chunk_size:
                if buf_texts:
                    chunks.append(_make_chunk(buf_texts, buf_pages, section_title, chunk_idx))
                    chunk_idx += 1
                    buf_texts, buf_pages, buf_tokens = [], [], 0
                tokens = tokenizer.encode(elem.text)
                for i in range(0, len(tokens), chunk_size):
                    piece = tokenizer.decode(tokens[i:i + chunk_size])
                    chunks.append(_make_chunk([piece], [elem.page_num], section_title, chunk_idx))
                    chunk_idx += 1
                continue
            if buf_tokens + elem_tokens > chunk_size and buf_texts:
                chunks.append(_make_chunk(buf_texts, buf_pages, section_title, chunk_idx))
                chunk_idx += 1
                keep_texts, keep_pages, keep_tokens = [], [], 0
                for t, p in zip(reversed(buf_texts), reversed(buf_pages)):
                    t_tok = count_tokens(t)
                    if keep_tokens + t_tok > overlap:
                        break
                    keep_texts.insert(0, t)
                    keep_pages.insert(0, p)
                    keep_tokens += t_tok
                buf_texts, buf_pages, buf_tokens = keep_texts, keep_pages, keep_tokens
            buf_texts.append(elem.text)
            buf_pages.append(elem.page_num)
            buf_tokens += elem_tokens
        if buf_texts:
            chunks.append(_make_chunk(buf_texts, buf_pages, section_title, chunk_idx))
            chunk_idx += 1

    return [c for c in chunks if c.token_count >= 30]


# ══════════════════════════════════════════════════════════════
# Hybrid Index
# ══════════════════════════════════════════════════════════════

class HybridIndex:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.chunks = []
        self.faiss_index = None
        self.bm25 = None

    def build(self, chunks):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(
            texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text):
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        return [t for t in text.split() if len(t) > 1]

    def search(self, query, top_k=15, rrf_k=60):
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores_d, indices_d = self.faiss_index.search(q_emb, top_k)
        dense_ranking = [(int(idx), rank) for rank, idx in enumerate(indices_d[0]) if idx >= 0]
        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_sparse = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_ranking = [(int(idx), rank) for rank, idx in enumerate(top_sparse) if bm25_scores[idx] > 0]
        rrf = {}
        for idx, rank in dense_ranking:
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        for idx, rank in sparse_ranking:
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        sorted_results = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in sorted_results[:top_k]]


# ══════════════════════════════════════════════════════════════
# Answer Generation
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer questions based
ONLY on the provided source passages.

RULES:
1. Use ONLY information from the provided sources. No external knowledge.
2. Cite sources using [Source N] notation.
3. If multiple sources support a claim, cite all: [Source 1, Source 3].
4. If the sources don't contain enough information to answer the question, say:
   "The provided document sections do not contain sufficient information to answer this question."
5. Do not speculate beyond what the sources explicitly state.
6. For numbers, quote exact figures from the sources.
7. Be concise but complete."""


def generate_answer(query, passages, api_key, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)
    context_parts = []
    for i, (chunk, score) in enumerate(passages, 1):
        context_parts.append(f"[Source {i}] {chunk.citation()}\n{chunk.text}")
    context = "\n\n---\n\n".join(context_parts)

    user_msg = f"""Based on the following source passages, answer the question.

SOURCE PASSAGES:
{context}

QUESTION: {query}

Provide a precise, well-cited answer using only the sources above."""

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "source_num": i + 1,
                "citation": chunk.citation(),
                "text": chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
                "score": round(score, 4),
            }
            for i, (chunk, score) in enumerate(passages)
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


# ══════════════════════════════════════════════════════════════
# Flask Application
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

# Global state
state = {
    "embed_model": None,
    "index": None,
    "metadata": None,
    "num_chunks": 0,
    "ready": False,
}

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "docqa_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_embed_model():
    """Load embedding model once (lazy loading)."""
    if state["embed_model"] is None:
        print("Loading embedding model (first time, ~10s)...")
        state["embed_model"] = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("✅ Embedding model loaded.")
    return state["embed_model"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload and process a PDF document."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save file
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    try:
        t0 = time.time()

        # Ingest
        elements, metadata = ingest_pdf(filepath)

        # Chunk
        chunks = chunk_elements(elements, chunk_size=512, overlap=64)

        # Index
        embed_model = get_embed_model()
        index = HybridIndex(embed_model)
        index.build(chunks)

        elapsed = time.time() - t0

        # Store in global state
        state["index"] = index
        state["metadata"] = metadata
        state["num_chunks"] = len(chunks)
        state["ready"] = True

        return jsonify({
            "success": True,
            "filename": file.filename,
            "pages": metadata["total_pages"],
            "elements": len(elements),
            "chunks": len(chunks),
            "time": round(elapsed, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Ask a question about the uploaded document."""
    if not state["ready"]:
        return jsonify({"error": "No document loaded. Upload a PDF first."}), 400

    data = request.get_json()
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set on server"}), 500

    try:
        t0 = time.time()

        # Retrieve
        candidates = state["index"].search(query, top_k=15, rrf_k=60)
        passages = candidates[:6]

        # Generate
        result = generate_answer(query, passages, api_key)
        result["latency"] = round(time.time() - t0, 2)
        result["question"] = query

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    """Check if a document is loaded."""
    if state["ready"]:
        return jsonify({
            "ready": True,
            "filename": state["metadata"]["file_name"],
            "pages": state["metadata"]["total_pages"],
            "chunks": state["num_chunks"],
        })
    return jsonify({"ready": False})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Document Q&A System — Web Interface")
    print("=" * 50)
    print(f"  Open: http://localhost:5000")
    print(f"  API key: {'✅ Set' if os.environ.get('OPENAI_API_KEY') else '❌ Missing'}")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)