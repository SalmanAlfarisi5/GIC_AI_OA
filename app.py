"""
Document Q&A Web Application (local, full-featured).

Multi-format, multimodal RAG with cross-encoder re-ranking.
Supports PDF, DOCX, TXT, Markdown. Images from PDF + DOCX.

All pipeline logic lives in rag_core.py (shared with the eval harness and the
Hugging Face Space). This module is just the Flask layer.

Multimodal upgrades (both on by default, see rag_core.DEFAULT_CONFIG):
  #1 images are fed to the VLM at answer time, not just their captions
  #2 CLIP visual retrieval is fused into the hybrid search
"""

import os, time, tempfile

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from rag_core import (
    make_config, ingest_document, chunk_elements, HybridIndex,
    get_embed_model, get_clip_model, get_reranker, ask, CaptionCache,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_file_type(fn):
    ext = os.path.splitext(fn)[1].lower()
    return ext if ext in SUPPORTED_EXTENSIONS else None


# ── Pipeline config ───────────────────────────────────────────────────────────
CONFIG = make_config(
    top_k_retrieval=20,
    top_k_rerank=6,
    use_reranker=True,
    use_image_in_answer=True,    # #1
    use_clip_retrieval=True,     # #2
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

state = {"index": None, "metadata": None, "num_chunks": 0, "ready": False}
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "docqa_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CAPTION_CACHE = CaptionCache(os.path.join(UPLOAD_DIR, "caption_cache.json"))


def warm_models():
    get_embed_model(CONFIG["embedding_model"])
    if CONFIG["use_clip_retrieval"]:
        get_clip_model(CONFIG["clip_model"])
    if CONFIG["use_reranker"]:
        get_reranker(CONFIG["reranker_model"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    ft = get_file_type(file.filename)
    if not ft:
        return jsonify({"error": f"Unsupported. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"}), 400

    # secure_filename prevents path traversal via crafted filenames.
    safe_name = secure_filename(file.filename) or "upload" + ft
    filepath = os.path.join(UPLOAD_DIR, safe_name)
    file.save(filepath)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        t0 = time.time()
        elements, metadata = ingest_document(
            filepath, api_key=api_key, extract_images=CONFIG["extract_images"],
            caption_model=CONFIG["caption_model"], cache=CAPTION_CACHE)
        chunks = chunk_elements(elements, CONFIG["chunk_size"], CONFIG["chunk_overlap"])

        clip = get_clip_model(CONFIG["clip_model"]) if CONFIG["use_clip_retrieval"] else None
        index = HybridIndex(get_embed_model(CONFIG["embedding_model"]), clip_model=clip)
        index.build(chunks)
        if CONFIG["use_reranker"]:
            get_reranker(CONFIG["reranker_model"])

        state.update({"index": index, "metadata": metadata,
                      "num_chunks": len(chunks), "ready": True})
        types = {}
        for e in elements:
            types[e.element_type] = types.get(e.element_type, 0) + 1
        return jsonify({
            "success": True, "filename": safe_name, "file_type": ft,
            "pages": metadata["total_pages"], "elements": len(elements),
            "chunks": len(chunks), "images": metadata.get("images_extracted", 0),
            "element_types": types, "time": round(time.time() - t0, 1),
            "reranker": "on" if CONFIG["use_reranker"] else "off",
            "clip_retrieval": "on" if CONFIG["use_clip_retrieval"] else "off",
            "image_in_answer": "on" if CONFIG["use_image_in_answer"] else "off",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_endpoint():
    if not state["ready"]:
        return jsonify({"error": "No document loaded."}), 400
    query = (request.get_json() or {}).get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500
    try:
        t0 = time.time()
        reranker = get_reranker(CONFIG["reranker_model"]) if CONFIG["use_reranker"] else None
        result = ask(query, state["index"], CONFIG, reranker=reranker, api_key=api_key)
        return jsonify({
            "answer": result["answer"],
            "question": query,
            "latency": round(time.time() - t0, 2),
            "reranked": reranker is not None,
            "sources": [{k: s[k] for k in ("source_num", "citation", "text", "score", "has_image")}
                        for s in result["sources"]],
            "usage": result["usage"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    if state["ready"]:
        return jsonify({"ready": True, "filename": state["metadata"]["file_name"],
                        "pages": state["metadata"]["total_pages"],
                        "chunks": state["num_chunks"],
                        "images": state["metadata"].get("images_extracted", 0),
                        "reranker": "on" if CONFIG["use_reranker"] else "off"})
    return jsonify({"ready": False})


if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  Document Q&A — Multimodal RAG + Re-Ranker")
    print("  Formats: PDF, DOCX, TXT, Markdown   Images: PDF + DOCX")
    print(f"  Re-ranker: {'ON' if CONFIG['use_reranker'] else 'OFF'} |"
          f" CLIP retrieval: {'ON' if CONFIG['use_clip_retrieval'] else 'OFF'} |"
          f" Image-in-answer: {'ON' if CONFIG['use_image_in_answer'] else 'OFF'}")
    print(f"  http://localhost:5000   API key: {'set' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
    print("=" * 56 + "\n")
    warm_models()
    # debug defaults OFF (the Werkzeug debugger is an RCE risk); opt in explicitly.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug, port=int(os.environ.get("PORT", 5000)))
