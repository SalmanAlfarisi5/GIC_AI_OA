"""
build_samples.py — Pre-index the curated demo documents for the Space.

The public Space serves a fixed set of *pre-indexed* sample documents so that:
  • there is NO captioning/ingestion cost at request time, and
  • there is nothing for a visitor to abuse via the curated path.

For each file in space/sample_docs/, this ingests + chunks it ONCE (captioning
any images a single time, via the disk cache) and pickles the resulting chunks
to space/samples/<id>.pkl. The Space rebuilds the (free, local) embedding
indexes from those chunks at startup — no re-captioning, no API calls to serve.

Text/markdown samples need NO OpenAI key. Image-bearing PDFs/DOCX need a key
once, here, to caption their figures (the captions are then baked into the pkl).

Usage:
  python build_samples.py
  python build_samples.py --src space/sample_docs --out space/samples
"""

import os, glob, json, pickle, argparse

import rag_core
from rag_core import ingest_document, chunk_elements, CaptionCache, make_config


def title_from(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace("_", " ").replace("-", " ").title()


def build(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = make_config()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    cache = CaptionCache(os.path.join(out_dir, "caption_cache.json"))

    files = sorted(f for f in glob.glob(os.path.join(src_dir, "*"))
                   if os.path.splitext(f)[1].lower() in {".pdf", ".docx", ".txt", ".md"})
    if not files:
        raise SystemExit(f"No sample documents found in {src_dir}")

    manifest = []
    for fp in files:
        sid = os.path.splitext(os.path.basename(fp))[0]
        print(f"• {sid} …", end=" ", flush=True)
        elements, meta = ingest_document(
            fp, api_key=api_key, extract_images=cfg["extract_images"],
            caption_model=cfg["caption_model"], cache=cache)
        chunks = chunk_elements(elements, cfg["chunk_size"], cfg["chunk_overlap"])
        with open(os.path.join(out_dir, f"{sid}.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        n_img = sum(1 for c in chunks if c.has_image)
        manifest.append({
            "id": sid, "title": title_from(fp), "file_type": meta["file_type"],
            "pages": meta.get("total_pages", 0), "chunks": len(chunks),
            "images": meta.get("images_extracted", 0), "image_chunks": n_img,
        })
        print(f"{len(chunks)} chunks, {meta.get('images_extracted',0)} images")

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"\n✅ Wrote {len(manifest)} sample index(es) + manifest to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="space/sample_docs")
    ap.add_argument("--out", default="space/samples")
    a = ap.parse_args()
    build(a.src, a.out)
