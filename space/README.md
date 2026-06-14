---
title: Multimodal Document RAG
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Multimodal document Q&A with hybrid + CLIP retrieval
---

# Multimodal Document RAG — Demo

Ask grounded, cited questions over a document. This Space backs the live demo on
[alfarisisalman.com](https://alfarisisalman.com).

**Pipeline:** ingest (PDF/DOCX/TXT/MD, images captioned by a VLM) → chunk →
**hybrid retrieval** (BGE dense + numeric-aware BM25 + **CLIP visual** search,
fused with Reciprocal Rank Fusion) → **cross-encoder re-ranker** → answer with
**gpt-4o-mini**, with the actual chart/table **images attached to the prompt**
(not just their captions).

## Cost & abuse guardrails

The OpenAI key is a Space secret, so all spend protection is enforced **server-side**:

- **Curated samples are pre-indexed** — answering one makes a single cheap model
  call and **zero** ingestion/captioning calls.
- **Uploads are strictly capped**: ≤ 5 MB, ≤ 8 pages, ≤ 6 images captioned, and
  limited per session and globally per hour.
- **Rate limits**: per-session and global rolling-window caps on questions and uploads.
- **gpt-4o-mini only**, low image detail, small `max_tokens`, ≤ 2 images per answer.
- `DEMO_DISABLED=1` is a kill switch; a hard **monthly budget cap** in the OpenAI
  dashboard is the final backstop. See `DEPLOY.md`.

All limits are environment variables (e.g. `SESSION_MAX_Q`, `GLOBAL_MAX_Q_PER_HOUR`,
`UPLOAD_MAX_PAGES`) — tune them in the Space settings without code changes.

## Endpoints (used by the portfolio site)

- `/ask` → `(sample_id, question, token)` → JSON `{answer, sources}` (samples only)
- `/samples` → JSON manifest of available sample documents

Code: https://github.com/SalmanAlfarisi5  ·  Built by Salman Alfarisi.
