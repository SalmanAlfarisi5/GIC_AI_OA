# Multimodal Document RAG

Multi-format document Q&A with **three-way hybrid retrieval**, **cross-encoder
re-ranking**, and **true multimodal grounding** — the actual chart/table images
are fed to the model at answer time, not just their text captions. Supports PDF,
DOCX, TXT, and Markdown. Evaluated against [FinanceBench](https://github.com/patronus-ai/financebench).

🔗 **Live demo:** [alfarisisalman.com/multimodal-rag](https://alfarisisalman.com/multimodal-rag)

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — RECALL  (cast a wide net)                    │
│  • BGE dense embeddings        (semantic)               │
│  • numeric-aware BM25          (exact terms / figures)  │
│  • CLIP visual search          (query → image pixels)   │ ◀── upgrade #2
│  → Reciprocal Rank Fusion → top 20 candidates           │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — PRECISION                                    │
│  Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)       │
│  → top 6 passages                                       │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — MULTIMODAL GENERATION                        │
│  gpt-4o-mini + grounding prompt                         │
│  + the real image pixels of image-bearing sources       │ ◀── upgrade #1
│  → cited answer; out-of-document questions are refused  │
└─────────────────────────────────────────────────────────┘
```

### The two multimodal upgrades

A plain RAG captions an image to text once, embeds the caption, and then throws
the pixels away — so if the caption missed the number you asked about, it's gone.
This system fixes that on both the retrieval and the generation side:

1. **Images at answer time** (`use_image_in_answer`) — when a retrieved chunk
   carries an image, the actual pixels are attached to the gpt-4o-mini prompt
   (it is already a vision model), so it can re-read the chart for the specific
   question. Cost-capped to ≤ N images per answer at low detail.
2. **CLIP visual retrieval** (`use_clip_retrieval`) — image pixels are embedded
   with CLIP and a visual leg is fused into RRF, so a figure is findable by its
   content even when its caption didn't mention the asked-for detail. Runs on CPU.

A third, smaller fix: BM25 now uses a **numeric-aware tokenizer** that keeps
`$1,234.5`, `12.5%`, and `($4.2)` as whole tokens instead of shredding them —
directly relevant to financial questions where the answer *is* a figure.

## Supported formats

| Format | Text | Headings | Tables | Images |
|--------|------|----------|--------|--------|
| **PDF** | PyMuPDF | Font-size heuristic | Built-in | Extracted + captioned + CLIP-embedded |
| **DOCX** | python-docx | Word styles | Native | OPC rels + captioned + CLIP-embedded |
| **Markdown** | Built-in | `#` syntax | — | — |
| **TXT** | Built-in | ALL CAPS / numbered | — | — |

## Repository layout

```
├── rag_core.py              # ★ canonical pipeline (ingest → chunk → retrieve → rerank → generate)
├── app.py                   # Flask web app (local, full-featured) — imports rag_core
├── eval_compare.py          # FinanceBench before/after A/B benchmark (LLM-as-judge)
├── build_samples.py         # pre-index the demo's curated sample documents
├── templates/index.html     # dark-themed web UI
├── document_qa_system.ipynb # narrative walkthrough + benchmark
├── results.md               # pre-upgrade baseline scores
├── space/                   # ← deployable Hugging Face Space (public demo)
│   ├── app.py               #   Gradio app with cost guardrails
│   ├── rag_core.py          #   vendored copy of the root module
│   ├── sample_docs/         #   curated sample source documents
│   ├── DEPLOY.md            #   Space + portfolio + budget-cap setup
│   └── README.md            #   HF Space card
└── requirements.txt
```

`rag_core.py` is the single source of truth; the Flask app, the eval harness,
and the Space all import it. Both upgrades are config flags in
`rag_core.DEFAULT_CONFIG`, so they can be toggled for A/B testing.

## Design decisions

| Component | Choice | Why |
|-----------|--------|-----|
| PDF parser | PyMuPDF | Fast text + tables + images |
| DOCX parser | python-docx | Native styles + tables + image access |
| Dense embeddings | bge-base-en-v1.5 | Top MTEB retrieval quality |
| Visual embeddings | CLIP ViT-B/32 | Image+text in one space; runs on CPU |
| Vector store | FAISS IndexFlatIP | Exact search at single-doc scale |
| Keyword index | BM25 Okapi (numeric-aware) | Exact terms, numbers, acronyms |
| Fusion | RRF (k=60) | Score-agnostic merge of 3 legs |
| Re-ranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision; ~200ms / 20 candidates |
| LLM / VLM | gpt-4o-mini | Cost-effective, vision-capable, strong grounding |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

For the benchmark: `git clone https://github.com/patronus-ai/financebench.git`

## Usage

### Web interface (local)
```bash
python app.py        # http://localhost:5000   (set FLASK_DEBUG=1 only for dev)
```

### Before/after benchmark
```bash
python eval_compare.py --max-docs 1 --max-qs 3 --nonsense 2   # quick smoke test
python eval_compare.py --max-docs 4 --max-qs 0                # full run → comparison_results.md
```
Runs the same FinanceBench questions through **baseline** (caption-only) and
**upgraded** (#1 + #2) configs and reports the per-metric delta. Captions are
cached on disk so the two configs share the expensive ingestion work.

### Public demo (Hugging Face Space)
See [`space/DEPLOY.md`](space/DEPLOY.md). The Space serves pre-indexed sample
documents (zero ingestion cost), strictly caps uploads, rate-limits per session
and globally, and uses gpt-4o-mini — with a hard OpenAI budget cap as the final
backstop. The portfolio website calls it via `@gradio/client`.

## Notes & limitations

- Scanned PDFs need OCR (not implemented).
- TXT/MD have no embedded images; DOCX/TXT page numbers are heuristic estimates.
- Image captioning costs API budget; captions are cached by content hash to avoid
  re-paying, and uploads on the public demo cap how many images are captioned.
- CLIP ViT-B/32 is a pragmatic CPU-friendly choice; ColPali/ColQwen would score
  higher on figure-heavy docs but need a GPU.
