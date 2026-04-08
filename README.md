# Document-Grounded Q&A System (Multimodal)

A RAG system that answers questions from a single long document (100+ pages)
using **text, tables, and images** — with citation support and hallucination resistance.
Evaluated against [FinanceBench](https://github.com/patronus-ai/financebench).

## Problem Framing

This is a **document-grounded QA problem** where the core challenges are:

1. **Retrieval quality** — finding the right passages in a dense 100+ page document
2. **Citation fidelity** — every claim must trace back to specific pages and sections
3. **Hallucination control** — refuse to answer rather than fabricate
4. **Multimodal content** — charts, graphs, and diagrams contain information that text extraction alone misses

## Architecture

```
PDF → PyMuPDF Parser ─┬─ Text blocks ──→ Section-Aware Chunker → [FAISS + BM25] → RRF → LLM + Citations
                      ├─ Tables ────────→
                      └─ Images ──→ GPT-4o-mini Vision (caption) →
```

### Multimodal Pipeline

Images embedded in PDFs are extracted, filtered (skip icons/logos <100px or <5KB),
and captioned using GPT-4o-mini's vision capability. The captions describe chart data,
graph trends, diagram structures, and visible text/numbers. These captions become
searchable text chunks indexed alongside regular text — so questions about visual
content (e.g., "What does the revenue chart show?") can be answered.

### Pipeline Stages

1. **Ingestion** — PyMuPDF extracts text, tables, and embedded images per page
2. **Image Captioning** — GPT-4o-mini vision describes each image in detail
3. **Chunking** — Elements grouped by section heading, split into 512-token chunks with overlap
4. **Dense Index (FAISS)** — Semantic similarity search via `bge-base-en-v1.5` embeddings
5. **Sparse Index (BM25)** — Keyword matching for exact terms, numbers, acronyms
6. **Hybrid Retrieval (RRF)** — Reciprocal Rank Fusion combines both rankings
7. **Generation** — Strict grounding prompt with citations; image sources tagged `[image]`

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| PDF Parser | PyMuPDF | Fast, extracts text + tables + images |
| Image Captioning | GPT-4o-mini vision | Supports image input, cost-effective |
| Image Filter | ≥100px and ≥5KB | Removes icons/logos without losing real charts |
| Chunking | Heading-grouped + token overlap | Topical coherence |
| Embeddings | bge-base-en-v1.5 | Top retrieval quality in its size class |
| Vector Store | FAISS flat | Exact search, fine for single-doc scale |
| Sparse Retrieval | BM25 | Exact term matching |
| Fusion | RRF (k=60) | Score-agnostic, no normalization needed |
| LLM | gpt-4o-mini | Cost-effective, strong instruction following |

## Evaluation

Automated benchmark across 8-10 FinanceBench documents:

**Benchmark questions** — scored on faithfulness, relevance, citation quality, and correctness vs gold answers.

**Adversarial nonsense questions** — e.g., "What is the recipe for chocolate cake?" Measures abstention rate (should refuse to answer).

Results are written to `results.md` with per-document breakdowns.

## Setup

### Requirements
- Python 3.9+
- OpenAI API key (for generation + image captioning)

### Install
```bash
pip install -r requirements.txt
```

### Benchmark Data (for evaluation)
```bash
cd ~
git clone https://github.com/patronus-ai/financebench.git
```

### Configure
```bash
export OPENAI_API_KEY="sk-..."
```

## How to Run

### Web Interface (Demo)
```bash
python app.py
# Open http://localhost:5000
```
Upload a PDF → images are automatically extracted and captioned → ask questions.

### Notebook (Full Pipeline + Benchmark)
```bash
jupyter notebook document_qa_system.ipynb
```
Update paths in Section 12 to your FinanceBench clone, run all cells.

## Project Structure

```
.
├── app.py                         # Flask web app (multimodal)
├── templates/
│   └── index.html                 # Dark-themed web interface
├── document_qa_system.ipynb       # Full pipeline + benchmark
├── requirements.txt
├── README.md
├── code_walkthrough.md            # Line-by-line code explanation
└── results.md                     # Generated after benchmark
```

## Assumptions & Limitations

1. **PDF only** — only accepts PDF files; no DOCX, HTML, TXT, or scanned images without OCR
2. **Image captioning cost** — each extracted image uses one GPT-4o-mini vision API call (~$0.01 per image)
3. **Image quality** — captions depend on the vision model; complex or low-resolution charts may lose detail
4. **Tables** — simple table layouts extract well; complex merged cells may fail
5. **Heading detection** — font-size heuristic works for most professional documents, not all
6. **Single document** — one document at a time by design
7. **API dependency** — requires OpenAI API key for generation and image captioning

## Domain Generalization

Nothing is domain-specific. Chunking is structure-based, embeddings are general-purpose,
and image captioning is instructed to describe any visual content. To specialize:
fine-tune embeddings, add domain-specific prompt instructions, or customize image captioning prompts.

## Future Improvements

1. **Cross-encoder re-ranker** for better retrieval precision
2. **OCR** for scanned PDFs without extractable text
3. **Multi-format support** (DOCX, HTML, TXT)
4. **Index caching** to avoid re-embedding on restart
5. **Iterative retrieval** for multi-hop questions
6. **Image captioning cache** to avoid re-captioning identical images
7. **Semantic chunking** using embedding similarity boundaries