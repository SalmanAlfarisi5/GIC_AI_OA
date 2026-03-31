# Document-Grounded Q&A System for Long Documents

A RAG system that answers questions from a single long document (100+ pages)
with citation support and hallucination resistance, evaluated against the
[FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

## Problem Framing

This is a **document-grounded QA problem** where the core challenges are:

1. **Retrieval quality** — finding the right passages in a dense 100+ page document
2. **Citation fidelity** — every claim must trace back to specific pages and sections
3. **Hallucination control** — refuse to answer rather than fabricate

With only one document, the difficulty isn't scale — it's precision: locating the
most relevant content and generating answers strictly grounded in that content.

## Architecture

```
PDF → PyMuPDF Parser → Section-Aware Chunker → [FAISS + BM25] → RRF Fusion → LLM + Citations
```

The pipeline has six stages:

1. **Ingestion** — PyMuPDF extracts text block-by-block with page numbers, detects headings via font-size heuristics, and extracts tables
2. **Chunking** — Elements are grouped by section heading, then split into 512-token chunks with 64-token overlap to preserve context at boundaries
3. **Dense Indexing (FAISS)** — Chunks are embedded with `bge-base-en-v1.5` and indexed for semantic similarity search
4. **Sparse Indexing (BM25)** — Same chunks are tokenized for keyword matching — critical for exact terms, numbers, and acronyms
5. **Hybrid Retrieval (RRF)** — Rankings from both retrievers are fused using Reciprocal Rank Fusion, which doesn't require score normalization
6. **Generation** — Top passages are sent to `gpt-4o-mini` with a strict grounding prompt that enforces citations and allows refusal

## Key Design Decisions

| Decision | Choice | Why | Alternatives |
|----------|--------|-----|-------------|
| PDF Parser | PyMuPDF | Fast, structure-aware, table support | pdfplumber (slower), unstructured (heavier) |
| Chunking | Heading-grouped + token overlap | Keeps topically related content together | Recursive char splitting (heading-unaware) |
| Dense Embeddings | bge-base-en-v1.5 | Top retrieval quality in its size class | all-MiniLM-L6 (faster/weaker) |
| Vector Store | FAISS flat | Single doc = small corpus; exact search is fine | Chroma (overkill here) |
| Sparse Retrieval | BM25 | Exact term matching for numbers, acronyms | TF-IDF (weaker) |
| Fusion | Reciprocal Rank Fusion | Score-agnostic, no normalization needed | Linear combination (requires normalization) |
| LLM | gpt-4o-mini | Cost-effective, strong instruction following | gpt-4o (better/costlier) |
| Benchmark | FinanceBench | Gold answers, multiple reasoning types | Manual QA (not reproducible) |

### Why Hybrid Retrieval?

Dense and sparse search have complementary strengths. Dense (FAISS) handles paraphrases
("company revenue" ↔ "total sales") while sparse (BM25) handles exact matches ("GAAP", specific
dollar amounts). Financial and technical documents need both. RRF combines their rankings cleanly.

## Evaluation Strategy

The system is evaluated automatically across **8-10 FinanceBench documents** on two types of questions:

### Benchmark Questions (from FinanceBench)
Each document has human-annotated questions with gold answers. Scored on:
- **Faithfulness** (1-5): Grounded in retrieved sources?
- **Relevance** (1-5): Answers the question?
- **Citation Quality** (1-5): Proper `[Source N]` references?
- **Correctness** (1-5): Matches gold answer?

### Adversarial Nonsense Questions
Questions with no answer in any financial document (e.g., "What is the recipe for chocolate cake?").
A robust system should refuse to answer. Measured by **Abstention Rate** (higher = better).

All results are written to `results.md` with per-document breakdowns and per-question details.

---

## Setup

### Requirements
- Python 3.9+
- OpenAI API key

### Install
```bash
pip install -r requirements.txt
```

### Get Benchmark Data (for evaluation)
```bash
# Clone OUTSIDE this repo, not inside it
cd ~
git clone https://github.com/patronus-ai/financebench.git
```

### Configure
```bash
export OPENAI_API_KEY="sk-..."
```

---

## How to Run

### Option 1: Web Interface (Demo)

A Flask web app for interactive single-document Q&A.

```bash
export OPENAI_API_KEY="sk-..."
python app.py
```

Then open **http://localhost:5000** in your browser. You can:
1. Upload a PDF (drag & drop or click)
2. Wait for processing (parsing, chunking, embedding)
3. Ask questions and get grounded answers with citations

### Option 2: Jupyter Notebook (Full Pipeline + Benchmark)

```bash
jupyter notebook document_qa_system.ipynb
```

**For the full FinanceBench benchmark:**
1. Update paths in Section 11 to your FinanceBench clone:
   ```python
   CONFIG["pdf_dir"] = os.path.expanduser("~/financebench/pdfs")
   CONFIG["benchmark_file"] = os.path.expanduser("~/financebench/data/financebench_open_source.jsonl")
   ```
2. Run all cells — the system will process 8-10 documents and generate `results.md`

**For single-document testing:** Use Section 12 of the notebook.

---

## Project Structure

```
.
├── app.py                         # Flask web app (single document Q&A)
├── templates/
│   └── index.html                 # Web interface (dark theme)
├── document_qa_system.ipynb       # Full pipeline + FinanceBench benchmark
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── code_walkthrough.md            # Line-by-line code explanation (interview prep)
└── results.md                     # Generated after running benchmark
```

The FinanceBench data (PDFs + questions) is **not included** in this repo.
Clone it separately as described in Setup above.

---

## Assumptions & Limitations

1. **PDF only** — The system currently only accepts PDF files. Other formats (DOCX, HTML, TXT) are not supported. PDFs must contain extractable text (not scanned images without OCR).
2. **Tables** — Simple tables extract well; complex merged cells may fail
3. **Images/Charts** — Text-only extraction; figures and charts are ignored
4. **Heading detection** — Font-size heuristic works for most documents, not all
5. **Single document** — One document at a time by design
6. **Evaluation** — LLM-as-judge has known biases; not a substitute for human review
7. **API dependency** — Requires OpenAI API key for answer generation (retrieval works offline)

## Domain Generalization

Nothing in this system is finance-specific:
- Chunking is structure-based, not content-based
- Embeddings are general-purpose
- The generation prompt is domain-agnostic

To specialize for a domain: fine-tune embeddings on domain data, add domain-specific
preprocessing, or customize the prompt.

## Future Improvements

With more time, in order of impact:

1. **Cross-encoder re-ranker** — Add a second-stage re-ranker (e.g., `ms-marco-MiniLM`) to improve precision
2. **Multi-format support** — Accept DOCX, HTML, TXT, and scanned PDFs (via OCR)
3. **Index caching** — Serialize FAISS index to disk so re-embedding isn't needed on restart
4. **Iterative retrieval** — If first retrieval is weak, reformulate the query and search again
5. **Semantic chunking** — Use embedding similarity to find natural topic boundaries
6. **Query decomposition** — Break complex multi-part questions into sub-questions
7. **Multimodal** — Process charts and images with a vision model