# Document Q&A System

A multimodal RAG system that answers questions from long documents with citations and hallucination resistance. Supports **PDF, DOCX, TXT, and Markdown**. Evaluated against [FinanceBench](https://github.com/patronus-ai/financebench).

## Architecture

```
Document ─→ Format Router ─┬─ PDF Parser (PyMuPDF) ──→ text + tables + images (captioned)
                           ├─ DOCX Parser (python-docx) → text + tables
                           ├─ Markdown Parser ──────────→ text (heading-aware)
                           └─ TXT Parser ───────────────→ text (heuristic headings)
                                      │
                                      ▼
                           Section-Aware Chunker (512 tok, 64 overlap)
                                      │
                                      ▼
                            ┌─────────┴─────────┐
                            │ FAISS (semantic)  │ BM25 (keyword)
                            └─────────┬─────────┘
                                      │ RRF Fusion
                                      ▼
                              LLM + Grounding Prompt → Answer with [Source N] citations
```

## Supported Formats

| Format | Parser | Headings | Tables | Images |
|--------|--------|----------|--------|--------|
| **PDF** | PyMuPDF | Font-size heuristic | Built-in table extraction | Extracted & captioned via GPT-4o-mini vision |
| **DOCX** | python-docx | Word heading styles | Native table parsing | Not yet supported |
| **Markdown** | Built-in | `#` syntax + underline style | — | — |
| **TXT** | Built-in | ALL CAPS, numbered headings, underlines | — | — |

## Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| PDF Parser | PyMuPDF | Fast, structure-aware, table + image extraction |
| DOCX Parser | python-docx | Native heading style detection, table support |
| TXT/MD Parser | Custom regex | Lightweight, no dependencies, handles common heading patterns |
| Embeddings | bge-base-en-v1.5 | Top retrieval quality in its size class (MTEB) |
| Vector Store | FAISS IndexFlatIP | Exact search, fine for single-doc scale |
| Keyword Index | BM25 Okapi | Essential for exact terms, numbers, acronyms |
| Fusion | Reciprocal Rank Fusion | Score-agnostic, no normalization needed |
| LLM | gpt-4o-mini | Cost-effective, strong instruction following |
| Image Captioning | GPT-4o-mini vision | Converts charts/graphs to searchable text |

## Evaluation

Automated benchmark across 8–10 FinanceBench documents:

- **Benchmark questions** — scored on faithfulness, relevance, citation quality, and correctness vs gold answers
- **Adversarial nonsense questions** — measures abstention rate (system should refuse to answer irrelevant questions)
- Results written to `results.md`

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

For FinanceBench evaluation:
```bash
cd ~
git clone https://github.com/patronus-ai/financebench.git
```

## Usage

### Web Interface

```bash
python app.py
# Open http://localhost:5000
```

Upload any supported file → ask questions → get cited answers.

### Jupyter Notebook

```bash
jupyter notebook document_qa_system.ipynb
```

Update FinanceBench paths in Section 12 and run all cells for the full benchmark.

## Project Structure

```
├── app.py                     # Flask web app
├── templates/index.html       # Web interface
├── document_qa_system.ipynb   # Full pipeline + benchmark
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── code_walkthrough.md        # Line-by-line code explanation
├── interview_prep.md          # 37 technical interview Q&As
└── results.md                 # Generated after benchmark run
```

## Limitations

- **PDF images** — captioned via vision model; complex or low-res charts may lose detail
- **DOCX/TXT/MD images** — not extracted (only PDF images are supported)
- **Scanned PDFs** — requires an extractable text layer; pure image scans need OCR (not implemented)
- **Table extraction** — works well for simple layouts; complex merged cells may fail
- **Heading detection** — heuristic-based; may misclassify in unusual document layouts
- **Single document** — processes one document at a time

## Future Improvements

1. Cross-encoder re-ranker for better retrieval precision
2. OCR for scanned PDFs (Tesseract / pdf2image)
3. Image extraction from DOCX files
4. Index caching to disk (avoid re-embedding on restart)
5. Iterative retrieval for multi-hop questions
6. Query decomposition for complex questions
7. Confidence scoring to flag uncertain answers