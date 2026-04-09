# Document Q&A System

A multimodal RAG system that answers questions from long documents with citations and hallucination resistance. Supports **PDF, DOCX, TXT, and Markdown**. Images extracted from **PDF and DOCX**. Evaluated against [FinanceBench](https://github.com/patronus-ai/financebench).

## Architecture

```
Document ─→ Format Router ─┬─ PDF Parser ──────→ text + tables + images (captioned)
                           ├─ DOCX Parser ─────→ text + tables + images (captioned)
                           ├─ Markdown Parser ─→ text (heading-aware)
                           └─ TXT Parser ──────→ text (heuristic headings)
                                    │
                                    ▼
                         Section-Aware Chunker (512 tok, 64 overlap)
                                    │
                              ┌─────┴─────┐
                              │FAISS│BM25 │
                              └─────┬─────┘
                                    │ RRF Fusion
                                    ▼
                            LLM + Grounding Prompt → Answer with [Source N] citations
```

## Supported Formats

| Format | Text | Headings | Tables | Images |
|--------|------|----------|--------|--------|
| **PDF** | ✅ PyMuPDF | ✅ Font-size heuristic | ✅ Built-in | ✅ `page.get_images()` → captioned |
| **DOCX** | ✅ python-docx | ✅ Word heading styles | ✅ Native | ✅ OPC relationships → captioned |
| **Markdown** | ✅ Built-in | ✅ `#` syntax + underlines | — | — |
| **TXT** | ✅ Built-in | ✅ ALL CAPS, numbered, underlines | — | — |

### Image Pipeline

Both PDF and DOCX images are processed through the same `caption_image()` function:

```
PDF:  page.get_images() → extract_image(xref) → filter ≥100px/≥5KB → caption via GPT-4o-mini vision
DOCX: document.part.rels → filter image rels → target_part.blob → filter ≥5KB → caption via GPT-4o-mini vision
```

Captions become regular text chunks, searchable alongside document text. Image-sourced citations are tagged `[image]`.

## Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| PDF Parser | PyMuPDF | Fast, text + tables + images |
| DOCX Parser | python-docx | Native heading styles + table + image access |
| TXT/MD Parser | Custom regex | Lightweight, no dependencies |
| Image Captioning | GPT-4o-mini vision | Shared by PDF and DOCX, format-agnostic |
| Embeddings | bge-base-en-v1.5 | Top retrieval quality (MTEB) |
| Vector Store | FAISS IndexFlatIP | Exact search for single-doc scale |
| Keyword Index | BM25 Okapi | Exact terms, numbers, acronyms |
| Fusion | RRF (k=60) | Score-agnostic |
| LLM | gpt-4o-mini | Cost-effective, strong instruction following |

## Evaluation

Automated benchmark across 8–10 FinanceBench documents: benchmark questions scored on faithfulness, relevance, citation quality, and correctness vs gold answers; adversarial nonsense questions measuring abstention rate. Results written to `results.md`.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

For evaluation: `cd ~ && git clone https://github.com/patronus-ai/financebench.git`

## Usage

### Web Interface
```bash
python app.py
# http://localhost:5000 — upload PDF/DOCX/TXT/MD, ask questions
```

### Notebook
```bash
jupyter notebook document_qa_system.ipynb
# Update FinanceBench paths in Section 12, run all cells
```

## Project Structure

```
├── app.py                     # Flask web app (multimodal, multi-format)
├── templates/index.html       # Dark-themed web interface
├── document_qa_system.ipynb   # Full pipeline + FinanceBench benchmark
├── requirements.txt
├── README.md
├── code_walkthrough.md        # Line-by-line code explanation
├── interview_prep.md          # 37 technical interview Q&As
└── results.md                 # Generated after benchmark run
```

## Limitations

- **Scanned PDFs** — requires extractable text layer; pure image scans need OCR (not implemented)
- **TXT/MD images** — plain text formats have no embedded images
- **Image captioning cost** — ~$0.01 per image via GPT-4o-mini vision
- **Image quality** — complex or low-res charts may lose detail in captions
- **Table extraction** — simple layouts work; complex merged cells may fail
- **Heading detection** — heuristic-based; may misclassify in unusual layouts
- **Single document** — one document at a time

## Future Improvements

1. Cross-encoder re-ranker for better retrieval precision
2. OCR for scanned PDFs (Tesseract + pdf2image)
3. Index caching to disk
4. Iterative retrieval for multi-hop questions
5. Query decomposition for complex questions
6. Confidence scoring to flag uncertain answers