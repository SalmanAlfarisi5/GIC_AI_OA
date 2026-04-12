# Document Q&A System

Multi-format, multimodal RAG with **two-stage retrieval** (hybrid search + cross-encoder re-ranking).
Supports PDF, DOCX, TXT, Markdown. Images from PDF + DOCX. Evaluated against [FinanceBench](https://github.com/patronus-ai/financebench).

## Architecture

```
Query
  │
  ▼
┌──────────────────────────────────────┐
│  Stage 1: RECALL                     │
│  FAISS (semantic) + BM25 (keyword)   │
│  → RRF Fusion → top 20 candidates    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Stage 2: PRECISION                  │
│  Cross-Encoder Re-Ranker             │
│  ms-marco-MiniLM-L-6-v2              │
│  → top 6 passages                    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Stage 3: GENERATION                 │
│  GPT-4o-mini + grounding prompt      │
│  → Answer with [Source N] citations  │
└──────────────────────────────────────┘
```

### Why Two Stages?

First-stage retrieval (FAISS + BM25) is fast but imprecise — it encodes query and passage separately. The cross-encoder re-ranker jointly encodes each `(query, passage)` pair, enabling token-level attention between them. This is far more accurate but too slow for the full corpus (5000+ chunks × 10ms = 50s). Two stages give us bi-encoder speed + cross-encoder accuracy.

## Supported Formats

| Format | Text | Headings | Tables | Images |
|--------|------|----------|--------|--------|
| **PDF** | ✅ PyMuPDF | ✅ Font-size heuristic | ✅ Built-in | ✅ Extracted + captioned |
| **DOCX** | ✅ python-docx | ✅ Word styles | ✅ Native | ✅ OPC rels + captioned |
| **Markdown** | ✅ Built-in | ✅ `#` syntax | — | — |
| **TXT** | ✅ Built-in | ✅ ALL CAPS / numbered | — | — |

## Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| PDF Parser | PyMuPDF | Fast, text + tables + images |
| DOCX Parser | python-docx | Native styles + tables + image access |
| Embeddings | bge-base-en-v1.5 | Top MTEB retrieval quality |
| Vector Store | FAISS IndexFlatIP | Exact search for single-doc scale |
| Keyword Index | BM25 Okapi | Exact terms, numbers, acronyms |
| Fusion | RRF (k=60) | Score-agnostic ranking merge |
| **Re-Ranker** | **ms-marco-MiniLM-L-6-v2** | **Cross-encoder precision; ~200ms for 20 candidates** |
| LLM | gpt-4o-mini | Cost-effective, strong grounding |
| Image Captioning | GPT-4o-mini vision | Converts charts/graphs to text |

## Evaluation

Automated across 8–10 FinanceBench documents: benchmark questions (faithfulness, relevance, citation quality, correctness vs gold) + adversarial nonsense questions (abstention rate). Results → `results.md`.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

For evaluation: `cd ~ && git clone https://github.com/patronus-ai/financebench.git`

## Usage

### Web Interface
```bash
python app.py    # http://localhost:5000
```

### Notebook
```bash
jupyter notebook document_qa_system.ipynb
```

## Project Structure

```
├── app.py                     # Flask web app
├── templates/index.html       # Dark-themed UI
├── document_qa_system.ipynb   # Full pipeline + benchmark
├── requirements.txt
├── README.md
└── results.md                 # Generated after benchmark
```

## Limitations

- Scanned PDFs need OCR (not implemented)
- TXT/MD have no embedded images
- Image captioning: ~$0.01/image via vision API
- Re-ranker adds ~200ms latency per query
- Single document at a time

## Future Improvements

1. Index caching to disk (avoid re-embedding)
2. OCR for scanned PDFs
3. Iterative retrieval for multi-hop questions
4. Query decomposition for complex questions
5. Confidence scoring to flag uncertain answers
6. Try ColPali framework (State of the art methodology)
