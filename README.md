# Document-Grounded Q&A System for Long Documents

A RAG system that answers questions from a single long document (100+ pages)
with citation support and hallucination resistance.

## Problem Framing

This is a **document-grounded QA problem** where the core challenges are:

1. **Retrieval quality** — finding the right passages in a dense 100+ page document
2. **Citation fidelity** — every claim must trace back to specific pages and sections
3. **Hallucination control** — refuse to answer rather than fabricate

With only one document, the difficulty isn't scale, it's precision: locating the
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

### Why Hybrid Retrieval?

Dense and sparse search have complementary strengths. Dense (FAISS) handles paraphrases
("company revenue" ↔ "total sales") while sparse (BM25) handles exact matches ("GAAP", specific
dollar amounts). Financial and technical documents need both. RRF combines their rankings cleanly.

## Setup

### Requirements
- Python 3.9+
- OpenAI API key

### Install
```bash
pip install pymupdf sentence-transformers faiss-cpu rank_bm25 openai tiktoken
```

### Configure
```bash
export OPENAI_API_KEY="sk-..."
```

## How to Run

1. Open the notebook:
   ```bash
   jupyter notebook document_qa_system.ipynb
   ```

2. Set your PDF path in the notebook:
   ```python
   PDF_PATH = "/path/to/your/document.pdf"
   ```

3. Run cells sequentially (Sections 1–6 build the pipeline)

4. Ask questions:
   ```python
   result = ask("What was the total revenue?", index, CONFIG)
   ```

5. Run evaluation (Section 8) to score answers on faithfulness, relevance, and citation quality

## Evaluation

The system includes an **LLM-as-judge** evaluation that scores answers on:

- **Faithfulness** (1-5): Is every claim supported by the retrieved sources?
- **Relevance** (1-5): Does the answer address the question?
- **Citation Quality** (1-5): Are citations present, correct, and sufficient?

This provides a scalable quality signal. For production, human evaluation would be more reliable.

## Limitations

1. **Tables** — Simple tables extract well; complex merged cells may fail
2. **Images/Charts** — Text-only; figures are ignored
3. **Heading detection** — Font-size heuristic works for most documents, not all
4. **Single document** — One document at a time by design
5. **Evaluation** — LLM-as-judge has known biases; not a substitute for human review

## Domain Generalization

Nothing in this system is finance-specific:
- Chunking is structure-based, not content-based
- Embeddings are general-purpose
- The generation prompt is domain-agnostic

To specialize for a domain: fine-tune embeddings on domain data, add domain-specific
preprocessing, or customize the prompt.

## Future Improvements

With more time, in order of impact:

1. **Cross-encoder re-ranker** — Add a second-stage re-ranker (e.g., `ms-marco-MiniLM`) to improve precision after first-stage recall
2. **Index caching** — Serialize FAISS index to disk so re-embedding isn't needed on restart
3. **Iterative retrieval** — If first retrieval is weak, reformulate the query and search again
4. **Semantic chunking** — Use embedding similarity to find natural topic boundaries
5. **Query decomposition** — Break complex multi-part questions into sub-questions
6. **Multimodal** — Process charts and images with a vision model
