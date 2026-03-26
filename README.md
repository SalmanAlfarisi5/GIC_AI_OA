# Document-Grounded Q&A System for Long Documents

A retrieval-augmented generation (RAG) system that answers questions accurately from a single long document (100+ pages), with citation support and hallucination resistance.

## Problem Framing

I interpret this as a **document-grounded QA problem** over a single long source, where the core challenges are:

1. **Retrieval quality** — finding the right passages in a 100+ page document
2. **Citation fidelity** — every claim in the answer must be traceable to specific pages/sections
3. **Hallucination control** — the system must refuse to answer rather than fabricate

Since we're working with a single document (not a large corpus), the difficulty is not scale but **precision**: correctly locating the most relevant parts of a long, dense text and generating answers strictly grounded in those parts.

## System Design

### Architecture

```
PDF Document
    │
    ▼
┌─────────────────┐
│  PDF Parser      │  PyMuPDF: text, headings, tables
│  (Ingestion)     │  with page-level metadata
└────────┬────────┘
         ▼
┌─────────────────┐
│  Structure-Aware │  Heading-first grouping →
│  Chunker         │  token-bounded with overlap
└────────┬────────┘
         ▼
┌──────────────────────────┐
│  Dual Index               │
│  FAISS (dense) + BM25     │
│  (sparse)                 │
└────────┬─────────────────┘
         ▼
┌─────────────────┐
│  RRF Fusion      │  Reciprocal Rank Fusion
└────────┬────────┘
         ▼
┌─────────────────┐
│  Cross-Encoder   │  Re-rank for precision
│  Re-ranker       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM Generator   │  Strict grounding prompt
│  + Citations     │  with source attribution
└─────────────────┘
```

### Major Design Choices

| Component | Choice | Why | Alternatives Considered |
|-----------|--------|-----|------------------------|
| PDF Parser | PyMuPDF (fitz) | Fast, good structure extraction, built-in table support | pdfplumber (slower), unstructured (heavier), pdfminer (less structure) |
| Chunking | Heading-aware + token overlap | Preserves topical coherence within sections | Recursive character splitting (heading-unaware), sentence-level (too granular), semantic chunking (slower) |
| Embeddings | bge-base-en-v1.5 | Top retrieval quality in its size class (MTEB), 768-dim | all-MiniLM-L6 (faster/weaker), e5-large (better/slower) |
| Vector Store | FAISS IndexFlatIP | Single document = small corpus; exact search is fine; no overhead | Chroma/Qdrant (overkill for single doc), HNSW (unnecessary for <10k vectors) |
| Sparse Retrieval | BM25 (Okapi) | Essential for exact term matching (ticker symbols, accounting terms, specific numbers) | TF-IDF (weaker), SPLADE (better but heavier) |
| Fusion | Reciprocal Rank Fusion (k=60) | Score-agnostic — no need to normalize incompatible score distributions | Linear combination (requires score normalization), learned fusion (needs training data) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6 | Good accuracy/latency tradeoff for second-stage precision | Larger cross-encoders (slower), Cohere rerank API (external dependency) |
| LLM | gpt-4o-mini | Cost-effective, strong instruction following for grounding | gpt-4o (higher quality/cost), Claude (equally capable, different API) |
| Citations | Source numbering + page/section refs | Clear, verifiable, easy to trace back to document | Inline quotes (verbose), page-only (less precise) |

### Why Hybrid Retrieval?

Dense and sparse retrieval have complementary strengths:

- **Dense (FAISS)**: Handles paraphrases and semantic similarity ("company revenue" ↔ "total sales")
- **Sparse (BM25)**: Handles exact matches critical in technical documents ("GAAP", "10-K", specific numbers)

For financial and technical documents, relying on either alone leaves gaps. RRF fusion combines them without needing score normalization.

### Why Two-Stage Retrieval?

First-stage retrieval (top-20 from each retriever) optimizes **recall** — cast a wide net. The cross-encoder re-ranker then optimizes **precision** — select the truly relevant passages. Cross-encoders are too slow for full-corpus search but excellent for re-scoring a small candidate set.

## Setup Instructions

### Prerequisites

- Python 3.9+
- An OpenAI API key (for LLM generation and evaluation)

### Installation

```bash
pip install pymupdf sentence-transformers faiss-cpu rank_bm25 openai tiktoken tabulate
```

### Configuration

```bash
export OPENAI_API_KEY="sk-..."
```

## How to Run

### 1. Open the Notebook

```bash
jupyter notebook document_qa_system.ipynb
```

### 2. Set Your Document Path

In the notebook, update the `PDF_PATH` variable to point to your PDF:

```python
PDF_PATH = "/path/to/your/document.pdf"
```

### 3. Run All Cells

Run cells sequentially. The pipeline will:
1. Parse the PDF and extract structured content
2. Create heading-aware chunks with metadata
3. Build dual FAISS + BM25 indices
4. Load the cross-encoder re-ranker

### 4. Ask Questions

Use the pipeline directly:

```python
result = pipeline.ask("What was the total revenue?")
```

Or start the interactive session:

```python
interactive_qa(pipeline)
```

### 5. Evaluate

The evaluation section runs a battery of test questions and scores answers on faithfulness, relevance, and citation quality using LLM-as-judge.

## Assumptions and Limitations

### Assumptions
- Input is a single PDF document (not scanned images — must have extractable text)
- The document is in English
- An OpenAI API key is available for generation (retrieval works offline)

### Limitations

1. **Tables**: PyMuPDF handles simple table layouts well but may struggle with complex merged cells. For production, Camelot or Tabula could serve as fallbacks.
2. **Images/Charts**: Text-only extraction. Figures and charts are ignored. A multimodal approach (GPT-4V) would be needed to handle these.
3. **Cross-document reasoning**: Not supported. Designed for single-document QA.
4. **Heading detection**: Uses font-size heuristics, which work for most documents but may fail on unusual layouts. Using the PDF's outline/bookmark structure would be more robust.
5. **Evaluation**: LLM-as-judge has known biases (e.g., preference for longer answers). Human evaluation would be more reliable for production assessment.

### Domain Generalization

The system is domain-agnostic by design:
- No domain-specific rules, ontologies, or preprocessing
- Structure-based chunking works across document types
- General-purpose embeddings and prompts

To specialize for a domain: fine-tune embeddings on domain data, add domain-specific preprocessing, or customize the generation prompt.

## Future Improvements

With more time, I would prioritize:

1. **Iterative/agentic retrieval** — Let the LLM request additional context if the first retrieval is insufficient
2. **Semantic chunking** — Use embedding similarity between sentences to find natural breakpoints instead of fixed token windows
3. **Index caching** — Serialize FAISS index and embeddings to disk to avoid re-embedding on repeated use
4. **Query decomposition** — Break complex multi-part questions into sub-questions
5. **Confidence estimation** — Score answer reliability based on retrieval scores and source coverage
6. **Multimodal support** — Process charts, figures, and images using vision models
7. **Streaming** — Stream the LLM response for better UX
8. **Better evaluation** — Use established benchmarks (e.g., FinanceBench) with gold-standard answers
