# Document Q&A System — Evaluation Results

**Generated:** 2026-04-05 20:33:51
**Benchmark:** FinanceBench (open-source subset)
**Documents evaluated:** 4
**Benchmark questions:** 10
**Adversarial (nonsense) questions:** 12

---
## Overall Scores

| Metric | Score |
|--------|-------|
| Faithfulness | **3.46** / 5 |
| Relevance | **3.21** / 5 |
| Citation Quality | **2.64** / 5 |
| Correctness (vs gold) | **2.64** / 5 |
| Nonsense Abstention Rate | **100%** |
| Average Latency | **1.7s** per question |

> **Scoring guide:** Each dimension is rated 1-5 by an LLM judge.
> Faithfulness = grounded in sources. Relevance = answers the question.
> Citation Quality = proper [Source N] references. Correctness = matches gold answer.
> Abstention Rate = % of nonsense questions correctly refused (higher is better).

---
## Per-Document Results

| Document | Pages | Chunks | Qs | Faith. | Relev. | Cite. | Correct. | Abstain | Latency |
|----------|-------|--------|----|--------|--------|-------|----------|---------|---------|
| AMD_2022_10K | 121 | 359 | 7 | 3.9 | 3.9 | 3.6 | 3.6 | 100% | 3.2s |
| ADOBE_2017_10K | 107 | 299 | 1 | 3.0 | 2.0 | 1.0 | 1.0 | 100% | 1.1s |
| AMCOR_2020_10K | 195 | 658 | 1 | 2.0 | 2.0 | 1.0 | 1.0 | 100% | 0.9s |
| COCACOLA_2022_10K | 183 | 521 | 1 | 5.0 | 5.0 | 5.0 | 5.0 | 100% | 1.7s |

---
## Detailed Results

### AMD_2022_10K

#### Benchmark Questions

**❌ Q1: Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22? If the quick ratio is not relevant to measure liquidity, please state that and explain why.**

- **Gold answer:** Yes. The quick ratio is 1.57, calculated as (cash and cash equivalents+Short term investments+Accounts receivable, net+receivables from related parties)/ (current liabilities).
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=1 Rel=1 Cite=1 Correct=1
- **Sources:** [p. 18 — "Liquidity and Capital Resources Risks"], [p. 53 — "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURE ABOUT MARKET RISK"], [p. 82 — "Stock-Based Compensation"]
- **Judge reasoning:** The system answer does not provide any relevant information from the retrieved sources to address the question about AMD's liquidity profile or quick 
- **Latency:** 2.1s

**✅ Q2: What are the major products and services that AMD sells as of FY22?**

- **Gold answer:** AMD sells server microprocessors (CPUs) and graphics processing units (GPUs), data processing units (DPUs), Field Programmable Gate Arrays (FPGAs), and Adaptive System-on-Chip (SoC) products for data ...
- **System answer:** As of FY22, AMD sells a variety of major products and services, including:

1. **Microprocessors (CPUs)**: AMD's server microprocessors include the AMD EPYC™ Series, specifically the AMD EPYC 9004 Ser...
- **Scores:** Faith=5 Rel=5 Cite=5 Correct=4
- **Sources:** [p. 43 — "Overview"], [pp. 10, 11 — "Sales and Marketing"], [p. 60 — "NOTE 1 – The Company"]
- **Judge reasoning:** The system answer is well-supported by the retrieved sources and directly addresses the question about AMD's major products and services. All claims a
- **Latency:** 10.2s

**✅ Q3: What drove revenue change as of the FY22 for AMD?**

- **Gold answer:** In 2022, AMD reported Higher sales of their EPYC server processors, higher semi-custom product sales, and the inclusion of Xilinx embedded product sales
- **System answer:** The revenue change for AMD in FY22 was primarily driven by a 64% increase in Data Center segment revenue, which was mainly due to higher sales of EPYC™ server processors. Additionally, there was a 21%...
- **Scores:** Faith=5 Rel=5 Cite=4 Correct=4
- **Sources:** [pp. 104, 105 — "Exhibit
Description of Exhibits"], [p. 43 — "Overview"], [pp. 16, 17 — "Environmental Regulations"]
- **Judge reasoning:** The system answer is well-supported by the retrieved sources, particularly regarding the revenue changes in different segments. It directly addresses 
- **Latency:** 4.1s

**✅ Q4: What drove operating margin change as of the FY22 for AMD? If operating margin is not a useful metric for a company like this, then please state that and explain why.**

- **Gold answer:** The decrease in AMD's operating income was primarily driven by amortization of intangible assets associated with the Xilinx acquisition
- **System answer:** The change in operating margin for AMD in FY22 was primarily driven by the amortization of intangible assets associated with the Xilinx acquisition. Operating income for 2022 was reported at $1.3 bill...
- **Scores:** Faith=5 Rel=5 Cite=4 Correct=5
- **Sources:** [p. 68 — "NOTE 4 – Segment Reporting"], [p. 65 — "Income Taxes"], [p. 43 — "Introduction"]
- **Judge reasoning:** The system answer is well-supported by the retrieved sources and directly addresses the question about AMD's operating margin change. It accurately re
- **Latency:** 6.5s

**✅ Q5: Among operations, investing, and financing activities, which brought in the most (or lost the least) cash flow for AMD in FY22?**

- **Gold answer:** In 2022, AMD brought in the most cashflow from Operations
- **System answer:** In FY22, AMD's cash flow from operating activities provided the most cash, amounting to $3,565 million. This was followed by investing activities, which provided $1,999 million. Financing activities, ...
- **Scores:** Faith=5 Rel=5 Cite=5 Correct=5
- **Sources:** [p. 51 — "Liquidity and Capital Resources"], [pp. 59, 60 — "(In millions)
Supplemental cash flow information:"], [p. 52 — "Investing Activities"]
- **Judge reasoning:** The system answer is fully supported by the retrieved sources, directly addresses the question about cash flow activities for AMD in FY22, includes co
- **Latency:** 2.1s

**✅ Q6: From FY21 to FY22, excluding Embedded, in which AMD reporting segment did sales proportionally increase the most?**

- **Gold answer:** Data Center
- **System answer:** From FY21 to FY22, excluding Embedded, the AMD reporting segment that saw the highest proportional increase in sales was the Data Center segment, which experienced a 64% increase in revenue. In contra...
- **Scores:** Faith=5 Rel=5 Cite=5 Correct=5
- **Sources:** [p. 43 — "Overview"], [p. 75 — "(In millions)
Reportable segments before segment change:"], [p. 67 — "NOTE 4 – Segment Reporting"]
- **Judge reasoning:** The system answer accurately reflects the information from the retrieved sources, directly addresses the question, provides correct citations, and mat
- **Latency:** 2.0s

**❌ Q7: Did AMD report customer concentration in FY22?**

- **Gold answer:** Yes, one customer accounted for 16% of consolidated net revenue
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=1 Rel=1 Cite=1 Correct=1
- **Sources:** [p. 5 — "Additional Information"], [p. 105 — "Exhibit
Description of Exhibits"], [p. 43 — "Overview"]
- **Judge reasoning:** The system answer does not provide any information from the retrieved sources that supports the question about customer concentration. It fails to add
- **Latency:** 1.7s

#### Adversarial (Nonsense) Questions

**✅ Q1: What is the meaning of life?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question is unanswerable based on the provided document and refused to provide an answer, thus avoiding any f

**✅ Q2: Who invented algebra?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the invention of algebra is not addressed in the provided document sections, thus it refused t

**✅ Q3: What is the recipe for chocolate cake?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the chocolate cake recipe was unanswerable based on the provided document, and it appropriatel

### ADOBE_2017_10K

#### Benchmark Questions

**❌ Q1: What is the FY2017 operating cash flow ratio for Adobe? Operating cash flow ratio is defined as: cash from operations / total current liabilities. Round your answer to two decimal places. Please utilize information provided primarily within the balance sheet and the cash flow statement.**

- **Gold answer:** 0.83
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=3 Rel=2 Cite=1 Correct=1
- **Sources:** [pp. 50, 51 — "Other Liquidity and Capital Resources Considerations"], [p. 54 — "Fiscal"], [p. 4 — "SEGMENTS"]
- **Judge reasoning:** The system answer states that there is insufficient information to answer the question, which is somewhat supported by the retrieved sources, but it d
- **Latency:** 1.4s

#### Adversarial (Nonsense) Questions

**✅ Q1: What is the meaning of life?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the meaning of life is unanswerable based on the provided document, and it appropriately refus

**✅ Q2: Who invented algebra?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the invention of algebra is not addressed in the provided document sections, thus it appropria

**✅ Q3: What is the recipe for chocolate cake?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the chocolate cake recipe was unanswerable based on the provided document, and it appropriatel

### AMCOR_2020_10K

#### Benchmark Questions

**❌ Q1: What is Amcor's year end FY2020 net AR (in USD millions)? Address the question by adopting the perspective of a financial analyst who can only use the details shown within the balance sheet.**

- **Gold answer:** $1616.00
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=2 Rel=2 Cite=1 Correct=1
- **Sources:** [p. 31 — "(in millions)
2020
2019"], [p. 141 — "Fraudulent conveyance or transfer considerations"], [p. 31 — "(in millions)
2020
2019
2018"]
- **Judge reasoning:** The system answer claims insufficient information is available, which is not accurate as the retrieved sources contain relevant financial data. Howeve
- **Latency:** 0.8s

#### Adversarial (Nonsense) Questions

**✅ Q1: What is the meaning of life?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the meaning of life is unanswerable based on the provided document, and it appropriately refus

**✅ Q2: Who invented algebra?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the invention of algebra is unrelated to the provided document and appropriately refused to an

**✅ Q3: What is the recipe for chocolate cake?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the chocolate cake recipe was unanswerable based on the provided document, and it did not fabr

### COCACOLA_2022_10K

#### Benchmark Questions

**✅ Q1: What is Coca Cola's FY2022 dividend payout ratio (using total cash dividends paid and net income attributable to shareholders)? Round answer to two decimal places. Answer the question asked by assuming you only have access to information clearly displayed in the cash flow statement and the income statement.**

- **Gold answer:** 0.8
- **System answer:** To calculate Coca-Cola's FY2022 dividend payout ratio, we use the formula:

\[
\text{Dividend Payout Ratio} = \frac{\text{Total Dividends Paid}}{\text{Net Income Attributable to Shareowners}}
\]

From...
- **Scores:** Faith=5 Rel=5 Cite=5 Correct=5
- **Sources:** [pp. 67, 68 — "Year Ended December 31,
2022
2021
2020"], [p. 67 — "Year Ended December 31,
2022
2021
2020"], [p. 132 — "ITEM 15.  EXHIBITS AND FINANCIAL STATEMENT SCHEDULES"]
- **Judge reasoning:** The system answer accurately calculates the dividend payout ratio using the provided figures from the sources, directly addressing the question asked.
- **Latency:** 3.6s

#### Adversarial (Nonsense) Questions

**✅ Q1: What is the meaning of life?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the meaning of life is unanswerable based on the provided document, and it appropriately refus

**✅ Q2: Who invented algebra?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the invention of algebra is unrelated to the provided document and appropriately refused to an

**✅ Q3: What is the recipe for chocolate cake?**

- **Abstained:** Yes ✅
- **System answer:** The provided document sections do not contain sufficient information to answer this question.
- **Scores:** Faith=5 Rel=5
- **Judge reasoning:** The system correctly identified that the question about the chocolate cake recipe is unanswerable based on the provided document, and it did not fabri

---
## Configuration

```json
{
  "pdf_dir": "./financebench/pdfs",
  "benchmark_file": "./financebench/data/financebench_open_source.jsonl",
  "results_file": "./results.md",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "embedding_model": "BAAI/bge-base-en-v1.5",
  "top_k": 15,
  "top_k_final": 6,
  "rrf_k": 60,
  "llm_model": "gpt-4o-mini",
  "temperature": 0.0,
  "max_tokens": 1024,
  "max_documents": 10
}
```

---
## System Architecture

```
PDF → PyMuPDF Parser → Section-Aware Chunker → [FAISS + BM25] → RRF Fusion → LLM + Citations
```

| Component | Choice |
|-----------|--------|
| Embedding | BAAI/bge-base-en-v1.5 |
| Chunk size | 512 tokens (overlap 64) |
| Retrieval | Hybrid FAISS + BM25 with RRF (k=60) |
| Top-k | 15 candidates → 6 to LLM |
| LLM | gpt-4o-mini (temp=0.0) |
