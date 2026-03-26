# Technical Test: Build an AI System for Q&A on a Single Long Document

## Objective

Design and implement an AI system that can answer questions effectively from a single long document exceeding 100 pages.

This test is intentionally domain-independent. The purpose is not to optimize for one specific industry, but to assess how you approach an ambiguous applied AI problem from end to end.

We are evaluating less on the specific technology stack you choose, and more on whether you can:

- Break the problem down clearly
- Make thoughtful design decisions
- Do the necessary research
- Implement a sensible solution
- Test it in a structured and convincing way
- Explain your trade-offs and limitations

We will also discuss your design and thought process during the interview.

---

## Problem Statement

Build a system that can answer user questions based on the contents of one long document of more than 100 pages.

Your system should aim to produce answers that are:

- **Accurate**
- **Grounded** in the source document
- **Supported** by relevant evidence
- **Robust** to long, dense, and complex content
- **Resistant** to hallucination

The system does not need to be production-ready, but it should be designed thoughtfully enough that we can understand how you would approach a real-world implementation.

---

## Scope

You may build the system in any way you think is appropriate.

You do not need to use any specific model, framework, or vector database.

A simple approach is perfectly acceptable if it is well thought through and properly tested.

---

## Recommended Test Setup

Your system must be evaluated on one single long document exceeding 100 pages.

A finance document is one reasonable choice for testing, because it tends to contain:

- Long narrative sections
- Repeated concepts across sections
- Structured headings
- Tables and figures
- Dense factual content

Examples include:

- Annual reports
- 10-K filings
- Earnings reports
- Regulatory disclosures
[Finance long Document Benchmark](https://github.com/patronus-ai/financebench)

That said, the test is domain-independent. If you use a finance document, explain how your design would generalize to other domains.

---

## Deliverables

Your submission should be a **GitHub repository** containing your code and a README.

### README

Your README should include:

- **Problem framing** — how you interpreted and scoped the task
- **System design** — your architecture, major design choices, alternatives considered, and trade-offs
- **Setup instructions** — how to install dependencies and configure the environment
- **How to run the system** — clear steps to ingest a document and ask questions
- **Assumptions and limitations** — what you chose to leave out of scope and why
- **Future improvements** — what you would do next with more time

The README does not need to be long. We care more about clarity and depth of thought than volume.

### Code

Your repository should contain working code for the core system.

Your code should be understandable and reasonably organized. We do not require perfect engineering polish, but we do expect a structured implementation.

---


## Minimum Functional Requirements

At a minimum, your system should demonstrate the following.

### Document Handling

- Ingest and process a single document longer than 100 pages

### Question Answering

- Accept a user question about the document
- Identify the relevant content from the document
- Produce an answer grounded in that content

### Evidence / Grounding

- Show the supporting passages, sections, page numbers, or excerpts used to answer the question

---

## Constraints

Please treat this as a practical take-home exercise rather than an open-ended research project.

You are not expected to solve every edge case. We are more interested in whether you can make sensible scoping and design decisions under time constraints.

---
