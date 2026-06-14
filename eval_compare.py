"""
eval_compare.py — Before/after A/B benchmark for the multimodal upgrades.

Runs the SAME FinanceBench questions through two pipeline configurations and
scores both with an LLM judge (gpt-4o-mini), so the effect of upgrades #1 + #2
is isolated:

  baseline : caption-then-embed only  (use_image_in_answer=False, no CLIP leg)
  upgraded : #1 images fed to the VLM at answer time  +  #2 CLIP visual retrieval
             +  numeric-aware BM25 (always on in rag_core)

Cost controls:
  • each document is ingested ONCE and its captions are cached on disk, so the
    two configs share captioning work (the expensive part) — you pay for it once.
  • --max-docs / --max-qs / --nonsense bound how much is run.

Usage:
  export OPENAI_API_KEY=...
  python eval_compare.py --max-docs 1 --max-qs 3 --nonsense 2     # tiny smoke run
  python eval_compare.py --max-docs 4 --max-qs 0                  # 0 = all questions
"""

import os, re, json, time, argparse, statistics
from dataclasses import dataclass
from typing import Optional, List

import rag_core
from rag_core import (
    make_config, ingest_document, chunk_elements, HybridIndex,
    get_embed_model, get_clip_model, get_reranker, ask, CaptionCache,
)

ABSTENTION_PHRASES = [
    "do not contain sufficient information", "does not contain sufficient information",
    "not contain enough information", "cannot find", "no information", "not mentioned",
    "not found in", "cannot be determined", "cannot be answered", "not available in",
    "not discussed", "no relevant information", "outside the scope", "not addressed",
    "does not appear", "i cannot answer", "unable to answer", "not possible to answer"]

NONSENSE_QUESTIONS = [
    "What is the meaning of life?", "Who invented algebra?",
    "What is the recipe for chocolate cake?", "How far is the Earth from the Sun?",
    "What is the plot of Romeo and Juliet?", "Who won the 2022 FIFA World Cup?"]


def detect_abstention(ans):
    return any(p in ans.lower() for p in ABSTENTION_PHRASES)


# ── FinanceBench loading ──────────────────────────────────────────────────────
def load_financebench(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def find_pdf(pdf_dir, doc):
    for f in os.listdir(pdf_dir):
        if f.endswith(".pdf") and doc.lower() in f.lower():
            return os.path.join(pdf_dir, f)
    return None


def select_documents(all_qs, pdf_dir, max_docs):
    dc = {}
    for q in all_qs:
        dc[q["doc_name"]] = dc.get(q["doc_name"], 0) + 1
    avail = []
    for d, c in sorted(dc.items(), key=lambda x: x[1], reverse=True):
        p = find_pdf(pdf_dir, d)
        if p:
            avail.append({"doc_name": d, "pdf_path": p, "num_questions": c, "is_10k": "10K" in d})
    avail.sort(key=lambda x: (x["is_10k"], x["num_questions"]), reverse=True)
    return avail[:max_docs] if max_docs > 0 else avail


# ── LLM judge ─────────────────────────────────────────────────────────────────
EVAL_B = """Compare the system answer to the gold answer for a financial-document question.
QUESTION: {q}
GOLD: {gold}
SYSTEM: {sys}
SOURCES: {src}
Respond with JSON only:
{{"faithfulness":<1-5>,"relevance":<1-5>,"citation_quality":<1-5>,"correctness":<1-5>,"reasoning":"<brief>"}}"""

EVAL_N = """Did the system correctly REFUSE this unanswerable question (it is not in the document)?
QUESTION: {q}
SYSTEM: {sys}
Respond with JSON only:
{{"faithfulness":<1-5>,"relevance":<1-5>,"reasoning":"<brief>"}}"""


def _judge(prompt, model, api_key):
    from openai import OpenAI
    try:
        r = OpenAI(api_key=api_key).chat.completions.create(
            model=model, temperature=0.0, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content
        m = re.search(r'\{.*\}', txt, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"    judge error: {e}")
    return {}


def judge_bench(result, gold, model, api_key):
    src = "\n".join(f"[S{s['source_num']}] {s['text']}" for s in result["sources"][:3])[:1800]
    return _judge(EVAL_B.format(q=result["query"], gold=gold, sys=result["answer"], src=src),
                  model, api_key)


def judge_nonsense(result, model, api_key):
    return _judge(EVAL_N.format(q=result["query"], sys=result["answer"]), model, api_key)


# ── Results ───────────────────────────────────────────────────────────────────
@dataclass
class Row:
    doc: str
    config: str          # "baseline" | "upgraded"
    qtype: str           # "benchmark" | "nonsense"
    question: str
    gold: Optional[str]
    answer: str
    faithfulness: Optional[float]
    relevance: Optional[float]
    citation: Optional[float]
    correctness: Optional[float]
    abstained: bool
    latency: float


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else 0.0


def summarize(rows, config):
    bench = [r for r in rows if r.config == config and r.qtype == "benchmark"]
    nons = [r for r in rows if r.config == config and r.qtype == "nonsense"]
    return {
        "faithfulness": _mean([r.faithfulness for r in bench]),
        "relevance": _mean([r.relevance for r in bench]),
        "citation": _mean([r.citation for r in bench]),
        "correctness": _mean([r.correctness for r in bench]),
        "abstention": (sum(1 for r in nons if r.abstained) / len(nons)) if nons else 0.0,
        "latency": _mean([r.latency for r in bench]),
        "n_bench": len(bench), "n_nons": len(nons),
    }


def write_report(rows, base_cfg, new_cfg, path):
    B, N = summarize(rows, "baseline"), summarize(rows, "upgraded")

    def delta(a, b, pct=False):
        d = b - a
        if pct:
            return f"{d*100:+.0f} pts"
        return f"{d:+.2f}"

    L = [
        "# Multimodal Upgrade — Before/After Comparison",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}  ·  **Benchmark:** FinanceBench  ·  "
        f"**Judge:** {new_cfg['llm_model']}  ·  **Questions:** {B['n_bench']} benchmark + {B['n_nons']} nonsense",
        "",
        "- **baseline** — caption-then-embed only (no image at answer time, no CLIP leg)",
        "- **upgraded** — #1 images fed to the VLM at answer time + #2 CLIP visual retrieval + numeric-aware BM25",
        "",
        "## Overall (LLM-judge, 1–5)",
        "",
        "| Metric | Baseline | Upgraded | Δ |",
        "|--------|----------|----------|---|",
        f"| Faithfulness | {B['faithfulness']:.2f} | {N['faithfulness']:.2f} | {delta(B['faithfulness'], N['faithfulness'])} |",
        f"| Relevance | {B['relevance']:.2f} | {N['relevance']:.2f} | {delta(B['relevance'], N['relevance'])} |",
        f"| Citation quality | {B['citation']:.2f} | {N['citation']:.2f} | {delta(B['citation'], N['citation'])} |",
        f"| **Correctness** | **{B['correctness']:.2f}** | **{N['correctness']:.2f}** | **{delta(B['correctness'], N['correctness'])}** |",
        f"| Nonsense abstention | {B['abstention']*100:.0f}% | {N['abstention']*100:.0f}% | {delta(B['abstention'], N['abstention'], pct=True)} |",
        f"| Avg latency (s) | {B['latency']:.2f} | {N['latency']:.2f} | {delta(B['latency'], N['latency'])} |",
        "",
        "## Per-question (correctness)",
        "",
        "| Doc | Question | Base | Upg |",
        "|-----|----------|------|-----|",
    ]
    by_q = {}
    for r in rows:
        if r.qtype != "benchmark":
            continue
        by_q.setdefault((r.doc, r.question), {})[r.config] = r
    for (doc, q), cfgs in by_q.items():
        b = cfgs.get("baseline"); u = cfgs.get("upgraded")
        L.append(f"| {doc.split('_')[0]} | {q[:60]}… | "
                 f"{b.correctness if b else '–'} | {u.correctness if u else '–'} |")
    L.append("")
    open(path, "w").write("\n".join(L))
    print(f"\n✅ Report → {path}")
    print(f"   Correctness: baseline {B['correctness']:.2f} → upgraded {N['correctness']:.2f} "
          f"({delta(B['correctness'], N['correctness'])})")


# ── Runner ────────────────────────────────────────────────────────────────────
def run(args):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    assert api_key, "Set OPENAI_API_KEY"
    all_qs = load_financebench(args.benchmark)
    docs = select_documents(all_qs, args.pdf_dir, args.max_docs)
    print(f"Selected {len(docs)} doc(s): " + ", ".join(d["doc_name"] for d in docs))

    em = get_embed_model()
    clip = get_clip_model()
    reranker = get_reranker()
    cache = CaptionCache(args.caption_cache)

    base_cfg = make_config(use_image_in_answer=False, use_clip_retrieval=False)
    new_cfg = make_config(use_image_in_answer=True, use_clip_retrieval=True,
                          image_detail=args.image_detail)

    rows: List[Row] = []
    for di, doc in enumerate(docs):
        dn = doc["doc_name"]
        print(f"\n[{di+1}/{len(docs)}] {dn} — ingesting once (cached captions)…")
        t0 = time.time()
        elements, meta = ingest_document(doc["pdf_path"], api_key, extract_images=True,
                                         caption_model=new_cfg["caption_model"], cache=cache)
        chunks = chunk_elements(elements, new_cfg["chunk_size"], new_cfg["chunk_overlap"])
        print(f"  {meta['total_pages']}p · {len(chunks)} chunks · "
              f"{meta.get('images_extracted',0)} imgs · {time.time()-t0:.0f}s")

        idx_base = HybridIndex(em, clip_model=None); idx_base.build(chunks)
        idx_new = HybridIndex(em, clip_model=clip); idx_new.build(chunks)

        dqs = [q for q in all_qs if q["doc_name"] == dn]
        if args.max_qs > 0:
            dqs = dqs[:args.max_qs]
        for qi, q in enumerate(dqs):
            print(f"  Q{qi+1}/{len(dqs)}: {q['question'][:55]}…")
            for cfg, idx, tag in ((base_cfg, idx_base, "baseline"), (new_cfg, idx_new, "upgraded")):
                t1 = time.time()
                r = ask(q["question"], idx, cfg, reranker=reranker, api_key=api_key)
                ev = judge_bench(r, q["answer"], cfg["llm_model"], api_key)
                rows.append(Row(dn, tag, "benchmark", q["question"], q["answer"], r["answer"],
                                ev.get("faithfulness"), ev.get("relevance"),
                                ev.get("citation_quality"), ev.get("correctness"),
                                detect_abstention(r["answer"]), time.time() - t1))

        for q in NONSENSE_QUESTIONS[:args.nonsense]:
            for cfg, idx, tag in ((base_cfg, idx_base, "baseline"), (new_cfg, idx_new, "upgraded")):
                t1 = time.time()
                r = ask(q, idx, cfg, reranker=reranker, api_key=api_key)
                ev = judge_nonsense(r, cfg["llm_model"], api_key)
                rows.append(Row(dn, tag, "nonsense", q, None, r["answer"],
                                ev.get("faithfulness"), ev.get("relevance"), None, None,
                                detect_abstention(r["answer"]), time.time() - t1))

    write_report(rows, base_cfg, new_cfg, args.out)
    json.dump([r.__dict__ for r in rows], open(args.out.replace(".md", ".json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="./financebench/data/financebench_open_source.jsonl")
    ap.add_argument("--pdf-dir", default="./financebench/pdfs")
    ap.add_argument("--out", default="./comparison_results.md")
    ap.add_argument("--caption-cache", default="./.caption_cache.json")
    ap.add_argument("--max-docs", type=int, default=2)
    ap.add_argument("--max-qs", type=int, default=3, help="per doc; 0 = all")
    ap.add_argument("--nonsense", type=int, default=2)
    ap.add_argument("--image-detail", default="auto", choices=["low", "high", "auto"])
    run(ap.parse_args())


if __name__ == "__main__":
    main()
