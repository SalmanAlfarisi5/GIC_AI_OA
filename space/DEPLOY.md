# Deploying the demo (Hugging Face Space) + portfolio wiring

This mirrors how `saLLMan` is deployed: a Gradio Space holds the model + the
OpenAI key (as a secret), and the static portfolio calls it with `@gradio/client`.

> **Current status:** the Space **already exists and is RUNNING** at
> `Salmanalfarisi1/multimodal-rag-demo` — Gradio **5.49.1** / Python **3.12**
> (matching the saLLMan Space, which works with `@gradio/client` 2.2.1). It was
> deployed **without** `OPENAI_API_KEY`, so it is **inert and cannot spend** —
> `/ask` returns "Demo unavailable" until the key is added. The only steps left
> are **§0 (budget cap)** then **§3 (add the key)**. Everything else below is
> reference / how it was built.

## 0. Before anything — set a hard spending cap (do this first)

The single most important guardrail. In the OpenAI dashboard:

1. **Billing → Limits → set a low "hard" monthly budget cap** (e.g. $10–20). At
   the cap, the API stops serving — your worst case is bounded no matter what.
2. Create a **dedicated API key** for this demo (so you can revoke it without
   affecting anything else).

The Space-side rate limits below reduce the chance you ever reach the cap; the
cap guarantees you can't exceed it.

## 1. Build the sample indexes (one time)

From the repo root, with the ML deps installed:

```bash
pip install -r requirements.txt          # add gradio + pillow if not present
python build_samples.py                  # → writes space/samples/*.pkl + manifest.json
```

The two bundled samples are **text/markdown**, so this needs **no** OpenAI key.
If you add an image-bearing PDF to `space/sample_docs/`, set `OPENAI_API_KEY`
first so its figures get captioned once (the captions are baked into the pickle;
the Space never re-captions samples).

> The Space also self-builds the text samples at startup if no pickles are
> present, so this step is optional for the shipped samples — but pre-building is
> recommended (faster, deterministic cold starts).

## 2. Create the Space

```bash
# On huggingface.co: New Space → SDK: Gradio → CPU basic (free)
git clone https://huggingface.co/spaces/<you>/multimodal-rag-demo
cp -r space/* multimodal-rag-demo/        # app.py, rag_core.py, requirements.txt,
                                          # README.md, sample_docs/, samples/
cd multimodal-rag-demo && git add . && git commit -m "Multimodal RAG demo" && git push
```

`space/rag_core.py` is a vendored copy of the repo-root `rag_core.py`. If you
change the root version, re-copy it (`cp rag_core.py space/rag_core.py`).

## 3. Set Space secrets / variables (Settings → Variables and secrets)

| Name | Type | Value |
|------|------|-------|
| `OPENAI_API_KEY` | **secret** | the dedicated demo key from step 0 |
| `SESSION_MAX_Q` | variable | `20` (default) — tune freely |
| `GLOBAL_MAX_Q_PER_HOUR` | variable | `150` |
| `GLOBAL_MAX_Q_PER_DAY` | variable | `800` |
| `UPLOAD_MAX_PAGES` | variable | `8` |
| `UPLOAD_MAX_CAPTIONS` | variable | `6` |
| `DEMO_DISABLED` | variable | `1` to instantly switch the demo off |

The Space restarts and is live at `https://<you>-multimodal-rag-demo.hf.space`.

## 4. Point the portfolio at the Space

In `Portfolio-Website/src/lib/multimodalRag.js`, set:

```js
export const SPACE_ID = '<you>/multimodal-rag-demo';
export const SPACE_APP_URL = 'https://<you>-multimodal-rag-demo.hf.space';
export const SPACE_PAGE_URL = 'https://huggingface.co/spaces/<you>/multimodal-rag-demo';
```

Then `npm run build && npm run preview` to check locally, and deploy as usual.

> **Gradio/client version:** the Space is pinned to `sdk_version: 5.49.1` /
> `python_version: "3.12"` in `README.md` — the same as the saLLMan Space, which
> works with the portfolio's `@gradio/client` 2.2.1. Don't pin `gradio` in
> `requirements.txt`; let `sdk_version` drive it. (Older Gradio 4.x fails on this
> 2026 build env: Python 3.13 removed stdlib `audioop`, and newer
> `huggingface_hub` dropped `HfFolder` — both of which Gradio 4.44.1 needs.)

## 5. Run the before/after benchmark (optional, costs API budget)

```bash
export OPENAI_API_KEY=...
python eval_compare.py --max-docs 1 --max-qs 3 --nonsense 2    # ~ a few cents, smoke test
python eval_compare.py --max-docs 4 --max-qs 0                 # full run → comparison_results.md
```
