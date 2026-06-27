# News Bias Multi-Agent Pipeline

Type `Manhattan Institute` and a lazy demo calls it neutral because the CSV ran out of adjectives. This app keeps source context visible, checks spans verbatim, and makes the critic argue with a real bias judgment.

## What it does

The app builds a framing brief from an article set. It answers three reader questions:

- What framing is visible?
- Which evidence supports that judgment?
- What should I check before I trust or share the story?

The public claim stays narrow. This is a source-framing and evidence-discipline demo, with limits spelled out in `docs/limitations.md`.

## Why there are three implementations

The same pipeline is implemented three ways so the framework differences can be inspected without changing the product claim.

| Implementation | File | What it teaches |
|---|---|---|
| Static Python | `impls/static/pipeline.py` | The baseline sequence with no framework |
| LangChain | `impls/langchain/pipeline.py` | Bounded runnable composition |
| LangGraph | `impls/langgraph/pipeline.py` | Explicit state-machine nodes and edges |

The UI should lead with the brief. The implementation comparison belongs under the hood, where it can help builders without confusing readers.

## Live app

Live Streamlit app: <https://news-bias-multi-agent-pipeline.streamlit.app/>

Streamlit Cloud entrypoint:

```text
streamlit_app.py
```

The app is BYOK. Visitors paste model and GNews keys into the sidebar for the current browser session. The no-key `heuristic` provider runs without external model calls and is the default for CI and quick review.

## What changed in the 2026 overhaul

- The bias detector is wired into the runtime. The critic reviews a real `StructuredBiasJudgment`.
- Outputs are Pydantic models: `StructuredSummary`, `StructuredBiasJudgment`, `StructuredCritique`, `ReconciledReport`, and `PipelineTrace`.
- Every cited span is verified verbatim against article text.
- The three promised implementations share the same core.
- Provider selection is explicit: `heuristic`, `anthropic`, `openai`, `google`, or `ollama`.
- Source-context cues identify common think tanks, opinion outlets, wire services, and public broadcasters. Uncataloged sources stay visible for the reader.
- Story selections can be shared with URL parameters. The URL never includes pasted API keys.
- `scripts/post_deploy_canary.py` checks the public Streamlit URL and verifies story packs do not collapse to one generic label or confidence.
- CI runs unit tests, eval fixtures, and import/build smoke checks.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m pytest
python main.py "AI regulation last week" --impl static --provider heuristic
python -m streamlit run streamlit_app.py
python scripts/post_deploy_canary.py --skip-url
```

Optional local env:

```powershell
Copy-Item .env.example .env
# Fill only the keys you want to use locally.
```

## Provider modes

| Provider | Key source | Notes |
|---|---|---|
| `heuristic` | none | Default. Deterministic, testable, no model calls |
| `anthropic` | sidebar or `ANTHROPIC_API_KEY` for CLI | Direct HTTP call |
| `openai` | sidebar or `OPENAI_API_KEY` for CLI | Direct Responses API call |
| `google` | sidebar or `GOOGLE_API_KEY` for CLI | Direct Gemini API call |
| `ollama` | local `OLLAMA_HOST` | Keeps the old local path |

## Project structure

- `app.py` - canonical Streamlit app.
- `streamlit_app.py` - Streamlit Cloud entrypoint.
- `main.py` - CLI entry point.
- `core/` - schemas, provider adapter, citation verifier, search, extraction, prompts, and pipeline stages.
- `impls/` - static, LangChain, and LangGraph implementations.
- `tests/` - unit, eval, and smoke tests.
- `docs/requirements.md` - traceable R-NB requirements.
- `docs/product_vision.md` - product end state and next increments.
- `docs/three_implementations.md` - framework comparison.
- `docs/limitations.md` - what the demo cannot claim.
- `docs/trust_model.md` - BYOK and deployment trust boundary.
- `docs/research.md` - sources used to shape the overhaul.

## Evaluation

`tests/eval/golden_fixtures.yaml` contains ten small labeled fixtures: left, right, mixed, neutral, and undetermined cases. The gate requires each implementation to hit at least 80 percent agreement in heuristic mode.

This gate catches regression in the demo fixtures. Research-grade bias measurement would need a larger labeled corpus and independent annotation.

## Connects to

- `ai-field-brief` for source-scouting discipline and source registry patterns.
- `LLM-evaluation-framework` for judge and deterministic evaluation patterns.
- `trace-to-eval-harness` for turning pipeline traces into review packets.

## Security note

Earlier history contained a hardcoded GNews key. The current code reads keys from explicit caller inputs or environment variables for local CLI only. The Streamlit app keeps keys in session state and never writes them to disk.
