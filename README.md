# News Bias Multi-Agent Pipeline

A framing-brief generator for inspecting how an article set presents a story.
It is also a learning artifact for agent orchestration and evidence discipline.
It is not a production bias detector.

The public app should answer a simple reader question first: what framing is
visible here, what evidence supports that, and what should I check before I
trust or share the story?

The implementation comparison is now under the hood. The same pipeline is
implemented three ways so framework differences remain inspectable:

| Implementation | File | What it teaches |
|---|---|---|
| Static Python | `impls/static/pipeline.py` | The baseline sequence with no framework |
| LangChain | `impls/langchain/pipeline.py` | Bounded runnable composition |
| LangGraph | `impls/langgraph/pipeline.py` | Explicit state machine nodes and edges |

## Deployment target

Target host: Streamlit Community Cloud.

Public URL: `https://news-bias-multi-agent-pipeline.streamlit.app/`

Entry point: `app.py`

The app is BYOK. Visitors paste model and GNews keys into the sidebar
for the current browser session. The no-key `heuristic` provider runs
without external model calls and is the default for CI and quick review.

Why Streamlit Cloud: this is a Python teaching demo with a Streamlit UI
already at the product boundary. A Vercel deployment would require a
separate Python service or a full Next.js rewrite before the core idea is
visible.

## What changed in the 2026 overhaul

- The bias detector is now wired into the runtime. The critic reviews a
  real `StructuredBiasJudgment`, not a placeholder score.
- Outputs are Pydantic models: `StructuredSummary`,
  `StructuredBiasJudgment`, `StructuredCritique`, `ReconciledReport`,
  and `PipelineTrace`.
- Every cited span is verified verbatim against the article text.
- The three promised implementations exist and share the same core.
- Provider selection is explicit: `heuristic`, `anthropic`, `openai`,
  `google`, or `ollama`.
- Dead FAISS, sqlite, duplicate UI, pycache, and setup-script artifacts
  were removed from the active runtime.
- Source-context cues now identify common think tanks, opinion outlets,
  wire services, and public broadcasters. Uncataloged sources stay visible
  instead of being silently treated as neutral.
- Story selections can be shared with URL parameters. The URL never includes
  pasted API keys.
- `scripts/post_deploy_canary.py` checks the public Streamlit URL and verifies
  story packs do not collapse to one generic label or confidence.
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
| `ollama` | local `OLLAMA_HOST` | Keeps the old no-cloud local path |

## Project structure

- `app.py` - canonical Streamlit app.
- `main.py` - CLI entry point.
- `core/` - shared schemas, provider adapter, citation verifier, search,
  extraction, prompts, and pipeline stages.
- `impls/` - static, LangChain, and LangGraph implementations.
- `tests/` - unit, eval, and smoke tests.
- `docs/requirements.md` - traceable R-NB requirements.
- `docs/product_vision.md` - product end state and next increments.
- `docs/three_implementations.md` - framework comparison.
- `docs/limitations.md` - what the demo cannot claim.
- `docs/trust_model.md` - BYOK and deployment trust boundary.
- `docs/research.md` - sources used to shape the overhaul.

## Evaluation

`tests/eval/golden_fixtures.yaml` contains ten small labeled fixtures:
left, right, mixed, neutral, and undetermined cases. The gate requires
each implementation to hit at least 80 percent agreement in heuristic
mode.

This is a regression gate, not a research benchmark. The limitations are
documented in `docs/limitations.md`.

## live demo

Live Streamlit app: <https://news-bias-multi-agent-pipeline.streamlit.app/>

Streamlit Cloud entrypoint:

```text
streamlit_app.py
```

## connects to

- `ai-field-brief` for source-scouting discipline and source registry patterns.
- `LLM-evaluation-framework` for judge and deterministic evaluation patterns.
- `trace-to-eval-harness` for turning pipeline traces into review packets.

## Security note

Earlier history contained a hardcoded GNews key. The current code reads
keys from explicit caller inputs or environment variables for local CLI
only. The Streamlit app keeps keys in session state and never writes
them to disk.
