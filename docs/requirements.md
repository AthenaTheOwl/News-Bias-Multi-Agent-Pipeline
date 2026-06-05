# Requirements

These requirements define the 2026 overhaul. They are intentionally
small enough for a teaching repo and strict enough to prevent the old
placeholder behavior from returning.

## Core pipeline

**R-NB-001:** Ship three named implementations of the same pipeline:
`impls/static`, `impls/langchain`, and `impls/langgraph`.

Acceptance: each exposes `run(...) -> PipelineTrace`; all pass the same
fixture tests.

**R-NB-002:** Every implementation runs the sequence preprocess,
search/fetch, summarize, bias_detect, critique, reconcile.

Acceptance: every trace contains those six stage records in order.

**R-NB-003:** Bias judgments and critiques are structured Pydantic
models.

Acceptance: traces contain `StructuredBiasJudgment` and
`StructuredCritique`, not only free text.

**R-NB-004:** Citations use verbatim spans.

Acceptance: `core.citation.verify_citations` returns no errors for
every trace produced by tests.

## Providers and retrieval

**R-NB-005:** Provider selection supports heuristic, Anthropic, OpenAI,
Google, and Ollama.

Acceptance: provider code is isolated in `core/llm_provider.py`; agents
and implementations do not read model keys from environment variables.

**R-NB-006:** The Streamlit app uses BYOK.

Acceptance: visitor-supplied keys flow through `LLMKeys`; the app does
not write keys to disk.

**R-NB-007:** The no-key path remains runnable.

Acceptance: `provider=heuristic` passes all tests and can run the UI.

**R-NB-008:** News search uses GNews when a key is present and RSS
fallback when no key is present or GNews fails.

Acceptance: `tests/unit/test_news_search.py` covers fallback behavior.

## UI and deploy

**R-NB-009:** `app.py` is the canonical UI entry point.

Acceptance: duplicate old Streamlit files are removed.

**R-NB-010:** The UI exposes implementation and provider selectors.

Acceptance: sidebar controls select static, LangChain, or LangGraph and
heuristic, Anthropic, OpenAI, Google, or Ollama.

**R-NB-011:** The deployment target is Streamlit Community Cloud.

Acceptance: `.streamlit/config.toml`, `requirements.txt`, and
`docs/deploy.md` exist.

## Tests and docs

**R-NB-012:** Unit tests cover citation verification, provider key
handling, search fallback, and implementation trace shape.

Acceptance: `python -m pytest tests/unit` passes.

**R-NB-013:** Eval fixtures gate the bias-label behavior.

Acceptance: all three implementations score at least 80 percent on the
ten checked-in fixtures.

**R-NB-014:** CI runs tests on every push and pull request.

Acceptance: `.github/workflows/ci.yml` runs pytest.

**R-NB-015:** The README and docs keep the learning-project framing.

Acceptance: docs state that the project is not a production bias
detector and list concrete limitations.
