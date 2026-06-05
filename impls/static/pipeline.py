"""Static implementation: the reference mental model.

What this version teaches:

- A multi-agent pipeline does not require a framework. The core idea is
  a sequence of small roles with typed inputs and outputs.
- The same evidence contracts used by LangChain and LangGraph are visible
  here without orchestration machinery.
- This is the baseline for debugging. If a framework version acts oddly,
  compare its trace to this one first.

How it differs from siblings:

- LangChain wraps the same stages as runnable composition.
- LangGraph models the stages as explicit state transitions.
- Static Python is the least abstract and the easiest to step through.
"""

from __future__ import annotations

from core.pipeline import run_pipeline
from core.schemas import Article, LLMKeys, PipelineTrace


def run(
    subject: str,
    *,
    provider: str = "heuristic",
    model: str | None = None,
    keys: LLMKeys | None = None,
    max_articles: int = 5,
    fixture_articles: list[Article] | None = None,
) -> PipelineTrace:
    return run_pipeline(
        subject,
        implementation="static",
        provider=provider,
        model=model,
        keys=keys,
        max_articles=max_articles,
        fixture_articles=fixture_articles,
        framework_notes=[
            "Sequential Python calls; no orchestration framework.",
            "Best baseline for inspecting typed stage outputs.",
        ],
    )
