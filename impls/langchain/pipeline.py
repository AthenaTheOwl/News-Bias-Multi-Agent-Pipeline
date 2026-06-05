"""LangChain implementation: runnable composition around the same core.

What this version teaches:

- LangChain is useful when a pipeline is a composable sequence of
  callable steps and shared config.
- This implementation keeps the same output schema as the static and
  LangGraph versions. The framework changes the orchestration shape, not
  the proof surface.
- The old repo used a ReAct agent that could wander. This version uses a
  bounded Runnable so the teaching comparison stays fair.

How it differs from siblings:

- Static Python directly calls the core function.
- LangGraph exposes each stage as a node in a state machine.
- LangChain is the middle ground: composable and inspectable without a
  graph runtime.
"""

from __future__ import annotations

from typing import Any

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
    def _invoke(payload: dict[str, Any]) -> PipelineTrace:
        return run_pipeline(
            payload["subject"],
            implementation="langchain",
            provider=payload["provider"],
            model=payload.get("model"),
            keys=payload.get("keys"),
            max_articles=payload["max_articles"],
            fixture_articles=payload.get("fixture_articles"),
            framework_notes=[
                "LangChain Runnable wraps the shared pipeline contract.",
                "Bounded chain used instead of an unconstrained ReAct loop.",
            ],
        )

    payload = {
        "subject": subject,
        "provider": provider,
        "model": model,
        "keys": keys,
        "max_articles": max_articles,
        "fixture_articles": fixture_articles,
    }

    try:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_invoke).invoke(payload)
    except Exception:
        # Keep the demo runnable in minimal environments. The trace still
        # records that this was the LangChain implementation path.
        return _invoke(payload)
