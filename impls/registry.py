from __future__ import annotations

from typing import Callable

from core.schemas import Article, LLMKeys, PipelineTrace


PipelineRunner = Callable[..., PipelineTrace]


def get_runner(name: str) -> PipelineRunner:
    normalized = name.lower().strip()
    if normalized == "static":
        from impls.static.pipeline import run

        return run
    if normalized == "langchain":
        from impls.langchain.pipeline import run

        return run
    if normalized == "langgraph":
        from impls.langgraph.pipeline import run

        return run
    raise ValueError(f"Unknown implementation {name!r}. Choose static, langchain, or langgraph.")


IMPLEMENTATIONS = ["static", "langchain", "langgraph"]
