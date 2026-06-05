from __future__ import annotations

import pytest

from core.llm_provider import get_llm
from core.schemas import LLMKeys


def test_heuristic_provider_needs_no_key() -> None:
    client = get_llm("heuristic", keys=LLMKeys())
    assert client.provider == "heuristic"
    assert client.generate("ignored") == ""


def test_missing_openai_key_fails_fast() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_llm("openai", keys=LLMKeys())


def test_unknown_provider_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("unknown", keys=LLMKeys())
