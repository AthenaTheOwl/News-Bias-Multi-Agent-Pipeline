from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import requests

from core.schemas import LLMKeys


DEFAULT_MODELS = {
    "heuristic": "rules-v1",
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "ollama": "llama3",
}


@dataclass(frozen=True)
class LLMClient:
    provider: str
    model: str
    generate: Callable[[str], str]


def _missing(provider: str, key_name: str) -> ValueError:
    return ValueError(
        f"{provider} provider selected but {key_name} was not supplied. "
        "Use BYOK in the Streamlit sidebar, set the key in tests, or choose heuristic."
    )


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_llm(provider: str = "heuristic", model: str | None = None, keys: LLMKeys | None = None) -> LLMClient:
    """Return a provider-neutral text-generation client.

    The provider-specific API calls live here so agents and
    implementations never read environment variables or import vendor SDKs.
    The direct HTTP approach keeps Streamlit Cloud setup small and makes
    BYOK explicit.
    """

    keys = keys or LLMKeys()
    provider = (provider or "heuristic").lower().strip()
    model = model or DEFAULT_MODELS.get(provider, "rules-v1")

    if provider == "heuristic":
        return LLMClient(provider=provider, model=model, generate=lambda prompt: "")

    if provider == "anthropic":
        if not keys.anthropic_token:
            raise _missing("anthropic", "ANTHROPIC_API_KEY")

        def generate(prompt: str) -> str:
            data = _post_json(
                "https://api.anthropic.com/v1/messages",
                {
                    "x-api-key": keys.anthropic_token or "",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                {
                    "model": model,
                    "max_tokens": 1200,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return "\n".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ).strip()

        return LLMClient(provider=provider, model=model, generate=generate)

    if provider == "openai":
        if not keys.openai_token:
            raise _missing("openai", "OPENAI_API_KEY")

        def generate(prompt: str) -> str:
            data = _post_json(
                "https://api.openai.com/v1/responses",
                {
                    "Authorization": f"Bearer {keys.openai_token}",
                    "Content-Type": "application/json",
                },
                {
                    "model": model,
                    "input": prompt,
                    "temperature": 0.1,
                    "max_output_tokens": 1200,
                },
            )
            if data.get("output_text"):
                return str(data["output_text"]).strip()
            chunks: list[str] = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        chunks.append(content.get("text", ""))
            return "\n".join(chunks).strip()

        return LLMClient(provider=provider, model=model, generate=generate)

    if provider == "google":
        if not keys.google_token:
            raise _missing("google", "GOOGLE_API_KEY")

        def generate(prompt: str) -> str:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={keys.google_token}"
            )
            data = _post_json(
                url,
                {"Content-Type": "application/json"},
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200},
                },
            )
            parts: list[str] = []
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        parts.append(part["text"])
            return "\n".join(parts).strip()

        return LLMClient(provider=provider, model=model, generate=generate)

    if provider == "ollama":
        host = keys.ollama_host.rstrip("/")

        def generate(prompt: str) -> str:
            data = _post_json(
                f"{host}/api/generate",
                {"Content-Type": "application/json"},
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 8192},
                },
            )
            return str(data.get("response", "")).strip()

        return LLMClient(provider=provider, model=model, generate=generate)

    raise ValueError(f"Unknown LLM provider {provider!r}. Choose heuristic, anthropic, openai, google, or ollama.")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a model response."""

    if not text.strip():
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
