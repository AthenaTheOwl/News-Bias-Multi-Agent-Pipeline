# models/__init__.py
from typing import Any

try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    # fallback to deprecated import to keep your env working
    from langchain_community.llms import Ollama  # type: ignore

def _make(model_name: str):
    return Ollama(model=model_name)

# Your actual models
llama3 = _make("llama3")
gemma3 = _make("gemma3:4b")
deepseek = _make("deepseek-r1:8b")
