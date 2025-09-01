# agents/summarizer.py
import requests
from typing import List, Tuple

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"   # or "llama3:instruct"

_SUMMARY_PROMPT = """
You are a news summarizer. Produce a rich but compact, structured summary.

Return ONLY the following sections:

TITLE: <infer a short, neutral title from the article>
TL;DR: <2-3 sentence neutral abstract>

KEY POINTS:
- <5-10 concise bullets capturing facts, actors, numbers, dates, locations>
- <avoid opinions; stick to what the article states>

CONTEXT & BACKGROUND:
- <2-4 bullets on prior events, timelines, comparisons if present in the article>

NUMBERS & QUOTES:
- <list any dollar amounts, stats, dates, and up to 2 short quotes with speakers>

FRAMING/LANGUAGE NOTES:
- <list 2-5 phrases with charged tone or framing (if any); else 'None'>

ARTICLE TEXT:
{article_text}
""".strip()

_SYNTHESIS_PROMPT = """
You are a cross-article synthesizer.

Given several article summaries below, produce:
1) A neutral, cohesive synthesis (6-10 bullet points).
2) 3-5 themes or trends that emerge.
3) Notable disagreements or uncertainties (if any).
4) A short overall TL;DR (2-3 sentences).

Return ONLY those sections in a clean, readable format.

SUMMARIES:
{summaries}
""".strip()


def _ollama_call(prompt: str, model: str = MODEL, num_ctx: int = 8192, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def summarize_article(article_text: str) -> str:
    prompt = _SUMMARY_PROMPT.format(article_text=article_text)
    return _ollama_call(prompt)


def summarize_articles_bulk(texts: List[str]) -> Tuple[List[str], str]:
    """Return (per_article_summaries, overall_synthesis)."""
    per_summaries: List[str] = []
    for t in texts:
        per_summaries.append(summarize_article(t))

    # overall synthesis
    joined = "\n\n---\n\n".join(per_summaries)
    overall = _ollama_call(_SYNTHESIS_PROMPT.format(summaries=joined))
    return per_summaries, overall
