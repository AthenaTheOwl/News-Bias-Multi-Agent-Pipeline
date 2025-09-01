# langgraph/news_graph.py
from __future__ import annotations

import json
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

# local modules
from agents.preprocessor import run_preprocessor_agent_str
from retrieval.rss_fetch import search_articles
from retrieval.text_extraction import extract_article_text
from agents.summarizer import summarize_articles_bulk
from agents.writer import write_final_report
# (optional) critic
try:
    from agents.critic import critic_analysis
    HAS_CRITIC = True
except Exception:
    HAS_CRITIC = False


# ---------- Graph State ----------
class NewsState(TypedDict, total=False):
    user_prompt: str

    # preprocessor
    structured_query: dict          # {"query": "...", "from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}

    # search
    hits: List[Dict[str, str]]      # [{"title": "...", "link": "..."}]

    # fetch
    article_texts: List[str]
    titles: List[str]

    # summarize
    per_summaries: List[str]
    overall_summary: str

    # critic & writer
    critique: str
    final_report: str

    # control / config
    max_articles: int
    do_critic: bool

    # error
    error: str


# ---------- Helpers ----------
def _safe_parse_structured_query(text: str | dict) -> dict:
    """Accepts a JSON string from the preprocessor or a dict; normalizes safely."""
    if isinstance(text, dict):
        return text
    # strip any accidental <think> content; keep last {...} block if present
    s = text.strip()
    # try to find first '{'
    i = s.find("{")
    if i >= 0:
        s = s[i:]
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return {"query": str(text).strip()}
        return obj
    except Exception:
        # If model returned plain text like: "Search for news articles about X ..."
        return {"query": str(text).strip()}


# ---------- Nodes ----------
def preprocess_node(state: NewsState) -> NewsState:
    prompt = state["user_prompt"]
    structured = run_preprocessor_agent_str(prompt)
    sq = _safe_parse_structured_query(structured)
    return {**state, "structured_query": sq}


def search_node(state: NewsState) -> NewsState:
    sq = state["structured_query"]
    max_articles = state.get("max_articles", 6)
    try:
        hits = search_articles(sq, max_articles=max_articles)
        return {**state, "hits": hits}
    except Exception as e:
        return {**state, "error": f"search failed: {e}", "hits": []}


def fetch_node(state: NewsState) -> NewsState:
    hits = state.get("hits", [])
    texts: List[str] = []
    titles: List[str] = []
    for i, a in enumerate(hits, 1):
        url = a.get("link", "")
        title = a.get("title", f"Article {i}")
        titles.append(title)
        if not url:
            texts.append("")
            continue
        try:
            body = extract_article_text(url)
            texts.append(body or "")
        except Exception as e:
            texts.append(f"[Fetch error for {url}: {e}]")
    return {**state, "article_texts": texts, "titles": titles}


def summarize_node(state: NewsState) -> NewsState:
    texts = [t for t in state.get("article_texts", []) if t and len(t.strip()) > 200]
    if not texts:
        return {**state, "per_summaries": [], "overall_summary": "No fetchable article bodies."}
    per_sums, overall = summarize_articles_bulk(texts)
    return {**state, "per_summaries": per_sums, "overall_summary": overall}


def critic_node(state: NewsState) -> NewsState:
    if not state.get("do_critic", False) or not HAS_CRITIC:
        return state
    # Very light proxy (you can plug your bias detector here)
    summary = state.get("overall_summary", "")
    full_text = "\n\n".join(state.get("article_texts", [])[:2])  # give some context
    critique = critic_analysis(summary, full_text, score=0, flagged=False)
    return {**state, "critique": critique}


def writer_node(state: NewsState) -> NewsState:
    summary = state.get("overall_summary", "")
    critique = state.get("critique", "No critic step executed.")
    final = write_final_report(summary, critique, bias_score=0, flagged=False)
    return {**state, "final_report": final}


# ---------- Graph build ----------
def build_news_graph():
    g = StateGraph(NewsState)
    g.add_node("preprocess", preprocess_node)
    g.add_node("search",     search_node)
    g.add_node("fetch",      fetch_node)
    g.add_node("summarize",  summarize_node)
    g.add_node("critic",     critic_node)
    g.add_node("writer",     writer_node)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "search")
    g.add_edge("search",     "fetch")
    g.add_edge("fetch",      "summarize")
    g.add_edge("summarize",  "critic")
    g.add_edge("critic",     "writer")
    g.add_edge("writer",     END)

    return g.compile()


# ---------- Public API ----------
def run_news_graph(user_prompt: str, max_articles: int = 8, do_critic: bool = True) -> Dict[str, Any]:
    graph = build_news_graph()
    initial: NewsState = {
        "user_prompt": user_prompt,
        "max_articles": max_articles,
        "do_critic": do_critic,
    }
    out = graph.invoke(initial)
    return out
