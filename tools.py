# tools.py — LangChain-compatible tools

from typing import Union, List
from langchain_core.tools import tool

@tool
def preprocessor_tool(user_prompt: str) -> str:
    """Convert vague user prompt into JSON: {"query": "...", "from": "YYYY-MM-DD", "to": "YYYY-MM-DD"} (dates optional)."""
    from agents.preprocessor import run_preprocessor_agent_str
    return run_preprocessor_agent_str(user_prompt)

@tool
def search_tool(structured_query_json: str) -> str:
    """Search news using a structured JSON string from the preprocessor. Returns a numbered list of Title - URL."""
    from retrieval.rss_fetch import search_articles
    articles = search_articles(structured_query_json)
    if not articles:
        return "No results."
    lines = [f"{i+1}. {a['title']} - {a['link']}" for i, a in enumerate(articles)]
    return "\n".join(lines)

@tool
def summarizer_tool(article_text: Union[str, List[str]]) -> str:
    """Summarize ONE article OR a LIST of articles.
    If a list is provided, returns a cross-article synthesis + per-article summaries."""
    from agents.summarizer import summarize_article, summarize_articles_bulk

    if isinstance(article_text, list):
        per_sums, overall = summarize_articles_bulk(article_text)
        parts = ["### OVERALL SYNTHESIS", overall, "### PER-ARTICLE SUMMARIES"]
        for i, s in enumerate(per_sums, 1):
            parts.append(f"\n#### Article {i}\n{s}")
        return "\n".join(parts)
    else:
        return summarize_article(article_text)

@tool
def fetch_and_summarize_tool(structured_query_json: str, max_articles: int = 5) -> str:
    """End-to-end: search → fetch article bodies → summarize multiple → cross-article synthesis."""
    from retrieval.rss_fetch import search_articles
    from retrieval.text_extraction import extract_article_text
    from agents.summarizer import summarize_articles_bulk

    # 1) search
    hits = search_articles(structured_query_json, max_articles=max_articles)
    if not hits:
        return "No articles found."

    # 2) fetch bodies
    texts: List[str] = []
    titles: List[str] = []
    for i, a in enumerate(hits, 1):
        url = a.get("link", "")
        title = a.get("title", f"Article {i}")
        titles.append(title)
        try:
            body = extract_article_text(url)
            if body and len(body.strip()) > 200:
                texts.append(body)
        except Exception as e:
            texts.append(f"[Error fetching {url}: {e}]")

    if not texts:
        return "Found links but failed to fetch article bodies."

    # 3) summarize bulk
    per_sums, overall = summarize_articles_bulk(texts)

    # 4) stitch with titles so you know which summary maps to which link
    out = ["### OVERALL SYNTHESIS", overall, "### PER-ARTICLE SUMMARIES"]
    for i, (title, s) in enumerate(zip(titles, per_sums), 1):
        out.append(f"\n#### {i}. {title}\n{s}")

    return "\n".join(out)

@tool
def critic_tool(summary: str, full_text: str, bias_score: float, flagged: bool) -> str:
    """Critically assess the summary for bias, using the full article and bias signal."""
    from agents.critic import critic_analysis
    return critic_analysis(summary, full_text, bias_score, flagged)

@tool
def writer_tool(summary: str, critique: str, bias_score: float, flagged: bool) -> str:
    """Write a professional final article using the summary, critique, and bias info."""
    from agents.writer import write_final_report
    return write_final_report(summary, critique, bias_score, flagged)
