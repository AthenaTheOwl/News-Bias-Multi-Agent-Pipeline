# retrieval/rss_fetch.py

import feedparser
import requests
import json
from typing import List, Dict, Union

def fetch_rss_articles(feed_url, max_articles=10):
    """
    Fetches articles from a standard RSS feed.
    Returns a list of dicts with title and link.
    """
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "link": entry.link
        })
    return articles

def _fallback_rss(max_articles: int) -> List[Dict[str, str]]:
    fallback_feeds = [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.cnn.com/rss/edition_world.rss",
        "https://feeds.npr.org/1004/rss.xml"
    ]
    out = []
    for url in fallback_feeds:
        out.extend(fetch_rss_articles(url, max_articles=2))
    return out[:max_articles]

def _normalize_structured_query(q: Union[str, dict]) -> dict:
    """
    Accepts either JSON string or dict:
    {
      "query": "Kristie Mewis",
      "from": "YYYY-MM-DD",  # optional
      "to": "YYYY-MM-DD"     # optional
    }
    """
    if isinstance(q, dict):
        return q
    try:
        return json.loads(q)
    except Exception:
        # last resort: treat as plain keywords
        return {"query": str(q).strip()}

def search_articles(structured_query: Union[str, dict], max_articles=10) -> List[Dict[str, str]]:
    """
    Search for news articles via GNews API using a structured query.
    structured_query can be a dict or a JSON string with keys: query, from, to.
    Returns a list of dicts with title and link. Falls back to RSS on failure.
    """
    sq = _normalize_structured_query(structured_query)
    query = sq.get("query", "").strip()
    if not query:
        return _fallback_rss(max_articles)

    api_key = "475c36139f626eb21eff1c0b6ca50605"  # <-- replace with your key
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": max_articles,
        "token": api_key,
    }
    # Optional date filters if provided
    if sq.get("from"):
        params["from"] = sq["from"]
    if sq.get("to"):
        params["to"] = sq["to"]

    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
        articles = []
        for a in data.get("articles", []):
            title = a.get("title") or a.get("description") or a.get("content") or "Untitled"
            link = a.get("url") or ""
            if link:
                articles.append({"title": title, "link": link})
        if not articles:
            raise ValueError("No articles found in GNews response.")
        return articles
    except Exception as e:
        print(f"Error searching articles: {e}")
        print("📡 Falling back to RSS feeds...")
        return _fallback_rss(max_articles)
