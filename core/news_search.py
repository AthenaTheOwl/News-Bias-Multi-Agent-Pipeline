from __future__ import annotations

import hashlib
import html
import re
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests

from core.schemas import Article, StructuredQuery


FALLBACK_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.npr.org/1004/rss.xml",
    "https://rss.cnn.com/rss/edition_world.rss",
]

STOPWORDS = {
    "about",
    "after",
    "before",
    "last",
    "news",
    "over",
    "past",
    "this",
    "today",
    "week",
    "year",
}


def article_id_for(value: str) -> str:
    return "a-" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def clean_feed_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


def search_articles(
    structured_query: StructuredQuery,
    *,
    max_articles: int = 5,
    gnews_token: str | None = None,
    timeout: int = 20,
) -> list[dict[str, str]]:
    """Return lightweight article hits from GNews or RSS fallback.

    GNews is primary when a key is supplied. If no key is supplied, or if
    the request fails, RSS fallback keeps the demo usable without secrets.
    """

    if gnews_token:
        params: dict[str, Any] = {
            "q": structured_query.query,
            "lang": "en",
            "max": max_articles,
            "token": gnews_token,
        }
        if structured_query.date_from:
            params["from"] = structured_query.date_from
        if structured_query.date_to:
            params["to"] = structured_query.date_to
        try:
            response = requests.get("https://gnews.io/api/v4/search", params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            hits = []
            for item in data.get("articles", []):
                url = item.get("url") or ""
                if not url:
                    continue
                hits.append(
                    {
                        "title": item.get("title") or item.get("description") or "Untitled",
                        "url": url,
                        "source": (item.get("source") or {}).get("name") or "GNews",
                        "description": item.get("description") or "",
                    }
                )
            if hits:
                return hits[:max_articles]
        except Exception:
            pass

    return fallback_rss(structured_query, max_articles=max_articles)


def _query_terms(query: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) >= 3 and token not in STOPWORDS
    ]


def _hit_from_entry(entry: Any, default_source: str) -> dict[str, str] | None:
    link = getattr(entry, "link", "")
    if not link:
        return None
    title = getattr(entry, "title", "Untitled")
    source = default_source
    entry_source = getattr(entry, "source", None)
    if entry_source is not None:
        source = getattr(entry_source, "title", source)
    return {
        "title": clean_feed_text(title),
        "url": link,
        "source": source,
        "description": clean_feed_text(getattr(entry, "summary", "")),
    }


def _score_hit(hit: dict[str, str], terms: list[str]) -> int:
    if not terms:
        return 1
    haystack = f"{hit.get('title', '')} {hit.get('description', '')}".lower()
    return sum(1 for term in terms if term in haystack)


def _google_news_query(structured_query: StructuredQuery) -> str:
    pieces = [structured_query.query]
    if structured_query.date_from:
        pieces.append(f"after:{structured_query.date_from}")
    if structured_query.date_to:
        pieces.append(f"before:{structured_query.date_to}")
    return " ".join(pieces)


def fallback_rss(structured_query: StructuredQuery | None = None, max_articles: int = 5) -> list[dict[str, str]]:
    terms = _query_terms(structured_query.query if structured_query else "")
    hits: list[dict[str, str]] = []

    if structured_query is not None:
        search_url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(_google_news_query(structured_query))
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        parsed = feedparser.parse(search_url)
        for entry in parsed.entries:
            hit = _hit_from_entry(entry, "Google News search")
            if hit and _score_hit(hit, terms) > 0:
                hits.append(hit)
            if len(hits) >= max_articles:
                return hits

    for feed_url in FALLBACK_FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[: max(1, max_articles * 2)]:
            hit = _hit_from_entry(entry, getattr(parsed.feed, "title", "RSS"))
            if hit and _score_hit(hit, terms) > 0:
                hits.append(hit)
            if len(hits) >= max_articles:
                return hits
    return hits[:max_articles]


def hits_to_articles(hits: list[dict[str, str]], texts: list[str]) -> list[Article]:
    articles: list[Article] = []
    for index, hit in enumerate(hits):
        title = hit.get("title") or f"Article {index + 1}"
        url = hit.get("url") or f"fixture://article-{index + 1}"
        text = texts[index] if index < len(texts) and texts[index].strip() else hit.get("description") or title
        articles.append(
            Article(
                id=article_id_for(url or title),
                title=clean_feed_text(title),
                url=url,
                source=hit.get("source") or "unknown",
                text=clean_feed_text(text),
            )
        )
    return articles
