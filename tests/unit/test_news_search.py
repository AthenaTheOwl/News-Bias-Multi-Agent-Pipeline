from __future__ import annotations

from core.news_search import clean_feed_text, fallback_rss, hits_to_articles
from core.schemas import StructuredQuery


def test_hits_to_articles_uses_metadata_when_text_missing() -> None:
    hits = [{"title": "One", "url": "https://example.com/one", "source": "Example", "description": "A useful fallback body."}]
    articles = hits_to_articles(hits, [""])
    assert articles[0].title == "One"
    assert articles[0].text == "A useful fallback body."


def test_clean_feed_text_strips_html() -> None:
    assert clean_feed_text('<a href="https://example.com">Climate bill</a>&nbsp;<font>Source</font>') == "Climate bill Source"


def test_fallback_rss_is_patchable(monkeypatch) -> None:
    class Feed:
        feed = type("FeedMeta", (), {"title": "Fixture RSS"})()
        entries = [type("Entry", (), {"title": "Fixture story", "link": "https://example.com/story", "summary": "fixture summary"})()]

    monkeypatch.setattr("core.news_search.feedparser.parse", lambda _url: Feed())
    hits = fallback_rss(max_articles=1)
    assert hits == [
        {
            "title": "Fixture story",
            "url": "https://example.com/story",
            "source": "Fixture RSS",
            "description": "fixture summary",
        }
    ]


def test_fallback_rss_scores_query_terms(monkeypatch) -> None:
    class SearchFeed:
        feed = type("FeedMeta", (), {"title": "Search"})()
        entries = [
            type("Entry", (), {"title": "Climate bill advances", "link": "https://example.com/climate", "summary": "New climate policy."})(),
            type("Entry", (), {"title": "Sports roundup", "link": "https://example.com/sports", "summary": "Transfer news."})(),
        ]

    class EmptyFeed:
        feed = type("FeedMeta", (), {"title": "Empty"})()
        entries: list[object] = []

    monkeypatch.setattr("core.news_search.feedparser.parse", lambda url: SearchFeed() if "news.google.com" in url else EmptyFeed())
    hits = fallback_rss(StructuredQuery(query="climate bill"), max_articles=5)
    assert [hit["title"] for hit in hits] == ["Climate bill advances"]


def test_structured_query_shape() -> None:
    query = StructuredQuery(query="AI regulation", date_from="2026-06-01")
    assert query.query == "AI regulation"
