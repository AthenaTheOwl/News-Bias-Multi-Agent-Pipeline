from __future__ import annotations

from core.framing import article_frames, framing_table, takeaways, watch_items
from impls.registry import get_runner
from tests.fixtures import LEFT_ARTICLE, MIXED_ARTICLES


def test_article_frames_names_visible_frames() -> None:
    frames = article_frames(LEFT_ARTICLE)
    assert "Public investment" in frames


def test_framing_table_maps_sources_to_frames() -> None:
    trace = get_runner("static")("policy comparison", fixture_articles=MIXED_ARTICLES)
    rows = framing_table(trace)
    assert len(rows) == 2
    assert all(row["frames"] for row in rows)


def test_takeaways_and_watch_items_are_reader_facing() -> None:
    trace = get_runner("static")("policy comparison", fixture_articles=MIXED_ARTICLES)
    assert takeaways(trace)
    items = watch_items(trace)
    assert any("source rating" in item for item in items)
