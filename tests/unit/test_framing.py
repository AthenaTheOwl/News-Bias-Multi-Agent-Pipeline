from __future__ import annotations

from core.framing import (
    article_frames,
    framing_table,
    source_context,
    source_context_summary,
    source_diversity,
    takeaways,
    watch_items,
)
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
    assert "source_context" in rows[0]


def test_source_context_identifies_common_policy_sources() -> None:
    from core.schemas import Article

    manhattan = Article(
        id="mi",
        title="Housing policy paper",
        url="fixture://paper",
        source="Manhattan Institute",
        text="Housing policy paper from Manhattan Institute",
    )
    cap = Article(
        id="cap",
        title="Energy policy memo",
        url="fixture://paper",
        source="Center for American Progress",
        text="Energy policy memo from Center for American Progress",
    )
    reuters = Article(
        id="wire",
        title="Wire story",
        url="fixture://paper",
        source="Reuters",
        text="Wire story",
    )
    assert source_context(manhattan) == ("Conservative / right-leaning policy institute", "right")
    assert source_context(cap) == ("Progressive / left-leaning policy institute", "left")
    assert source_context(reuters) == ("Wire service / reference news source", "reference")


def test_source_diversity_warns_on_single_source() -> None:
    trace = get_runner("static")("climate", fixture_articles=[LEFT_ARTICLE])
    diversity = source_diversity(trace)
    assert diversity["rating"] == "thin"


def test_source_context_summary_reports_uncataloged_sources() -> None:
    trace = get_runner("static")("policy comparison", fixture_articles=MIXED_ARTICLES)
    summary = source_context_summary(trace)
    assert summary["uncataloged"] == 2
    assert "seed catalog" in str(summary["summary"])


def test_takeaways_and_watch_items_are_reader_facing() -> None:
    trace = get_runner("static")("policy comparison", fixture_articles=MIXED_ARTICLES)
    assert takeaways(trace)
    items = watch_items(trace)
    assert any("source rating" in item for item in items)
