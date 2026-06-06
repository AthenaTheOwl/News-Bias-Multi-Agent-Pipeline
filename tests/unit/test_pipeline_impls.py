from __future__ import annotations

from core.citation import verify_citations
from core.pipeline import _first_sentence
from core.schemas import Article
from impls.registry import IMPLEMENTATIONS, get_runner
from tests.fixtures import LEFT_ARTICLE, MIXED_ARTICLES, NEUTRAL_ARTICLE, RIGHT_ARTICLE


def test_all_impls_return_same_trace_shape() -> None:
    for name in IMPLEMENTATIONS:
        trace = get_runner(name)("climate bill", fixture_articles=[LEFT_ARTICLE], max_articles=1)
        assert trace.implementation == name
        assert [stage.name for stage in trace.stages] == [
            "preprocess",
            "search_fetch",
            "summarize",
            "bias_detect",
            "critique",
            "reconcile",
        ]
        assert not verify_citations(trace)


def test_heuristic_labels_basic_cases() -> None:
    assert get_runner("static")("climate", fixture_articles=[LEFT_ARTICLE]).report.final_label == "Lean Left"
    assert get_runner("static")("border", fixture_articles=[RIGHT_ARTICLE]).report.final_label == "Lean Right"
    assert get_runner("static")("football", fixture_articles=[NEUTRAL_ARTICLE]).report.final_label == "Center"
    assert get_runner("static")("policy comparison", fixture_articles=MIXED_ARTICLES).report.final_label == "Mixed"


def test_empty_article_set_is_honest_and_citation_clean() -> None:
    trace = get_runner("static")("obscure query", fixture_articles=[])
    assert trace.report.final_label == "Undetermined"
    assert trace.report.article_count == 0
    assert not verify_citations(trace)


def test_first_sentence_uses_fallback_for_clipped_feed_metadata() -> None:
    text = "GUEST COLUMN: Climate Advocates Meet with Sen. Mike Barrett to Discuss Energy Affordability Bill"
    assert _first_sentence(text, text) == text


def test_metadata_only_without_direction_is_undetermined_not_center() -> None:
    article = Article(
        id="meta-1",
        title="AI Regulation Forum Opens",
        url="fixture://ai-regulation-forum",
        source="Fixture",
        text="AI Regulation Forum Opens Fixture",
    )
    trace = get_runner("static")("AI regulation", fixture_articles=[article])
    assert trace.report.final_label == "Undetermined"
    assert trace.report.confidence == 0.35


def test_cataloged_conservative_source_context_contributes_right_signal() -> None:
    article = Article(
        id="mi-1",
        title="Manhattan Institute argues city policy raises housing costs",
        url="fixture://manhattan-institute/housing",
        source="Manhattan Institute",
        text="Manhattan Institute argues city policy raises housing costs and limits free market housing supply.",
    )
    trace = get_runner("static")("Manhattan Institute housing", fixture_articles=[article])
    assert trace.report.final_label == "Lean Right"
    assert any("Conservative" in feature for feature in trace.bias_judgment.proxy_features)
