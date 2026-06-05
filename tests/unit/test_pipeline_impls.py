from __future__ import annotations

from core.citation import verify_citations
from core.pipeline import _first_sentence
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
