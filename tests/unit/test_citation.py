from __future__ import annotations

import pytest

from core.citation import CitationError, raise_on_citation_errors, verify_citation
from core.schemas import Citation
from impls.static.pipeline import run
from tests.fixtures import LEFT_ARTICLE


def test_verify_citation_accepts_verbatim_span() -> None:
    span = "public investment in workers, clean energy, and environmental justice."
    assert verify_citation(Citation(article_id=LEFT_ARTICLE.id, span_text=span), [LEFT_ARTICLE]) is None


def test_verify_citation_rejects_missing_span() -> None:
    error = verify_citation(Citation(article_id=LEFT_ARTICLE.id, span_text="invented phrase"), [LEFT_ARTICLE])
    assert error is not None
    assert "span not found" in error


def test_pipeline_citations_raise_when_corrupted() -> None:
    trace = run("climate bill", fixture_articles=[LEFT_ARTICLE])
    trace.bias_judgment.evidence[0].span_text = "invented phrase"
    with pytest.raises(CitationError):
        raise_on_citation_errors(trace)
