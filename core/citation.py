from __future__ import annotations

from core.schemas import Article, Citation, PipelineTrace


class CitationError(ValueError):
    pass


def verify_citation(citation: Citation, articles: list[Article]) -> str | None:
    article_by_id = {article.id: article for article in articles}
    article = article_by_id.get(citation.article_id)
    if article is None:
        return f"{citation.article_id}: article not found"
    if citation.span_text not in article.text:
        return f"{citation.article_id}: span not found: {citation.span_text!r}"
    return None


def verify_citations(trace: PipelineTrace) -> list[str]:
    citations: list[Citation] = []
    citations.extend(trace.summary.framing_notes)
    citations.extend(trace.bias_judgment.evidence)
    citations.extend(trace.critique.trigger_phrases)

    errors = [
        error
        for citation in citations
        if (error := verify_citation(citation, trace.articles)) is not None
    ]
    return errors


def raise_on_citation_errors(trace: PipelineTrace) -> None:
    errors = verify_citations(trace)
    if errors:
        raise CitationError("; ".join(errors))
