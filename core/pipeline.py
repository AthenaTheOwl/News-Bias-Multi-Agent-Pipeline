from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from core.citation import verify_citations
from core.framing import source_context
from core.llm_provider import LLMClient, extract_json_object, get_llm
from core.news_search import hits_to_articles, search_articles
from core.prompts import load_prompt
from core.schemas import (
    Article,
    Citation,
    LLMKeys,
    PipelineTrace,
    ReconciledReport,
    StageRecord,
    StructuredBiasJudgment,
    StructuredCritique,
    StructuredQuery,
    StructuredSummary,
)
from core.text_extraction import extract_article_text


LEFT_TERMS = [
    "climate",
    "equity",
    "workers",
    "union",
    "public investment",
    "reproductive",
    "discrimination",
    "corporate greed",
    "voting rights",
    "environmental justice",
]

RIGHT_TERMS = [
    "border",
    "taxpayer",
    "government overreach",
    "freedom",
    "law and order",
    "parental rights",
    "illegal immigration",
    "religious liberty",
    "small business",
    "woke",
    "school choice",
    "free market",
    "deregulation",
    "progressive policies",
]

CHARGED_TERMS = [
    "reckless",
    "radical",
    "slammed",
    "disaster",
    "betrayal",
    "elite",
    "crackdown",
    "boondoggle",
    "attack",
    "scheme",
]

NON_POLITICAL_TERMS = ["football", "soccer", "earnings", "match", "tournament", "weather", "product launch"]


def _metadata_only(articles: list[Article]) -> bool:
    if not articles:
        return False
    thin_count = sum(1 for article in articles if len(article.text) < 240 and article.text.count(".") <= 1)
    return thin_count / len(articles) >= 0.6


def preprocess_subject(subject: str, today: date | None = None) -> StructuredQuery:
    today = today or date.today()
    cleaned = " ".join(subject.split()).strip()
    lowered = cleaned.lower()
    query = re.sub(r"\b(today|last week|last year)\b", "", cleaned, flags=re.I).strip(" ,-")

    if "today" in lowered:
        return StructuredQuery(query=query or cleaned, date_from=today.isoformat(), date_to=today.isoformat())
    if "last week" in lowered:
        start = today - timedelta(days=7)
        return StructuredQuery(query=query or cleaned, date_from=start.isoformat(), date_to=today.isoformat())
    if "last year" in lowered:
        year = today.year - 1
        return StructuredQuery(query=query or cleaned, date_from=f"{year}-01-01", date_to=f"{year}-12-31")
    return StructuredQuery(query=query or cleaned)


def _first_sentence(text: str, fallback: str) -> str:
    match = re.search(r"(.{40,220}?[.!?])(?:\s|$)", text.strip())
    if not match:
        return fallback
    sentence = match.group(1).strip()
    if len(sentence) < 80 and len(text.strip()) > len(sentence) + 20:
        return fallback
    return sentence


def _choose_span(article: Article, terms: list[str] | None = None) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", article.text.strip())
    if terms:
        for sentence in sentences:
            low = sentence.lower()
            if any(term in low for term in terms) and len(sentence) >= 8:
                return sentence.strip()
    for sentence in sentences:
        if len(sentence) >= 40:
            return sentence.strip()
    return article.text[:160].strip()


def _citation(article: Article, terms: list[str] | None = None) -> Citation:
    return Citation(article_id=article.id, span_text=_choose_span(article, terms))


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def fetch_articles(
    query: StructuredQuery,
    *,
    keys: LLMKeys,
    max_articles: int,
    fixture_articles: list[Article] | None = None,
) -> tuple[list[Article], list[str]]:
    if fixture_articles is not None:
        return fixture_articles[:max_articles], ["fixture articles supplied"]

    hits = search_articles(query, max_articles=max_articles, gnews_token=keys.gnews_token)
    if not hits:
        return [], [f"no matching articles found for query {query.query!r}"]
    texts: list[str] = []
    notes: list[str] = []
    for hit in hits:
        try:
            text = extract_article_text(hit["url"])
            if len(text) < 120:
                text = hit.get("description") or hit["title"]
                notes.append(f"used metadata text for {hit['url']}")
        except Exception as exc:
            text = hit.get("description") or hit["title"]
            notes.append(f"fetch fallback for {hit['url']}: {exc.__class__.__name__}")
        texts.append(text)
    return hits_to_articles(hits, texts), notes


def summarize(articles: list[Article], llm: LLMClient) -> StructuredSummary:
    if not articles:
        return StructuredSummary(
            headline="No articles found",
            neutral_summary="No article text was available for this query.",
            key_points=[],
            framing_notes=[],
        )

    if llm.provider != "heuristic":
        prompt = (
            load_prompt("summarize")
            + "\n\nARTICLES:\n"
            + "\n\n".join(f"[{a.id}] {a.title}\n{a.text[:2400]}" for a in articles)
        )
        parsed = extract_json_object(llm.generate(prompt))
        if parsed:
            try:
                spans = []
                for item in parsed.get("framing_spans", [])[:4]:
                    if isinstance(item, dict):
                        spans.append(Citation(article_id=item["article_id"], span_text=item["span_text"]))
                    elif articles:
                        spans.append(_citation(articles[0]))
                return StructuredSummary(
                    headline=str(parsed.get("headline") or articles[0].title),
                    neutral_summary=str(parsed.get("neutral_summary") or _first_sentence(articles[0].text, articles[0].title)),
                    key_points=[str(item) for item in parsed.get("key_points", [])][:8],
                    framing_notes=spans or [_citation(articles[0])],
                )
            except Exception:
                pass

    key_points = [_first_sentence(article.text, article.title) for article in articles[:5]]
    framing_notes = [
        _citation(article, CHARGED_TERMS)
        for article in articles[:3]
        if any(term in article.text.lower() for term in CHARGED_TERMS)
    ]
    if not framing_notes:
        framing_notes = [_citation(articles[0])]
    return StructuredSummary(
        headline=f"News-bias pass on {len(articles)} article(s)",
        neutral_summary=" ".join(key_points[:2]),
        key_points=key_points,
        framing_notes=framing_notes,
    )


def detect_bias(summary: StructuredSummary, articles: list[Article], llm: LLMClient) -> StructuredBiasJudgment:
    if not articles:
        return StructuredBiasJudgment(
            label="Undetermined",
            confidence=0.0,
            rationale="No article text was available for this query.",
            evidence=[],
            proxy_features=[],
        )

    combined = "\n".join(article.text.lower() for article in articles)
    left_hits = [term for term in LEFT_TERMS if term in combined]
    right_hits = [term for term in RIGHT_TERMS if term in combined]
    charged_hits = [term for term in CHARGED_TERMS if term in combined]
    non_political = any(term in combined for term in NON_POLITICAL_TERMS)
    source_contexts = [source_context(article) for article in articles]
    left_source_hits = [context[0] for context in source_contexts if context and context[1] == "left"]
    right_source_hits = [context[0] for context in source_contexts if context and context[1] == "right"]

    if llm.provider != "heuristic":
        prompt = (
            load_prompt("bias_detect")
            + "\n\nSUMMARY:\n"
            + summary.model_dump_json(indent=2)
            + "\n\nARTICLES:\n"
            + "\n\n".join(f"[{a.id}] {a.text[:2400]}" for a in articles)
        )
        parsed = extract_json_object(llm.generate(prompt))
        if parsed:
            try:
                evidence = [
                    Citation(article_id=item["article_id"], span_text=item["span_text"])
                    for item in parsed.get("evidence", [])
                    if isinstance(item, dict)
                ]
                return StructuredBiasJudgment(
                    label=parsed.get("label", "Undetermined"),
                    confidence=_safe_float(parsed.get("confidence"), 0.5),
                    rationale=str(parsed.get("rationale") or "Model returned a structured bias judgment."),
                    evidence=evidence or [_citation(articles[0] if articles else Article(id="none", title="none", url="none", text="No text"))],
                    proxy_features=[str(item) for item in parsed.get("proxy_features", [])],
                )
            except Exception:
                pass

    if non_political and not left_hits and not right_hits and not left_source_hits and not right_source_hits:
        label = "Center"
        confidence = 0.72
        rationale = "The article set appears non-political or lacks ideological policy framing."
    elif (left_hits or left_source_hits) and (right_hits or right_source_hits):
        label = "Mixed"
        confidence = 0.64
        rationale = "The article set contains cues associated with both left-leaning and right-leaning frames."
    elif left_hits or left_source_hits:
        label = "Lean Left"
        confidence = min(0.88, 0.55 + 0.08 * len(left_hits) + 0.06 * len(left_source_hits))
        if left_hits:
            rationale = "The article set includes policy language often associated with left-leaning framing."
        else:
            rationale = "The directional signal comes from cataloged source context, not article-language evidence alone."
    elif right_hits or right_source_hits:
        label = "Lean Right"
        confidence = min(0.88, 0.55 + 0.08 * len(right_hits) + 0.06 * len(right_source_hits))
        if right_hits:
            rationale = "The article set includes policy language often associated with right-leaning framing."
        else:
            rationale = "The directional signal comes from cataloged source context, not article-language evidence alone."
    elif charged_hits:
        label = "Undetermined"
        confidence = 0.45
        rationale = "The article set contains charged language, but the direction of political framing is unclear."
    elif _metadata_only(articles):
        label = "Undetermined"
        confidence = 0.35
        rationale = "The run mostly has headline or metadata text, so the app should not infer a center frame."
    else:
        label = "Center"
        confidence = 0.58
        rationale = "The article set contains limited ideological framing in the inspected text."

    evidence_terms = left_hits + right_hits + charged_hits
    evidence_articles = articles[:2] or [Article(id="none", title="none", url="none", text="No article text available.")]
    evidence = [_citation(article, evidence_terms or None) for article in evidence_articles]
    return StructuredBiasJudgment(
        label=label,
        confidence=confidence,
        rationale=rationale,
        evidence=evidence,
        proxy_features=left_hits + right_hits + charged_hits + left_source_hits + right_source_hits,
    )


def critique(summary: StructuredSummary, judgment: StructuredBiasJudgment, articles: list[Article], llm: LLMClient) -> StructuredCritique:
    if llm.provider != "heuristic":
        prompt = (
            load_prompt("critique")
            + "\n\nSUMMARY:\n"
            + summary.model_dump_json(indent=2)
            + "\n\nDETECTOR_OUTPUT:\n"
            + judgment.model_dump_json(indent=2)
            + "\n\nARTICLES:\n"
            + "\n\n".join(f"[{a.id}] {a.text[:2200]}" for a in articles)
        )
        parsed = extract_json_object(llm.generate(prompt))
        if parsed:
            try:
                phrases = [
                    Citation(article_id=item["article_id"], span_text=item["span_text"])
                    for item in parsed.get("trigger_phrases", [])
                    if isinstance(item, dict)
                ]
                return StructuredCritique(
                    refined_label=parsed.get("refined_label", judgment.label),
                    agree_with_detector=bool(parsed.get("agree_with_detector", True)),
                    reasoning=str(parsed.get("reasoning") or "Model accepted the detector output."),
                    trigger_phrases=phrases or judgment.evidence[:2],
                    proxy_notes=str(parsed.get("proxy_notes") or "No proxy caveat supplied."),
                )
            except Exception:
                pass

    if judgment.label in {"Lean Left", "Lean Right", "Mixed"} and judgment.confidence < 0.62:
        refined = "Undetermined"
        agree = False
        reasoning = "The detector saw framing cues, but the evidence is too thin for a confident political label."
    else:
        refined = judgment.label
        agree = True
        reasoning = "The detector label is consistent with the cited spans and the inspected article set."
    proxy_notes = (
        "Heuristic features are directional clues, not proof. Treat this as a traceable classroom signal, not a rating."
    )
    return StructuredCritique(
        refined_label=refined,
        agree_with_detector=agree,
        reasoning=reasoning,
        trigger_phrases=judgment.evidence[:3],
        proxy_notes=proxy_notes,
    )


def reconcile(summary: StructuredSummary, judgment: StructuredBiasJudgment, critique_result: StructuredCritique, articles: list[Article], llm: LLMClient) -> ReconciledReport:
    if llm.provider != "heuristic":
        prompt = (
            load_prompt("reconcile")
            + "\n\nSUMMARY:\n"
            + summary.model_dump_json(indent=2)
            + "\n\nBIAS:\n"
            + judgment.model_dump_json(indent=2)
            + "\n\nCRITIQUE:\n"
            + critique_result.model_dump_json(indent=2)
        )
        parsed = extract_json_object(llm.generate(prompt))
        if parsed:
            try:
                return ReconciledReport(
                    headline=str(parsed.get("headline") or summary.headline),
                    executive_summary=str(parsed.get("executive_summary") or summary.neutral_summary),
                    final_label=parsed.get("final_label", critique_result.refined_label),
                    confidence=_safe_float(parsed.get("confidence"), min(judgment.confidence, 0.7)),
                    caveats=[str(item) for item in parsed.get("caveats", [])][:6]
                    or ["Model-generated report; review citations before relying on it."],
                    article_count=len(articles),
                )
            except Exception:
                pass
    confidence = min(judgment.confidence, 0.82 if critique_result.agree_with_detector else 0.55)
    return ReconciledReport(
        headline=summary.headline,
        executive_summary=summary.neutral_summary,
        final_label=critique_result.refined_label,
        confidence=confidence,
        caveats=[
            "This is a learning demo, not a production bias detector.",
            "Labels are article-set judgments, not source ratings.",
            "Every cited phrase must appear verbatim in the fetched article text.",
        ],
        article_count=len(articles),
    )


def _stage(name: str, implementation: str, output: dict[str, Any], notes: list[str] | None = None) -> StageRecord:
    return StageRecord(name=name, implementation=implementation, output=output, notes=notes or [])


def run_pipeline(
    subject: str,
    *,
    implementation: str,
    provider: str = "heuristic",
    model: str | None = None,
    keys: LLMKeys | None = None,
    max_articles: int = 5,
    fixture_articles: list[Article] | None = None,
    framework_notes: list[str] | None = None,
) -> PipelineTrace:
    keys = keys or LLMKeys()
    llm = get_llm(provider, model, keys)
    query = preprocess_subject(subject)
    stages = [_stage("preprocess", implementation, query.model_dump())]

    articles, fetch_notes = fetch_articles(
        query,
        keys=keys,
        max_articles=max_articles,
        fixture_articles=fixture_articles,
    )
    stages.append(
        _stage(
            "search_fetch",
            implementation,
            {"article_count": len(articles), "article_ids": [a.id for a in articles]},
            fetch_notes,
        )
    )

    summary = summarize(articles, llm)
    stages.append(_stage("summarize", implementation, summary.model_dump()))

    judgment = detect_bias(summary, articles, llm)
    stages.append(_stage("bias_detect", implementation, judgment.model_dump()))

    critique_result = critique(summary, judgment, articles, llm)
    stages.append(_stage("critique", implementation, critique_result.model_dump()))

    report = reconcile(summary, judgment, critique_result, articles, llm)
    stages.append(_stage("reconcile", implementation, report.model_dump()))

    trace = PipelineTrace(
        subject=subject,
        implementation=implementation,
        provider=llm.provider,
        model=llm.model,
        structured_query=query,
        articles=articles,
        summary=summary,
        bias_judgment=judgment,
        critique=critique_result,
        report=report,
        stages=stages,
        framework_notes=framework_notes or [],
    )
    trace.citation_errors = verify_citations(trace)
    if trace.citation_errors:
        raise ValueError("Citation verifier failed: " + "; ".join(trace.citation_errors))
    return trace
