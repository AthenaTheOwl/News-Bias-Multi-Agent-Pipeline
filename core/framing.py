from __future__ import annotations

import re
from collections import Counter

from core.schemas import Article, PipelineTrace


SOURCE_CONTEXT: dict[str, tuple[str, str]] = {
    # Conservative, right-leaning, or libertarian policy and media sources.
    "manhattan institute": ("Conservative / right-leaning policy institute", "right"),
    "city journal": ("Conservative / right-leaning policy publication", "right"),
    "heritage foundation": ("Conservative / right-leaning policy institute", "right"),
    "american enterprise institute": ("Conservative / right-leaning policy institute", "right"),
    "aei": ("Conservative / right-leaning policy institute", "right"),
    "hoover institution": ("Conservative / right-leaning policy institute", "right"),
    "claremont institute": ("Conservative / right-leaning policy institute", "right"),
    "cato institute": ("Libertarian / right-leaning policy institute", "right"),
    "reason magazine": ("Libertarian / right-leaning publication", "right"),
    "reason.com": ("Libertarian / right-leaning publication", "right"),
    "national review": ("Conservative / right-leaning publication", "right"),
    "the federalist": ("Conservative / right-leaning publication", "right"),
    "daily wire": ("Conservative / right-leaning publication", "right"),
    "washington examiner": ("Conservative / right-leaning publication", "right"),
    "washington times": ("Conservative / right-leaning publication", "right"),
    "new york post": ("Conservative / right-leaning tabloid", "right"),
    "fox news": ("Right-leaning cable/news outlet", "right"),
    "newsmax": ("Conservative / right-leaning cable/news outlet", "right"),
    "breitbart": ("Conservative / right-leaning publication", "right"),
    "townhall": ("Conservative / right-leaning publication", "right"),
    "commentary magazine": ("Conservative / right-leaning publication", "right"),
    "the dispatch": ("Center-right publication", "right"),
    "wall street journal opinion": ("Conservative / right-leaning opinion page", "right"),
    "wsj opinion": ("Conservative / right-leaning opinion page", "right"),
    # Progressive, left-leaning, or labor-aligned policy and media sources.
    "center for american progress": ("Progressive / left-leaning policy institute", "left"),
    "cap action": ("Progressive / left-leaning policy institute", "left"),
    "american progress": ("Progressive / left-leaning policy institute", "left"),
    "economic policy institute": ("Labor-aligned / left-leaning policy institute", "left"),
    "roosevelt institute": ("Progressive / left-leaning policy institute", "left"),
    "brennan center": ("Progressive / left-leaning legal policy institute", "left"),
    "aclu": ("Civil-liberties advocacy group often aligned with progressive policy fights", "left"),
    "mother jones": ("Progressive / left-leaning publication", "left"),
    "jacobin": ("Socialist / left-leaning publication", "left"),
    "the nation": ("Progressive / left-leaning publication", "left"),
    "democracy now": ("Progressive / left-leaning news program", "left"),
    "vox": ("Left-leaning explanatory publication", "left"),
    "msnbc": ("Left-leaning cable/news outlet", "left"),
    "huffpost": ("Left-leaning publication", "left"),
    "slate": ("Left-leaning publication", "left"),
    "the intercept": ("Left-leaning investigative publication", "left"),
    # Reference and general-news contexts: shown to users, never used as a left/right cue.
    "associated press": ("Wire service / reference news source", "reference"),
    "ap news": ("Wire service / reference news source", "reference"),
    "reuters": ("Wire service / reference news source", "reference"),
    "bbc": ("Public broadcaster / general news source", "reference"),
    "pbs": ("Public broadcaster / general news source", "reference"),
    "npr": ("Public radio / general news source", "reference"),
    "axios": ("General news / political newsletter source", "reference"),
    "politico": ("General politics news source", "reference"),
}

FRAME_TERMS = {
    "Public investment": ["public investment", "workers", "union", "clean energy", "environmental justice", "equity"],
    "Cost and taxpayer burden": ["taxpayer", "cost", "price", "burden", "affordability", "mandate"],
    "Law and order": ["border", "law and order", "illegal immigration", "sheriff", "enforcement"],
    "Market and business impact": ["small business", "employers", "industry", "data centers", "fossil fuels"],
    "Institutional process": ["analysts", "committee", "agency", "court", "report", "lawmakers"],
    "Logistics and schedule": ["schedule", "match", "ticket", "venue", "tournament", "travel"],
}


def source_context(article: Article) -> tuple[str, str] | None:
    haystack = f"{article.source} {article.title} {article.url}".lower()
    for name, context in SOURCE_CONTEXT.items():
        if "." in name:
            matched = name in haystack
        else:
            matched = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", haystack) is not None
        if matched:
            return context
    return None


def source_context_summary(trace: PipelineTrace) -> dict[str, object]:
    contexts = [source_context(article) for article in trace.articles]
    cataloged = [context for context in contexts if context is not None]
    posture_counts = Counter(context[1] for context in cataloged)
    if not trace.articles:
        summary = "No sources were available."
    elif not cataloged:
        summary = "No source-context entries matched the seed catalog. Judge the result from article text and spans."
    elif posture_counts["left"] and posture_counts["right"]:
        summary = "Cataloged source context includes both left-leaning and right-leaning sources."
    elif posture_counts["right"]:
        summary = "Cataloged source context tilts right for this article set."
    elif posture_counts["left"]:
        summary = "Cataloged source context tilts left for this article set."
    else:
        summary = "Cataloged sources are reference or general-news contexts, not directional cues."
    return {
        "cataloged": len(cataloged),
        "uncataloged": max(0, len(trace.articles) - len(cataloged)),
        "posture_counts": dict(posture_counts),
        "summary": summary,
    }


def article_frames(article: Article) -> list[str]:
    text = f"{article.title} {article.text}".lower()
    frames = [
        frame
        for frame, terms in FRAME_TERMS.items()
        if any(term in text for term in terms)
    ]
    return frames or ["General reporting"]


def framing_table(trace: PipelineTrace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for article in trace.articles:
        rows.append(
            {
                "source": article.source,
                "article": article.title,
                "frames": ", ".join(article_frames(article)),
                "source_context": (source_context(article) or ("Not cataloged", ""))[0],
            }
        )
    return rows


def source_diversity(trace: PipelineTrace) -> dict[str, object]:
    sources = [article.source.strip() or "unknown" for article in trace.articles]
    if not sources:
        return {
            "unique_sources": 0,
            "article_count": 0,
            "dominant_source_share": 0.0,
            "rating": "no sources",
            "warning": "No sources were available for this run.",
        }
    counts = {source: sources.count(source) for source in sorted(set(sources))}
    dominant = max(counts.values())
    share = dominant / len(sources)
    if len(counts) == 1:
        rating = "thin"
        warning = "All articles came from one source. Treat the framing label as provisional."
    elif share >= 0.67:
        rating = "narrow"
        warning = "One source dominates the article set. Add more sources before relying on the label."
    elif len(counts) < 3:
        rating = "limited"
        warning = "The source set is usable but limited. More sources would make the brief stronger."
    else:
        rating = "diverse"
        warning = "The source set has enough variety for a first-pass framing brief."
    return {
        "unique_sources": len(counts),
        "article_count": len(sources),
        "dominant_source_share": round(share, 3),
        "rating": rating,
        "warning": warning,
    }


def takeaways(trace: PipelineTrace) -> list[str]:
    label = trace.report.final_label
    context = source_context_summary(trace)
    items = [
        f"The article set is classified as {label} because: {trace.bias_judgment.rationale}",
        str(context["summary"]),
    ]
    if trace.critique.agree_with_detector:
        items.append("The critic accepted the detector's label after checking the cited spans.")
    else:
        items.append("The critic weakened or changed the detector's label, so treat the result as provisional.")
    if trace.bias_judgment.proxy_features:
        items.append("The strongest visible signals were: " + ", ".join(trace.bias_judgment.proxy_features) + ".")
    else:
        items.append("No strong directional proxy terms dominated the article set.")
    return items


def watch_items(trace: PipelineTrace) -> list[str]:
    diversity = source_diversity(trace)
    context = source_context_summary(trace)
    items = [
        "Do not treat this as a source rating. It is a rating of this article set.",
        "Read the cited spans before trusting the label.",
    ]
    if diversity["rating"] in {"thin", "narrow", "limited"}:
        items.append(str(diversity["warning"]))
    if trace.report.article_count < 3:
        items.append("The source set is thin; add more coverage before drawing a strong conclusion.")
    if context["uncataloged"]:
        items.append(
            f"{context['uncataloged']} source(s) were not in the seed context catalog; inspect them manually."
        )
    if trace.provider == "heuristic":
        items.append("Heuristic mode is deterministic and inspectable, but less nuanced than a model-backed run.")
    if any(stage.notes for stage in trace.stages if stage.name == "search_fetch"):
        items.append("Some source text came from metadata fallback, so full-article reading may change the result.")
    return items
