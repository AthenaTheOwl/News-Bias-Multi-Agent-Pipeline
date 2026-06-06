from __future__ import annotations

from core.schemas import Article, PipelineTrace


FRAME_TERMS = {
    "Public investment": ["public investment", "workers", "union", "clean energy", "environmental justice", "equity"],
    "Cost and taxpayer burden": ["taxpayer", "cost", "price", "burden", "affordability", "mandate"],
    "Law and order": ["border", "law and order", "illegal immigration", "sheriff", "enforcement"],
    "Market and business impact": ["small business", "employers", "industry", "data centers", "fossil fuels"],
    "Institutional process": ["analysts", "committee", "agency", "court", "report", "lawmakers"],
    "Logistics and schedule": ["schedule", "match", "ticket", "venue", "tournament", "travel"],
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
            }
        )
    return rows


def takeaways(trace: PipelineTrace) -> list[str]:
    label = trace.report.final_label
    items = [
        f"The article set is classified as {label} because: {trace.bias_judgment.rationale}",
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
    items = [
        "Do not treat this as a source rating. It is a rating of this article set.",
        "Read the cited spans before trusting the label.",
    ]
    if trace.report.article_count < 3:
        items.append("The source set is thin; add more coverage before drawing a strong conclusion.")
    if trace.provider == "heuristic":
        items.append("Heuristic mode is deterministic and inspectable, but less nuanced than a model-backed run.")
    if any(stage.notes for stage in trace.stages if stage.name == "search_fetch"):
        items.append("Some source text came from metadata fallback, so full-article reading may change the result.")
    return items
