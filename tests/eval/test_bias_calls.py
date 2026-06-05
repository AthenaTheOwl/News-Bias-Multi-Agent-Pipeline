from __future__ import annotations

from pathlib import Path

import yaml

from core.news_search import article_id_for
from core.schemas import Article
from impls.registry import IMPLEMENTATIONS, get_runner


FIXTURE_PATH = Path(__file__).with_name("golden_fixtures.yaml")


def _load_cases() -> list[dict]:
    data = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    return list(data["fixtures"])


def _articles(case: dict) -> list[Article]:
    raw_articles = case.get("articles") or [case["article"]]
    out = []
    for raw in raw_articles:
        url = f"fixture://{case['id']}/{raw['title'].replace(' ', '-')}"
        out.append(
            Article(
                id=article_id_for(url),
                title=raw["title"],
                url=url,
                source="golden-fixture",
                text=raw["text"],
            )
        )
    return out


def test_eval_suite_hits_threshold() -> None:
    cases = _load_cases()
    for impl in IMPLEMENTATIONS:
        matches = 0
        for case in cases:
            trace = get_runner(impl)(
                case["subject"],
                fixture_articles=_articles(case),
                provider="heuristic",
                max_articles=4,
            )
            if trace.report.final_label == case["expected"]:
                matches += 1
        score = matches / len(cases)
        assert score >= 0.8, f"{impl} agreement {score:.2f} below threshold"
