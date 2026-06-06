from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.demo_cases import DEMO_CASES
from impls.registry import get_runner


DEFAULT_URL = "https://news-bias-multi-agent-pipeline.streamlit.app/"


def check_public_url(url: str) -> list[str]:
    errors: list[str] = []
    try:
        response = requests.get(url, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        return [f"public URL request failed: {exc}"]
    if response.status_code >= 400:
        errors.append(f"public URL returned HTTP {response.status_code}")
    lowered = response.text.lower()
    blocked_terms = ["sign in to streamlit", "you need to log in", "authentication required"]
    if any(term in lowered for term in blocked_terms):
        errors.append("public URL appears to require authentication")
    return errors


def check_story_pack_variance() -> list[str]:
    errors: list[str] = []
    labels: list[str] = []
    confidences: list[float] = []
    for case in DEMO_CASES:
        trace = get_runner("static")(
            case.subject,
            provider="heuristic",
            fixture_articles=list(case.articles),
        )
        labels.append(trace.report.final_label)
        confidences.append(round(trace.report.confidence, 2))
        if case.expected_label != trace.report.final_label:
            errors.append(
                f"{case.slug}: expected {case.expected_label}, got {trace.report.final_label}"
            )
    if len(set(labels)) < 3:
        errors.append(f"story packs collapsed to too few labels: {dict(Counter(labels))}")
    if labels and all(label == "Center" for label in labels):
        errors.append("story packs collapsed to Center")
    if confidences and all(confidence == 0.58 for confidence in confidences):
        errors.append("story packs collapsed to confidence 0.58")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-deploy canary for the public Streamlit app.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--skip-url",
        action="store_true",
        help="Run local product invariants without checking the public URL.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    if not args.skip_url:
        errors.extend(check_public_url(args.url))
    errors.extend(check_story_pack_variance())

    if errors:
        for error in errors:
            print(f"post_deploy_canary FAIL: {error}", file=sys.stderr)
        return 1
    print("post_deploy_canary OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
