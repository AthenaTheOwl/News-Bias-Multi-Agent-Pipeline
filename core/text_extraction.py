from __future__ import annotations

import re
from html.parser import HTMLParser

import requests


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = " ".join(data.split())
        if len(cleaned) >= 30:
            self.parts.append(cleaned)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "\n".join(parser.parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_article_text(url: str, *, timeout: int = 25) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "news-bias-multi-agent-demo/2026"},
    )
    response.raise_for_status()
    return html_to_text(response.text)
