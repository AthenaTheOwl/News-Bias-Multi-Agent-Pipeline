from __future__ import annotations

import argparse
import json
import os
import sys

from core.schemas import LLMKeys
from impls.registry import IMPLEMENTATIONS, get_runner


def _write_utf8(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the news-bias pipeline.")
    parser.add_argument("subject", nargs="?", default="AI regulation last week")
    parser.add_argument("--impl", choices=IMPLEMENTATIONS, default="static")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "heuristic"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-articles", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print full PipelineTrace JSON")
    args = parser.parse_args()

    keys = LLMKeys(
        anthropic_token=os.environ.get("ANTHROPIC_API_KEY"),
        openai_token=os.environ.get("OPENAI_API_KEY"),
        google_token=os.environ.get("GOOGLE_API_KEY"),
        gnews_token=os.environ.get("GNEWS_API_KEY"),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    trace = get_runner(args.impl)(
        args.subject,
        provider=args.provider,
        model=args.model,
        keys=keys,
        max_articles=args.max_articles,
    )
    if args.json:
        _write_utf8(trace.model_dump_json(indent=2))
    else:
        _write_utf8(trace.to_markdown())
        _write_utf8("\n---\n")
        _write_utf8(json.dumps({"implementation": trace.implementation, "provider": trace.provider}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
