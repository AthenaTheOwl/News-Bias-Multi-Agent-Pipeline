<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

> **SECURITY NOTICE (2026-05-31):** earlier revisions of this repo contained a
> hardcoded GNews API key in `retrieval/rss_fetch.py`. The key has been
> **revoked** and the code now reads `GNEWS_API_KEY` from the environment. If
> you forked, cloned, or copied any code from this repo before this notice,
> please **pull main and rotate any keys you may have copied**. A follow-up
> history-rewrite (git filter-repo) will scrub the leaked key from prior
> commits — until that lands, the key is still discoverable in git history but
> is no longer valid.

# N° 07 · news bias · multi-agent pipeline

> *agents reading agents reading the news.*

a small chain of agents that pulls news on a subject, summarizes it, decides where it leans, then turns around and critiques its own bias detection — and finally summarizes the whole loop. built three times in three styles, on purpose, as a way to learn the tools.

`python` · `streamlit` · `langchain` · `langgraph` · `gnews` · 2024 · **status: solved**

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the chain

```
   subject prompt (streamlit UI)
              │
              ▼
       ┌─────────────┐
       │ pull        │  ← GNews API
       └─────────────┘
              │
              ▼
       ┌─────────────┐
       │ summarize   │
       └─────────────┘
              │
              ▼
       ┌─────────────┐
       │ detect bias │  → left / right / center / etc.
       └─────────────┘
              │
              ▼
       ┌─────────────┐
       │ critique    │  ← evaluates the detector
       └─────────────┘
              │
              ▼
       ┌─────────────┐
       │ summarize   │  ← of everything above
       └─────────────┘
              │
              ▼
            output
```

each agent is small. each does one thing. the interesting part is the **critique** step — an agent whose only job is to second-guess the bias call, and a final summarizer that has to reconcile both views.

## the three implementations

| version | what it taught |
|---|---|
| `main.py` (static)         | the whole pipeline, hand-wired. no framework. |
| langchain prototype        | how chains, tools, and prompt templates feel in practice |
| langgraph workflow         | how the same chain looks as a state graph with explicit edges |

each one solves the same problem, on the same data, with the same prompts — so the differences are about the framework, not the task.

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## configuration

this pipeline calls the GNews API. you'll need your own key — get one at
[gnews.io](https://gnews.io/) and wire it up via environment variable:

```bash
cp .env.example .env
# then edit .env and set GNEWS_API_KEY=<your key>
```

at runtime the code reads `os.environ["GNEWS_API_KEY"]`. if the var is unset
the search step will raise `RuntimeError("GNEWS_API_KEY env var not set — see
.env.example")` and the pipeline falls back to plain RSS feeds.

`.env` is gitignored. **do not** commit real keys.

## status

repo contains the initial commit. files need cleanup and proper setup docs (LLM keys, env config, the 5 lite models in use). the bones are there; the polish isn't.

if you want to actually replicate it: 👋 reach out and i'll walk you through it.

## colophon

a learning project, kept honest. the point wasn't shipping a bias detector — it was learning langchain, langgraph, and the agentic patterns by re-implementing the same idea three different ways and noticing what changed.

resources used: HuggingFace's free curriculum, lots of docs, lots of small experiments.

*built downstairs.* — [the basement, room 7](https://github.com/AthenaTheOwl)
