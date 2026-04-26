<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

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

## status

repo contains the initial commit. files need cleanup and proper setup docs (LLM keys, env config, the 5 lite models in use). the bones are there; the polish isn't.

if you want to actually replicate it: 👋 reach out and i'll walk you through it.

## colophon

a learning project, kept honest. the point wasn't shipping a bias detector — it was learning langchain, langgraph, and the agentic patterns by re-implementing the same idea three different ways and noticing what changed.

resources used: HuggingFace's free curriculum, lots of docs, lots of small experiments.

*built downstairs.* — [the basement, room 7](https://github.com/AthenaTheOwl)
