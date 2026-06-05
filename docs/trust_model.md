# Trust model

## Hosted demo

Target host: Streamlit Community Cloud.

Entrypoint: `app.py`.

The deployed app should not store Anthropic, OpenAI, Google, or Ollama
credentials in Streamlit secrets. Visitors paste keys into sidebar
password fields for the current session.

## What the app sees

During a run, the app sees:

- the subject prompt
- selected implementation
- selected provider and model
- pasted keys for the current session
- fetched article text
- model responses
- structured trace data

## What the app stores

The app does not write user keys to disk. The code passes keys through a
`LLMKeys` object. The Streamlit UI keeps them in session state for the
browser session.

The repo does not commit generated article caches or local reports.

## Local CLI

For local CLI runs, environment variables are supported because they are
standard developer ergonomics:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `GNEWS_API_KEY`
- `OLLAMA_HOST`

`.env` files are gitignored.

## Server-side GNews key

The safest hosted posture is no server-side GNews key. The app works
with RSS fallback. If a maintainer chooses to configure a GNews key in
Streamlit secrets later, that key should be treated as cheap to rotate
and documented in this file.
