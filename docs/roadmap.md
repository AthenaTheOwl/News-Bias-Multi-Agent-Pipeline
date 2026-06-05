# Roadmap

## End state

The repo should be a small public showcase that a visitor can understand
in five minutes:

1. open the Streamlit app
2. enter a topic
3. choose static, LangChain, or LangGraph
4. run heuristic mode without keys or paste a provider key
5. inspect the same typed trace across implementations
6. read the limitations before trusting any label

## Current wave: shipped in this overhaul

- Shared Pydantic trace schema.
- Three real implementation directories.
- Bias detector wired before critic.
- Exact-span citation verifier.
- BYOK Streamlit UI.
- Ten-fixture eval gate.
- CI workflow.
- Deploy docs for Streamlit Community Cloud.

## Next iteration

- Add a "compare all three" saved report export.
- Add source-balance retrieval controls.
- Add a small human-reviewed fixture set from AllSides-rated sources.
- Add trace-to-eval export for failed runs.
- Add deployment badge after Streamlit Cloud app is connected.
