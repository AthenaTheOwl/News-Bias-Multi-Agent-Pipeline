from __future__ import annotations

import json

import streamlit as st

from core.llm_provider import DEFAULT_MODELS
from core.schemas import LLMKeys, PipelineTrace
from impls.registry import IMPLEMENTATIONS, get_runner


st.set_page_config(page_title="News Bias Multi-Agent Pipeline", layout="wide")


def _keys_from_sidebar() -> LLMKeys:
    st.sidebar.header("Keys")
    st.sidebar.caption("BYOK: keys stay in Streamlit session state and are not written to disk.")
    return LLMKeys(
        anthropic_token=st.sidebar.text_input("Anthropic API key", type="password"),
        openai_token=st.sidebar.text_input("OpenAI API key", type="password"),
        google_token=st.sidebar.text_input("Google API key", type="password"),
        gnews_token=st.sidebar.text_input("GNews API key (optional)", type="password"),
        ollama_host=st.sidebar.text_input("Ollama host", value="http://localhost:11434"),
    )


def _render_trace(trace: PipelineTrace) -> None:
    top = st.columns([1, 1, 1, 1])
    top[0].metric("Implementation", trace.implementation)
    top[1].metric("Provider", trace.provider)
    top[2].metric("Articles", len(trace.articles))
    top[3].metric("Final label", trace.report.final_label)

    st.subheader("Final report")
    st.markdown(trace.to_markdown())

    st.subheader("Pipeline trace")
    for stage in trace.stages:
        with st.expander(stage.name, expanded=stage.name in {"bias_detect", "critique", "reconcile"}):
            if stage.notes:
                st.info("\n".join(stage.notes))
            st.json(stage.output)

    st.subheader("Articles")
    for article in trace.articles:
        with st.expander(article.title):
            st.write(article.url)
            st.write(article.text[:2500])

    with st.expander("Raw PipelineTrace JSON"):
        st.code(trace.model_dump_json(indent=2), language="json")


st.title("News Bias Multi-Agent Pipeline")
st.caption("A learning artifact: three framework styles, one typed trace, exact-span citations.")

with st.sidebar:
    impl = st.radio("Implementation", IMPLEMENTATIONS, index=0)
    provider = st.selectbox("LLM provider", ["heuristic", "anthropic", "openai", "google", "ollama"], index=0)
    model = st.text_input("Model", value=DEFAULT_MODELS.get(provider, ""))
    max_articles = st.slider("Max articles", min_value=1, max_value=8, value=4)
    keys = _keys_from_sidebar()

subject = st.text_input("Subject", value="AI regulation last week")
run_button = st.button("Run pipeline", type="primary")

st.info(
    "Heuristic mode needs no keys and is meant for inspection. Provider modes use your pasted key for this session only."
)

if run_button:
    try:
        with st.spinner("Running pipeline..."):
            trace = get_runner(impl)(
                subject,
                provider=provider,
                model=model or None,
                keys=keys,
                max_articles=max_articles,
            )
        _render_trace(trace)
    except Exception as exc:
        st.error(f"Run failed: {exc}")

with st.expander("Compare the three implementation styles"):
    st.markdown(
        """
        - **static**: sequential Python calls; easiest baseline to debug.
        - **langchain**: bounded Runnable composition over the same core.
        - **langgraph**: explicit state graph with one node per stage.

        All three return the same `PipelineTrace` shape.
        """
    )

    if st.button("Run all three with heuristic mode"):
        results = {}
        for candidate in IMPLEMENTATIONS:
            results[candidate] = get_runner(candidate)(
                subject,
                provider="heuristic",
                keys=LLMKeys(gnews_token=keys.gnews_token),
                max_articles=max_articles,
            )
        st.json(
            {
                name: {
                    "label": trace.report.final_label,
                    "confidence": trace.report.confidence,
                    "stages": [stage.name for stage in trace.stages],
                }
                for name, trace in results.items()
            }
        )
