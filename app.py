from __future__ import annotations

from time import perf_counter
from urllib.parse import urlencode

import streamlit as st

from core.demo_cases import DemoCase, demo_case_titles, get_demo_case
from core.framing import framing_table, source_context_summary, source_diversity, takeaways, watch_items
from core.llm_provider import DEFAULT_MODELS
from core.schemas import Article, LLMKeys, PipelineTrace
from impls.registry import IMPLEMENTATIONS, get_runner


IMPLEMENTATION_NOTES = {
    "static": {
        "label": "Plain Python",
        "meaning": "The same analysis steps run in a simple sequence.",
    },
    "langchain": {
        "label": "LangChain",
        "meaning": "The same steps are wrapped as a bounded runnable chain.",
    },
    "langgraph": {
        "label": "LangGraph",
        "meaning": "The same steps run as explicit state-machine nodes.",
    },
}

LABEL_MEANING = {
    "Left": "The article set strongly emphasizes left-coded policy frames.",
    "Lean Left": "The article set leans toward left-coded policy frames, but this is not a source rating.",
    "Center": "The article set is mostly informational or does not contain clear ideological framing.",
    "Lean Right": "The article set leans toward right-coded policy frames, but this is not a source rating.",
    "Right": "The article set strongly emphasizes right-coded policy frames.",
    "Mixed": "The article set contains meaningful cues from both left-coded and right-coded frames.",
    "Undetermined": "The article set has too little or too ambiguous evidence for a directional label.",
}

STAGE_LABELS = {
    "preprocess": "Parse the subject and date window",
    "search_fetch": "Collect article text or metadata",
    "summarize": "Extract neutral summary and framing spans",
    "bias_detect": "Score framing cues and produce a label",
    "critique": "Challenge the detector output",
    "reconcile": "Write the final reader-facing result",
}

APP_URL = "https://news-bias-multi-agent-pipeline.streamlit.app/"
PROVIDERS = ["heuristic", "anthropic", "openai", "google", "ollama"]


st.set_page_config(page_title="News Bias Multi-Agent Pipeline", layout="wide")


def _query_value(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def _index(values: list[str] | tuple[str, ...], selected: str) -> int:
    try:
        return list(values).index(selected)
    except ValueError:
        return 0


def _share_url(mode: str, subject: str, provider: str, implementation: str, demo_case: DemoCase | None) -> str:
    params = {
        "mode": "live" if mode == "Live search" else "story",
        "subject": subject,
        "provider": provider,
        "impl": implementation,
    }
    if demo_case is not None:
        params["scenario"] = demo_case.slug
    return APP_URL + "?" + urlencode(params)


def _update_query_params(mode: str, subject: str, provider: str, implementation: str, demo_case: DemoCase | None) -> None:
    params = {
        "mode": "live" if mode == "Live search" else "story",
        "subject": subject,
        "provider": provider,
        "impl": implementation,
    }
    if demo_case is not None:
        params["scenario"] = demo_case.slug
    st.query_params.clear()
    st.query_params.update(params)


def _keys_from_sidebar() -> LLMKeys:
    st.sidebar.header("Optional keys")
    st.sidebar.caption("Leave blank for heuristic mode. Pasted keys are not written to disk.")
    return LLMKeys(
        anthropic_token=st.sidebar.text_input("Anthropic key", type="password"),
        openai_token=st.sidebar.text_input("OpenAI key", type="password"),
        google_token=st.sidebar.text_input("Google key", type="password"),
        gnews_token=st.sidebar.text_input("GNews key", type="password"),
        ollama_host=st.sidebar.text_input("Ollama host", value="http://localhost:11434"),
    )


def _run(
    implementation: str,
    subject: str,
    provider: str,
    model: str | None,
    keys: LLMKeys,
    max_articles: int,
    fixture_articles: list[Article] | None,
) -> tuple[PipelineTrace, float]:
    started = perf_counter()
    trace = get_runner(implementation)(
        subject,
        provider=provider,
        model=model or None,
        keys=keys,
        max_articles=max_articles,
        fixture_articles=fixture_articles,
    )
    return trace, perf_counter() - started


def _selected_articles(mode: str, demo_case: DemoCase | None) -> list[Article] | None:
    if mode == "Story pack" and demo_case is not None:
        return list(demo_case.articles)
    return None


def _article_title(trace: PipelineTrace, article_id: str) -> str:
    article = next((item for item in trace.articles if item.id == article_id), None)
    if article is None:
        return article_id
    return f"{article.source}: {article.title}"


def _render_framing_brief(trace: PipelineTrace, elapsed: float) -> None:
    label = trace.report.final_label
    diversity = source_diversity(trace)
    context = source_context_summary(trace)
    st.subheader("Framing brief")

    metrics = st.columns([1, 1, 1, 1, 1, 1])
    metrics[0].metric("Frame", label)
    metrics[1].metric("Confidence", f"{trace.report.confidence:.2f}")
    metrics[2].metric("Articles", trace.report.article_count)
    metrics[3].metric("Sources", diversity["unique_sources"])
    metrics[4].metric("Engine", IMPLEMENTATION_NOTES[trace.implementation]["label"])
    metrics[5].metric("Runtime", f"{int(elapsed * 1000)} ms")

    st.markdown(f"**Bottom line:** {LABEL_MEANING.get(label, 'Review the evidence below.')}")
    st.write(trace.report.executive_summary)
    st.caption(f"Source diversity: {diversity['rating']} - {diversity['warning']}")
    st.caption(f"Source context: {context['summary']}")

    st.markdown("**What to take away**")
    for item in takeaways(trace):
        st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown("**Coverage framing map**")
        rows = framing_table(trace)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No source rows available.")

    with st.container(border=True):
        st.markdown("**Watch before sharing**")
        for item in watch_items(trace):
            st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown("**Critic review**")
        verdict = "agreed with the detector" if trace.critique.agree_with_detector else "changed or weakened the detector label"
        st.write(f"The critic {verdict}. {trace.critique.reasoning}")
        st.caption(trace.critique.proxy_notes)


def _render_evidence(trace: PipelineTrace) -> None:
    st.subheader("Evidence spans")
    if not trace.bias_judgment.evidence:
        st.info("No citation spans were produced for this run.")
        return
    for citation in trace.bias_judgment.evidence:
        with st.container(border=True):
            st.caption(_article_title(trace, citation.article_id))
            st.write(citation.span_text)


def _render_sources(trace: PipelineTrace) -> None:
    st.subheader("Source set")
    if not trace.articles:
        st.info("No articles were available for this query.")
        return
    for article in trace.articles:
        with st.expander(f"{article.source}: {article.title}"):
            st.write(article.url)
            st.write(article.text[:1200])


def _render_path(trace: PipelineTrace) -> None:
    st.subheader("Analysis path")
    for stage in trace.stages:
        label = STAGE_LABELS.get(stage.name, stage.name)
        note = "; ".join(stage.notes) if stage.notes else "completed"
        st.markdown(f"- **{label}**: {note}")


def _render_developer_trace(trace: PipelineTrace) -> None:
    with st.expander("Developer trace"):
        st.json(
            {
                "subject": trace.subject,
                "implementation": trace.implementation,
                "provider": trace.provider,
                "stages": [stage.model_dump() for stage in trace.stages],
                "citation_errors": trace.citation_errors,
                "framework_notes": trace.framework_notes,
            }
        )


def _render_trace(trace: PipelineTrace, elapsed: float) -> None:
    _render_framing_brief(trace, elapsed)
    _render_evidence(trace)
    _render_sources(trace)
    with st.expander("Under the hood: analysis path"):
        _render_path(trace)
        _render_developer_trace(trace)


def _render_comparison(results: dict[str, tuple[PipelineTrace, float]]) -> None:
    labels = {trace.report.final_label for trace, _elapsed in results.values()}
    st.subheader("Implementation comparison")
    if len(labels) == 1:
        st.success(
            "All three implementations produced the same reader-facing result. That is good: the framework changed, not the analysis contract."
        )
    else:
        st.warning("The implementations disagreed. Open each tab below to inspect the stage path.")

    rows = []
    for name, (trace, elapsed) in results.items():
        rows.append(
            {
                "implementation": IMPLEMENTATION_NOTES[name]["label"],
                "what changed": IMPLEMENTATION_NOTES[name]["meaning"],
                "final label": trace.report.final_label,
                "confidence": round(trace.report.confidence, 3),
                "runtime": f"{int(elapsed * 1000)} ms",
                "citations": "clean" if not trace.citation_errors else "failed",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    tabs = st.tabs([IMPLEMENTATION_NOTES[name]["label"] for name in results])
    for tab, (name, (trace, elapsed)) in zip(tabs, results.items(), strict=True):
        with tab:
            st.markdown(f"**{IMPLEMENTATION_NOTES[name]['meaning']}**")
            _render_path(trace)
            with st.expander("Result details"):
                _render_framing_brief(trace, elapsed)
                _render_evidence(trace)


st.title("Framing Brief")
st.caption("Turn a news topic into a concise read on framing, evidence, and what to check before sharing.")

scenario_titles = demo_case_titles()
initial_scenario = _query_value("scenario", "climate-policy")
initial_case = get_demo_case(initial_scenario)
initial_mode = "Live search" if _query_value("mode", "story") == "live" else "Story pack"
initial_provider = _query_value("provider", "heuristic")
initial_impl = _query_value("impl", "static")

with st.sidebar:
    provider = st.selectbox("Analysis mode", PROVIDERS, index=_index(PROVIDERS, initial_provider))
    model = st.text_input("Model", value=DEFAULT_MODELS.get(provider, ""))
    max_articles = st.slider("Max live articles", min_value=1, max_value=8, value=4)
    keys = _keys_from_sidebar()

mode = st.radio(
    "Story source",
    ["Story pack", "Live search"],
    horizontal=True,
    index=_index(["Story pack", "Live search"], initial_mode),
)
demo_case: DemoCase | None = None

if mode == "Story pack":
    selected_title = st.selectbox("Story pack", scenario_titles, index=_index(scenario_titles, initial_case.title))
    demo_case = get_demo_case(selected_title)
    subject = st.text_input("Subject", value=_query_value("subject", demo_case.subject))
    st.write(demo_case.description)
else:
    subject = st.text_input("Subject", value=_query_value("subject", "AI regulation last week"))
    st.write("Live search uses a GNews token if supplied. Without one, it uses query-filtered public RSS metadata.")

fixture_articles = _selected_articles(mode, demo_case)

with st.expander("Advanced: choose implementation", expanded=False):
    selected_impl = st.radio(
        "Implementation",
        IMPLEMENTATIONS,
        index=_index(IMPLEMENTATIONS, initial_impl),
        horizontal=True,
    )
    st.caption(IMPLEMENTATION_NOTES[selected_impl]["meaning"])

with st.expander("Share this setup", expanded=False):
    st.caption("This URL stores only the story selection and engine choice. It never includes pasted API keys.")
    st.code(_share_url(mode, subject, provider, selected_impl, demo_case))
    if st.button("Update browser URL"):
        _update_query_params(mode, subject, provider, selected_impl, demo_case)

buttons = st.columns([1, 1])
run_single = buttons[0].button("Generate framing brief", type="primary")
run_compare = buttons[1].button("Under the hood: compare engines")

if run_single:
    try:
        _update_query_params(mode, subject, provider, selected_impl, demo_case)
        with st.spinner("Analyzing story..."):
            trace, elapsed = _run(selected_impl, subject, provider, model, keys, max_articles, fixture_articles)
        _render_trace(trace, elapsed)
    except Exception as exc:
        st.error(f"Run failed: {exc}")

if run_compare:
    try:
        _update_query_params(mode, subject, provider, selected_impl, demo_case)
        with st.spinner("Running the same story through all three implementations..."):
            results = {
                implementation: _run(implementation, subject, provider, model, keys, max_articles, fixture_articles)
                for implementation in IMPLEMENTATIONS
            }
        _render_comparison(results)
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")

if not run_single and not run_compare:
    st.subheader("End state")
    st.markdown(
        """
        This is not trying to be a magic bias labeler. The product goal is a small,
        inspectable **framing brief**:

        - what the article set says happened
        - which frames are visible across sources
        - which exact spans support the label
        - whether the critic thought the detector overreached
        - what to check before you trust or share the story
        """
    )
