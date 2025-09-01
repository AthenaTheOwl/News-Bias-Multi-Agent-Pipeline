# streamlit_ui.py
import streamlit as st
from langgraph.news_graph import run_news_graph

st.set_page_config(page_title="News Bias LangGraph", layout="wide")
st.title("📰 News Bias Multi-Agent (LangGraph)")

with st.form("q"):
    user_prompt = st.text_input("Your query (e.g., 'Singapore today', 'Brazil last 12 months')", "Singapore today")
    max_articles = st.slider("Max articles", 3, 20, 8)
    do_critic = st.checkbox("Run critic step", value=True)
    go = st.form_submit_button("Run")

if go:
    with st.spinner("Running graph..."):
        out = run_news_graph(user_prompt, max_articles=max_articles, do_critic=do_critic)

    st.subheader("Structured Query")
    st.code(out.get("structured_query", {}), language="json")

    st.subheader("Found Articles")
    hits = out.get("hits", [])
    if hits:
        for i, a in enumerate(hits, 1):
            st.write(f"**{i}. {a.get('title','(untitled)')}**")
            st.write(a.get("link",""))
    else:
        st.write("_None_")

    st.subheader("Overall Summary")
    st.markdown(out.get("overall_summary", "_None_"))

    if do_critic:
        st.subheader("Critique")
        st.markdown(out.get("critique", "_None_"))

    st.subheader("Final Report")
    st.markdown(out.get("final_report", "_None_"))
