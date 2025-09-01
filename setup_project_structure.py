import os
from pathlib import Path

# === Set your project root path here ===
PROJECT_ROOT = Path(r"E:\Agents\news_bias_multiagent_pipeline")

# === Directory structure ===
DIRS = [
    "agents",
    "retrieval",
    "vectorstore",
    "visualization",
    "cache/faiss_index",
    "reports",
    "models",
    "langgraph",
    "memory"
]

# === Files with safe quoting ===
FILES = {
    "main.py": """# Main pipeline orchestrator
if __name__ == '__main__':
    print('Pipeline entry point')
""",

    "agents/summarizer.py": "# Summarizer agent\n",
    "agents/bias_detector.py": "# Bias detection agent\n",
    "agents/critic.py": "# Critic agent\n",
    "agents/writer.py": "# Writer agent\n",

    "agents/preprocessor.py": r'''# agents/preprocessor.py
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:8b"

SYSTEM = """You are a preprocessor that turns vague prompts into structured search JSON.
Return ONLY valid JSON with keys: topic, start_date, end_date, structured_query.
- start_date/end_date format: YYYY-MM-DD
- If the prompt says "today", use the current date (America/New_York).
- If "last year", use Jan 1 to Dec 31 of last calendar year.
- If no dates, use a reasonable recent window: start_date = today-30d, end_date = today."""

def _ollama(prompt: str, timeout=60) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": f"{SYSTEM}\n\nUser: {prompt}\nAssistant:",
        "stream": False
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def run_preprocessor_agent(user_prompt: str) -> dict:
    raw = _ollama(user_prompt)
    try:
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.lower().startswith("json"):
                raw = raw.split("\n", 1)[1]
        data = json.loads(raw)
        assert all(k in data for k in ("topic", "start_date", "end_date", "structured_query"))
        return data
    except Exception:
        return {
            "topic": user_prompt,
            "start_date": "",
            "end_date": "",
            "structured_query": f"Search for news articles about {user_prompt} in the last 30 days."
        }
''',

    "retrieval/rss_fetch.py": "# RSS fetch functions\n",
    "retrieval/text_extraction.py": "# Article text extraction\n",

    "visualization/forge_sd.py": "# ForgeUI Stable Diffusion integration\n",

    "cache/news.sqlite": "",

    "models/__init__.py": """# Model interfaces
from langchain_community.llms import Ollama
from langchain_core.language_models.llms import LLM

# You can load multiple LLMs and switch as needed
llama3: LLM = Ollama(model="llama3")
gemma: LLM = Ollama(model="gemma:7b")
deepseek: LLM = Ollama(model="deepseek-r1:8b")  # Corrected to use r1:8b
""",

    "tools.py": """# LangChain-compatible tools
from langchain.tools import tool

@tool
def preprocessor_tool(user_prompt: str) -> str:
    from agents.preprocessor import run_preprocessor_agent
    return run_preprocessor_agent(user_prompt)

@tool
def summarizer_tool(article_text: str) -> str:
    return f"Summary of: {article_text[:75]}..."

@tool
def critic_tool(summary: str) -> str:
    return f"Bias critique for: {summary[:75]}..."

@tool
def writer_tool(summary_and_critique: str) -> str:
    return f"Full article based on: {summary_and_critique[:75]}..."

@tool
def search_tool(structured_query: str) -> str:
    return f"Searching with query: {structured_query}"
""",

    "agent_runner.py": """# LangChain agent runner logic
from langchain.agents import initialize_agent, AgentType, Tool
from models import llama3
from tools import summarizer_tool, critic_tool, writer_tool, search_tool, preprocessor_tool

tools = [
    Tool.from_function(preprocessor_tool, name="Preprocessor", description="Preprocess vague user queries into structured search requests"),
    Tool.from_function(summarizer_tool, name="SummarizeArticle", description="Summarize an article in 5 sentences or fewer"),
    Tool.from_function(critic_tool, name="BiasCritic", description="Critique article bias based on bias score and summary"),
    Tool.from_function(writer_tool, name="ArticleWriter", description="Write a professional article from summary and critique"),
    Tool.from_function(search_tool, name="SemanticSearch", description="Search recent stored summaries with FAISS/SQLite hybrid"),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llama3,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

def run_agent(prompt: str):
    print(f"\\n🧠 Running agent on: {prompt}")
    result = agent_executor.run(prompt)
    print("✅ Agent finished.")
    return result
""",

    "streamlit_ui.py": """# Streamlit UI for user interaction
import streamlit as st
from agent_runner import run_agent

st.title("News Bias Multi-Agent Pipeline")

user_prompt = st.text_input("Enter your query:")
if st.button("Run Pipeline"):
    if user_prompt:
        output = run_agent(user_prompt)
        st.write(output)
    else:
        st.warning("Please enter a query.")
""",

    "langgraph/agent_flow.py": """# LangGraph agent flow planning
from langgraph.graph import StateGraph
from tools import summarizer_tool, critic_tool, writer_tool

graph = StateGraph()
graph.add_node("summarizer", summarizer_tool)
graph.add_node("critic", critic_tool)
graph.add_node("writer", writer_tool)

graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "critic")
graph.add_edge("critic", "writer")

graph.set_finish_point("writer")
compiled_graph = graph.compile()

if __name__ == "__main__":
    result = compiled_graph.invoke({
        "article_text": "Put your raw article text here",
        "bias_score": 4,
        "flags": ["loaded language"]
    })
    print(result)
"""
}

# === Requirements ===
REQUIREMENTS = """langchain>=0.2.0
langgraph>=0.0.5
feedparser>=6.0.10
newspaper3k>=0.2.8
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
sqlite-utils>=3.31
transformers>=4.41.2
torch>=2.3.1
requests
lxml_html_clean
streamlit>=1.35.0
tqdm
"""

# === Create directories ===
for d in DIRS:
    path = PROJECT_ROOT / d
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured folder: {path}")

# === Create files ===
for filename, content in FILES.items():
    path = PROJECT_ROOT / filename
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {path}")

# === Create requirements.txt ===
req_path = PROJECT_ROOT / "requirements.txt"
if not req_path.exists():
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS)
    print(f"Created requirements.txt with core dependencies")

print(f"\n✅ Project structure created under: {PROJECT_ROOT}")
