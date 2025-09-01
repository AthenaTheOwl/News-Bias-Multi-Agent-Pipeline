# agents/preprocessor.py

from datetime import date, timedelta
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama  # fallback

# Use DeepSeek-R1 for parsing vague prompts
deepseek = Ollama(model="deepseek-r1:8b")

def fmt_human(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"

_today = date.today()
_today_str = fmt_human(_today)
_last_year_num = _today.year - 1
_last_week_start = _today - timedelta(days=7)
_last_week_start_str = fmt_human(_last_week_start)

# IMPORTANT: we force the model to output STRICT JSON with ISO dates
_TEMPLATE = """
You are a Preprocessor Agent.
Convert a vague news prompt into STRICT JSON:
{{
  "query": "<short keyword/topic, no quotes or filler>",
  "from": "YYYY-MM-DD",   // optional
  "to": "YYYY-MM-DD"      // optional
}}

Rules:
- If the prompt says "today", set both from/to to {today_iso}.
- If it says "last week", set from to {last_week_start_iso} and to to {today_iso}.
- If it says "last year", set from to "{last_year}-01-01" and to to "{last_year}-12-31".
- If no dates are implied, omit "from" and "to".
- The "query" must be a short keyword string (e.g., "Singapore elections", "Ukraine", "Kristie Mewis").

Examples (INPUT -> OUTPUT):
- "Singapore today" ->
{{"query":"Singapore","from":"{today_iso}","to":"{today_iso}"}}
- "Moon last year" ->
{{"query":"Moon","from":"{last_year}-01-01","to":"{last_year}-12-31"}}
- "Ukraine last week" ->
{{"query":"Ukraine","from":"{last_week_start_iso}","to":"{today_iso}"}}

Return ONLY the JSON. No extra text.

INPUT: "{user_prompt}"
"""

prompt = PromptTemplate.from_template(_TEMPLATE)
chain = prompt | deepseek | StrOutputParser()

def _strip_think(text: str) -> str:
    if "<think>" in text:
        parts = text.split("</think>", 1)
        return parts[1].strip() if len(parts) > 1 else text
    return text

def _coerce_json(text: str) -> dict:
    text = text.strip()
    text = _strip_think(text)
    # Try to find the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)
    return json.loads(text)

def run_preprocessor_agent(user_prompt: str) -> dict:
    structured = chain.invoke({
        "user_prompt": user_prompt,
        "today_iso": _today.strftime("%Y-%m-%d"),
        "last_year": str(_last_year_num),
        "last_week_start_iso": _last_week_start.strftime("%Y-%m-%d"),
    })
    data = _coerce_json(structured)
    # basic sanity cleanup
    if "query" in data:
        data["query"] = str(data["query"]).strip()
    return data

# Back-compat aliases
def preprocess_user_query(user_prompt: str) -> dict:
    return run_preprocessor_agent(user_prompt)

def run_preprocessor_agent_str(user_prompt: str) -> str:
    import json
    return json.dumps(run_preprocessor_agent(user_prompt))
