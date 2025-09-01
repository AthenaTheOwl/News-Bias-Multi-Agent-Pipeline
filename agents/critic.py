# agents/critic.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:8b"  # your existing critic model

_CRITIC_PROMPT = """
You are the Bias Critic. You receive:
1) A structured summary (with KEY POINTS and FRAMING/LANGUAGE NOTES)
2) The original article text
3) A simple bias proxy signal (score + flagged tokens), which might be wrong

TASK:
- Decide a refined political-bias judgment: one of [Left, Right, Neutral, Mixed, Undetermined].
- Explain WHY you reached that judgment, citing exact phrases and how they map to your decision rule.
- If the topic is non-political (e.g., sports match, prize money, transfer rumors), explicitly say the mapping to Left/Right is weak and default to Neutral unless policy/ideology is discussed.
- Keep it compact but specific.

Return EXACTLY this format:

REFINED BIAS JUDGMENT: <Left|Right|Neutral|Mixed|Undetermined>

REASONING:
- <bullet 1 referencing phrases and their interpretation>
- <bullet 2>
- <bullet 3>
- <add a bullet noting if the domain is sports/entertainment/business-without-policy and why that weakens political mapping>

TRIGGER PHRASES:
- "<quote 1>" -> <explain>
- "<quote 2>" -> <explain>
- (max 5 items)

NOTES ON PROXY:
- <when the proxy (score/flags) helped or misled, and why>

INPUTS
--- SUMMARY ---
{summary}

--- PROXY ---
Score: {score}
Flagged: {flagged}

--- ORIGINAL ---
{original}
""".strip()


def _ollama_call(prompt: str, model: str = MODEL, num_ctx: int = 8192, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def critic_analysis(summary: str, original: str, score, flagged) -> str:
    """
    Ask the critic to both classify and EXPLAIN its reasoning path.
    """
    prompt = _CRITIC_PROMPT.format(summary=summary, original=original, score=score, flagged=flagged)
    try:
        return _ollama_call(prompt)
    except requests.exceptions.RequestException as e:
        return f"[Error during critic analysis: {e}]"
    except ValueError:
        return "[Error: Invalid JSON response from critic]"
