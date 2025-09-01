import requests

def write_final_report(summary, critic_output, bias_score, flagged):
    prompt = f"""
You are a Writer Agent.
Prepare a final structured news bias report.

Inputs:
Summary: {summary}
Bias Proxy Score: {bias_score}
Flagged Phrases: {flagged}
Critic Output: {critic_output}

Format the output as:
Headline: <generated if missing>
Summary: <from summary input>
Bias Assessment: <refined judgment from critic>
Context Notes: <any relevant context or flagged terms>
"""
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
                          timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"[Error during writer agent generation: {e}]"
    except ValueError:
        return "[Error: Invalid JSON response from writer]"
