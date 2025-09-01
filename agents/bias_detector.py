# Bias detection agent
from transformers import pipeline

# Public sentiment model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def detect_bias(text):
    """
    Use sentiment as a proxy for subjectivity/bias.
    - Strong positive or negative sentiment -> consider biased (-1)
    - Neutral sentiment -> unbiased (0)
    """
    results = classifier(text[:512])  # limit text length for speed
    label = results[0]['label']
    score = results[0]['score']

    # Simplified bias scoring
    bias_score = -1 if label in ["NEGATIVE", "POSITIVE"] else 0
    flagged = [f"{label} ({score:.2f})"]

    return bias_score, flagged

if __name__ == "__main__":
    text = "Regulators slammed the brakes on what critics call the most reckless corporate experiment of the decade."
    print(detect_bias(text))
