This is a for-fun news article-based multi-agent project featuring agents to:
1. Parse subject prompt on a UI (built on Streamlit)
2. Pull articles on the specific subject (uses GNews API)
3. Summarize the article
4. Detect bias (leaning right, left, etc.,)
5. Critically evaluate the bias detection in step #4
6. Summarize all the preceding steps

Initially, it was a static implementation via a static main.py followed by a prototype on langchain followed by a workflow in langgraph.
It was mostly a way for me to learn how to use these tools as I self-learned this stuff with resources like the ones on HuggingFace.

Repo contains the initial commit. Files need a lot of clean-up and descriptions to setup LLMs and env (for example, the 5 lite models used).
Feel free to reach out if actually interested in replicating :)
