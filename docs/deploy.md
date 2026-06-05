# Deploy

Deployment target: Streamlit Community Cloud.

Requested URL: `https://news-bias-pipeline.streamlit.app`

## Steps

1. Push `main` to GitHub.
2. Open Streamlit Community Cloud.
3. Create an app from `AthenaTheOwl/News-Bias-Multi-Agent-Pipeline`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Python version: 3.12.
7. Secrets: leave blank for the default BYOK posture.
8. App URL: request `news-bias-pipeline`.

## Why not Vercel

Vercel is a better target for a TypeScript or Next.js app. This repo's
product boundary is Python plus Streamlit. Deploying to Vercel would
require a separate Python backend or a full frontend rewrite before the
agent comparison becomes visible.

## Smoke checks before deploy

```powershell
python -m pip install -r requirements.txt
python -m pytest
python main.py "AI regulation last week" --impl static --provider heuristic
python -m streamlit run app.py
```
