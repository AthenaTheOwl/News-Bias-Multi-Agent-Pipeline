# Research notes

These sources shaped the overhaul.

## Media-bias methodology

- [AllSides media-bias rating methods](https://www.allsides.com/about/media-bias-rating-methods): useful because it separates source ratings from single-article inspection and uses multipartisan human review plus blind surveys.
- [Ad Fontes methodology](https://adfontesmedia.com/methodology/): useful because it distinguishes reliability and bias and relies on trained content analysis.
- [Poynter on media-bias charts](https://www.poynter.org/fact-checking/media-literacy/2021/should-you-trust-media-bias-charts/): useful caution that bias charts need context and should not be treated as absolute truth.
- [MBIB paper](https://arxiv.org/abs/2304.13148): useful because it groups multiple bias types and reports that models struggle with cognitive and political bias.
- [Systematic review on media-bias detection](https://www.sciencedirect.com/science/article/pii/S0957417423021437): useful because it frames automatic media-bias detection as a broad family of tasks, not a single label.

## Engineering choices

- [Streamlit Community Cloud file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization): supports a root `app.py` plus `requirements.txt`, which matches this repo.
- [Streamlit Community Cloud secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management): supports secrets but the selected posture is BYOK with no provider keys stored server-side.
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/): used for typed traces and validation.
- [LangGraph StateGraph reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph): used for the explicit state-machine implementation.

## Sibling-repo patterns copied

- `supplier-risk-rag-agent`: BYOK posture, exact-span citation
  verification, and eval gates.
- `trace-to-eval-harness`: deterministic regression artifacts and
  honest "what it catches / what it does not catch" framing.
- `ai-field-brief`: evidence-spine habit: every promoted claim needs a
  source and a testable action path.
