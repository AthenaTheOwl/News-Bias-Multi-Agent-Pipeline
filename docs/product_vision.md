# product vision

## end state

News Bias Multi-Agent Pipeline should feel like a **framing brief** generator,
not a raw agent trace viewer.

A visitor brings a story or uses a curated story pack. The app returns a short
reader-facing brief:

- what the article set says happened
- which frames are visible across sources
- which exact spans support the label
- whether the critic accepted or weakened the detector output
- what to check before trusting or sharing the story

The implementation comparison remains useful, but it is not the primary
product. It belongs under the hood as proof that static Python, LangChain, and
LangGraph can share the same typed analysis contract.

## audience

- hiring managers evaluating an agent-orchestration artifact
- readers who want a quick explanation of framing differences
- developers comparing framework styles without losing the product goal

## non-goal

This is not a production bias rater. It does not rate publishers, infer intent,
or claim a political truth label. It classifies the framing of one article set
and shows the evidence it used.

## next increments

1. Extend the source-context seed catalog as new source families appear in
   real queries. Keep uncataloged sources explicit rather than guessing.
2. Add persistent report snapshots for completed briefs, not just shareable
   setup URLs.
3. Add a browser-level post-deploy canary once the deployment platform exposes
   stable DOM hooks for the hydrated Streamlit app.
