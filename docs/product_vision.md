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

1. Add source-diversity scoring so the brief warns when all sources come from
   one outlet family or one RSS wrapper.
2. Add shareable report URLs for completed briefs.
3. Add a small post-deploy canary that runs a story pack and verifies the brief
   renders without exposing developer trace by default.
