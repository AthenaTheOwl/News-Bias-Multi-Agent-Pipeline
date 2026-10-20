# Test whether the critic follows the judgment it receives

The runtime deliberately gives the critic a real structured bias judgment.
That makes the critique relevant, but it also creates a measurement question:
does the critic inspect article evidence or simply repeat the earlier conclusion?
The current three-implementation comparison does not settle that question for
model-backed providers (`anthropic`, `openai`, `google`, `ollama`). In the
default `heuristic` provider the question is already answered in code: the
critic reads only the detector's label and confidence (`critique` in
`core/pipeline.py`), so it relays the label by construction and a blinded first
pass cannot run there without writing a new heuristic critic. This ablation
targets model-backed providers.

Keep the existing runtime as the control. In an experimental branch, add a first
review pass containing article IDs, titles, source names and verbatim verified
spans drawn from the article set or the summary, not from the detector. Withhold
the entire `StructuredBiasJudgment`: label, confidence, rationale, the evidence
spans the detector selected and `proxy_features`. Also withhold the app's
directional source-context cue (`core/framing.py` `source_context`), because
that cue alone can set the heuristic label. Ask the critic to mark supported
observations and missing evidence. In a second pass, reveal the actual detector
judgment and reconcile disagreements. Preserve both passes in the trace instead
of replacing the first with a polished final answer.

Build matched fixtures with the same article evidence and different proposed
labels. Include a correct label, an unsupported confident label and an
undetermined case. Keep at least one clean control so a critic that rejects every
judgment does not appear useful. Human labels should be assigned from the article
set without reading the detector's conclusion.

Compare unsupported-label acceptance, false objections, evidence-span validity,
disagreement resolution, wall time and calls. Two runtime behaviors would bias
these metrics if left as they are:

- When a model critic's reply fails to parse, omits fields or uses a label
  outside the allowed set, the current `critique` does not record an objection:
  missing fields default to the detector's label and to agreement, and an
  unusable reply falls through to the heuristic critic, which accepts any
  confident label. The trace records no fallback. Record
  parse and validation fallbacks as their own outcome, report them separately
  and keep them out of acceptance. In the experimental branch, have `critique`
  add a fallback note to its stage record.
- Every implementation raises on any citation error, so a run where the critic
  cites a span missing from the article produces no trace. The harness must keep
  `citation_errors` without raising, or count aborted runs as span failures, so
  span validity is measured over all runs.

Keep implementation, provider, model, rubric and article set fixed. Run the
ablation on the static implementation. All three implementations call the same
`core` critique with the same prompt and parsing, so results carry over to
LangChain and LangGraph by construction; running all three would add calls but
no signal beyond sampling noise.

[Audit Without Verification](https://arxiv.org/abs/2609.07680) motivates this
experiment. It does not establish how this app's critic behaves. The protocol is
a proposed ablation; no live model experiment or changed pipeline is included in
this note.

Verbatim span validation remains essential but limited: a quote can exist and
still fail to support a framing judgment. Keep that semantic question visible to
the reader of the framing brief. A framing brief should report what the evidence
supports and what remains undetermined; agreement among three frameworks that
share one core is not three independent annotations.
