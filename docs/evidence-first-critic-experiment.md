# Test whether the critic follows the judgment it receives

The runtime deliberately gives the critic a real structured bias judgment.
That makes the critique relevant, but it also creates a measurement question:
does the critic inspect article evidence or simply repeat the earlier conclusion?
The current three-implementation comparison does not settle that question.

Keep the existing runtime as the control. In an experimental branch, add a first
review pass containing article IDs, titles, source context and verbatim verified
spans, while withholding the detector's label, confidence and rationale. Ask the
critic to mark supported observations and missing evidence. In a second pass,
reveal the actual detector judgment and reconcile disagreements. Preserve both
passes in the trace instead of replacing the first with a polished final answer.

Build matched fixtures with the same article evidence and different proposed
labels. Include a correct label, an unsupported confident label and an
undetermined case. Keep at least one clean control so a critic that rejects every
judgment does not appear useful. Human labels should be assigned from the article
set without reading the detector's conclusion.

Compare unsupported-label acceptance, false objections, evidence-span validity,
disagreement resolution, wall time and calls. Keep implementation, provider,
model, rubric and article set fixed. Test all three implementations on the same
fixtures; changing orchestration and the review procedure together would obscure
which change caused the difference.

[Audit Without Verification](https://arxiv.org/abs/2609.07680) motivates this
experiment. It does not establish how this app's critic behaves. The protocol is
a proposed ablation; no live model experiment or changed pipeline is included in
this note.

Verbatim span validation remains essential but limited: a quote can exist and
still fail to support a framing judgment. Keep that semantic question visible to
the reviewer. A shared draft should report what the evidence supports and what
remains undetermined; agreement among three frameworks that share one core is
not three independent annotations.
