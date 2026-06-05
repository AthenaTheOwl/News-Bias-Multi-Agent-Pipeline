# Limitations

This repo is a learning artifact, not a production bias detector.

- **Small sample size.** A single query usually inspects a handful of
  articles. That is not enough to rate a source or a topic ecosystem.
- **Collapsed label space.** Left, right, center, mixed, and
  undetermined are crude labels. Real framing includes source selection,
  emphasis, omission, tone, reliability, geography, and issue-specific
  context.
- **Heuristic mode is a teaching aid.** The default no-key path looks for
  framing cues and charged language. It is deterministic for tests, not
  a trustworthy classifier.
- **Model mode still needs human review.** A model can return valid JSON
  and still make a bad media judgment.
- **Citation verification checks span presence only.** It proves that a
  phrase appeared in the article text. It does not prove the phrase
  supports the conclusion.
- **GNews and RSS have coverage gaps.** Search results depend on the
  provider, time window, region, and source availability.
- **No longitudinal tracking.** The app does not measure outlet behavior
  over time.
- **No source-diversity quotas.** The current retrieval layer does not
  enforce left, center, and right source balance.

The design follows the caution in media-bias research: automated bias
detection struggles most on political and cognitive bias categories, and
source-level ratings usually rely on trained human review or blind
survey methods. See `docs/research.md`.
