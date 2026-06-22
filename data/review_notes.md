# Label Review Notes

Review date: 2026-06-22

The 38-row `needs_review.csv` queue was checked against the two-question taxonomy before
submission:

1. Does the comment make a debatable sports claim?
2. If yes, is the claim backed by a real explanatory chain or multiple pieces of evidence?

The final label set keeps the queued labels as written. The hardest boundary cases are the same
ones discussed in `README.md`: emotional comments that smuggle in claims, one-stat claims that
look like evidence, and casual history/evidence chains that still count as analysis.

No labels were changed in this final pass, so the `notes` field in `takemeter_labeled.csv` still
records the original Groq pre-label provenance rather than an override marker.
