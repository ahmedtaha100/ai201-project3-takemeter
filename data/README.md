# data/

These files are produced by the pipeline (see ../STATUS.md), not committed empty:

- `raw_comments.csv` — output of `src/collect.py` (run from a residential IP).
- `takemeter_labeled.csv` — the single, non-split labeled dataset (`text,label,notes`),
  output of `src/label.py` after human review. **This is the submission dataset.**
- `needs_review.csv` — low/medium-confidence rows for the human annotation pass.

`_smoketest/` (gitignored) holds synthetic verification data only — never the deliverable.
