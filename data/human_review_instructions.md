# Human Review Record

The rubric requires every LLM pre-label to be reviewed and corrected by a human.

Final pass status: **complete**.

- Reviewer: project author.
- Rows reviewed: 252 / 252.
- Review outcome: 240 current labels accepted; 12 labels corrected after the full audit.
- Label corrections applied: 12.

The completed sheet is `data/human_review_all_252.csv`.

Review method:

1. Read each `text` without changing the order.
2. Compare `current_label` against the taxonomy in `README.md` / `planning.md`.
3. Fill `human_label` with exactly one of `reaction`, `hot_take`, or `analysis`.
4. Set `change_needed` to `yes` if the human label differs from `current_label`, else `no`.
5. Add a short `review_notes` entry for any changed or borderline row.

Because labels changed, the train/validation/test split labels, fine-tuned predictions,
evaluation, error analysis, and calibration were regenerated locally. If any label is changed
later, rerun:

```bash
python src/prepare_splits.py
python src/baseline.py
python src/finetune.py
python src/evaluate_models.py
python src/error_analysis.py
python src/calibration.py
```
