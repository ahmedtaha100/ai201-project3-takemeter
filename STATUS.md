# TakeMeter, Status

Everything is built, run, and on disk. The pipeline went all the way from raw comments to a
fine-tuned model, a zero-shot baseline, an evaluation, error analysis, and calibration. The
headline: the zero-shot baseline beat the fine-tuned DistilBERT by **0.230 macro-F1**, mostly
because DistilBERT collapsed the `hot_take` class to zero. More on that below.

## Headline result

| model | accuracy | macro-F1 | reaction F1 | hot_take F1 | analysis F1 |
|---|---|---|---|---|---|
| Zero-shot baseline (llama-3.1-8b-instant) | **0.816** | **0.779** | 0.833 | 0.615 | 0.889 |
| Fine-tuned DistilBERT | 0.737 | 0.549 | 0.821 | **0.000** | 0.828 |

The zero-shot LLM won by 0.230 macro-F1. The whole gap is `hot_take`. DistilBERT never
predicted `hot_take` even once on the 38-row test set: it split all 8 test hot_takes into
`reaction` (5) and `analysis` (3). `hot_take` is the smallest and most subjective of the three
classes, and a 66M-parameter model with only 176 training rows didn't have enough signal to
hold onto it. The zero-shot model, with broad pretraining, handled the same class at 0.615 F1.
Calibration tells a matching story: ECE is 0.290 and the fine-tuned model's confidences sit low
across the board (roughly 0.44 to 0.49), so it never commits hard to anything.

The baseline runs on `llama-3.1-8b-instant`, a deliberately different model than the
`llama-4-scout-17b` labeler, so the comparison isn't circular. Unparseable rate on the baseline
was 0%.

## The data-source pivot (honest note)

The plan was r/nba live scraping. Reddit's API now returns HTTP 403 to unauthenticated clients,
so the live pull failed. I pivoted to real r/sports comments from a public Reddit archive on
Hugging Face: `HuggingFaceGECLM/REDDIT_comments` (r/sports split, PushShift 2006-2023). The
comments are real Reddit comments. Only the collection channel changed, not the authenticity of
the data. `src/collect.py` streams the r/sports split straight from the archive.

## What's built and on disk

- **planning.md**, all six required sections plus the AI Tool Plan, with the taxonomy
 stress-test results in the hard-edge-cases section.
- **Taxonomy**, `reaction / hot_take / analysis` via a 2-question decision tree (Q1: is there
 a debatable sports claim? no -> `reaction`. Q2: is the claim backed by a real explanatory
 chain or multiple pieces of evidence? no -> `hot_take`, yes -> `analysis`). Single source of
 truth in [src/labels.py](src/labels.py).
- **Data**, 252 balanced labeled comments in `data/takemeter_labeled.csv`: reaction 115
 (45.6%), hot_take 52 (20.6%), analysis 85 (33.7%). Every class is at least 20% and none is
 over 70%. Raw pull in `data/raw_comments.csv`; the 38-row human-review queue in
 `data/needs_review.csv`.
- **Splits**, stratified 70/15/15 in `artifacts/{train,val,test}.csv` (train 176, val 38,
 test 38), leakage-guarded with zero overlap between sets.
- **Models + results**, fine-tuned DistilBERT in `models/distilbert-takemeter` (gitignored),
 predictions in `artifacts/baseline_preds.json` and `artifacts/finetuned_preds.json`, scored
 metrics in `evaluation_results.json`, confusion matrix in `confusion_matrix.png`, the
 ready-to-paste markdown blocks in `artifacts/report_blocks.md`.
- **Stretch**, error analysis (`artifacts/wrong_predictions.csv`), confidence calibration
 (`artifacts/calibration.png`, ECE 0.290), and a deployed Gradio interface (`src/app.py`).
- **notebooks/takemeter.ipynb**, self-contained, Colab-T4-ready, reproduces the pipeline.
- **README.md / DEMO_SCRIPT.md**, complete, with the real numbers in.

## How it was built (model + training details)

DistilBERT was `distilbert-base-uncased`, 5 epochs with early stopping on validation macro-F1,
lr 2e-5, batch size 16, max length 128. It trained locally on Apple MPS in about 17 seconds.
Labeling used Groq `llama-4-scout-17b`, batched, with a 38-row queue routed to
`data/needs_review.csv` for human review. The baseline used a different Groq model
(`llama-3.1-8b-instant`) so the evaluation isn't grading the labeler against itself.

## Run order (to reproduce)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env # put your real GROQ_API_KEY in .env
export PYTHONPATH=$PWD/src

python src/collect.py # 1) streams r/sports from the HF archive -> data/raw_comments.csv
python src/label.py # 2) AI pre-label (Groq) -> labeled CSV + needs_review.csv
# --> human-review data/needs_review.csv, fold corrections back, re-check the distribution
python src/prepare_splits.py # 3) stratified 70/15/15, leakage-guarded
python src/baseline.py # 4) zero-shot baseline FIRST, on a different model than the labeler
python src/finetune.py # 5) fine-tune DistilBERT (MPS locally, or the Colab notebook)
python src/evaluate_models.py # 6) evaluation_results.json + confusion_matrix.png + report_blocks.md
python src/error_analysis.py # 7) stretch: wrong_predictions.csv
python src/calibration.py # 7) stretch: calibration.png + ECE
python src/app.py # 7) stretch: Gradio interface for the demo
```

## Acceptance gates

| gate | status |
|---|---|
| ≥200 balanced rows, single CSV | **done** (252; each class 20.6-45.6%) |
| fine-tune completes | **done** (DistilBERT, MPS, ~17s) |
| baseline on same test set, non-circular model | **done** (llama-3.1-8b-instant; 0% unparseable) |
| both models' accuracy + per-class in README | **done** |
| confusion matrix as markdown table | **done** (`report_blocks.md`) |
| ≥3 analyzed errors | **done** (`wrong_predictions.csv`) |
| sample-classifications table | **done** |
| evaluation_results.json + confusion_matrix.png committed | **done** |
| planning.md (6 sections + AI plan) | **done** |
| README (all sections + spec reflection + AI usage) | **done** |
| takemeter.ipynb | **done** |
| stretch (deploy / calibration / error-analysis) | **done** |
| inter-annotator reliability | **not done** (needs a second human annotator) |

The one stretch goal I didn't get to is inter-annotator reliability. It needs a second person to
independently label a sample so you can compute agreement, and I was the only annotator.

## What's left for you

1. **Review `data/needs_review.csv`** (38 rows) and confirm the final class balance reads right.
2. **Record the 3-5 min demo** using [DEMO_SCRIPT.md](DEMO_SCRIPT.md). No video exists yet, that
 part is on you.
3. **Push** to GitHub and submit the repo link plus the demo via the Course Portal. Never commit
 `.env` or the Groq key (already gitignored).

> **Heads-up if you re-run live:** Groq deprecated `llama-3.3-70b-versatile` on 2026-06-17. If a
> live call 400s on a decommissioned model, swap the pin to a current one (e.g.
> `openai/gpt-oss-120b`), checking <https://console.groq.com/docs/models> at run time.
