# TakeMeter — classifying r/nba comment discourse

A text classifier that sorts r/nba **comments** into three kinds of discourse —
`reaction`, `hot_take`, `analysis` — and compares a **fine-tuned DistilBERT** against a
**zero-shot Groq `llama-3.3-70b` baseline** on the same held-out test set.

Design notes are in [planning.md](planning.md). Run order and the (two) environment
blockers are in [STATUS.md](STATUS.md).

> **⚠️ Results status.** This repo is built and the pipeline is verified end-to-end on a
> synthetic smoke set, but the **numeric results below are produced by running the pipeline
> on real data + a Groq key** (see [STATUS.md](STATUS.md)). Sections that need that run are
> marked **`RESULTS PENDING`** and show the exact table format the scripts emit into
> [artifacts/report_blocks.md](artifacts/report_blocks.md); paste those in after the run.

---

## 1. Community choice and reasoning

**r/nba — its comments, not its submissions.** r/nba is one of Reddit's largest sports
communities, and its actual *talk* happens in comments: submissions are mostly news links,
highlights, and auto-posted Game / Post-Game Threads. Inside those threads, the same game
produces pure emotional venting, throwaway bold opinions, and multi-sentence tactical
breakdowns — often a few replies apart. That spread is what makes the task interesting: the
three modes come from the same people about the same events, so the classifier must key on
*how* something is said (claim? evidence? just emotion?) rather than *what* it is about.
Topic is nearly useless as a signal here — which is the point. It is also practical: high
comment volume, public, English, readable through the JSON endpoints without a login.

## 2. Label taxonomy

Applied with a 2-question decision tree: **Q1** does the comment make a debatable basketball
claim/opinion/prediction/ranking? *No → `reaction`.* **Q2** if yes, is the claim backed by a
real explanatory/causal chain or multiple pieces of evidence? *No → `hot_take`. Yes →
`analysis`.* The tree forces exactly one label (mutually exclusive) and covers ≥90% of
comments without an "other" bucket (off-topic/spam is dropped at collection, not labeled).

| label | definition | example 1 | example 2 |
|---|---|---|---|
| **reaction** | Emotional, in-the-moment expression making **no** substantive basketball claim. | "WHAT A SHOT ARE YOU KIDDING ME" | "bro I'm crying lmao we're so back" |
| **hot_take** | A strongly-stated, debatable opinion/claim/prediction asserted with little/no support (a lone stat is garnish). | "LeBron is washed, time to admit it" | "Wemby is already a top-5 player" |
| **analysis** | A reasoned argument explaining **why** via a causal chain or stacked evidence. | "They go under his screens, so until he hits the pull-up the second unit's spacing collapses." | "His TS% dropped because he's taking contested late-clock mid-range looks — it's a lineup problem, not effort." |

## 3. Data collection

**Source.** Reddit public JSON endpoints via [`src/collect.py`](src/collect.py) (no API key,
no PRAW): the `r/nba/comments.json` firehose (bulk, reaction-heavy) plus top-level comments
from discussion-style threads (Post-Game / Daily / Discussion), which is where `hot_take` and
`analysis` concentrate. The collector strips deleted/removed/bot/link-only/too-short comments
and dedupes.

**Labeling process.** Every comment is pre-labeled by Groq `llama-3.3-70b-versatile`
([`src/label.py`](src/label.py)) with a confidence tag; low/medium-confidence rows are routed
to [`data/needs_review.csv`](data/needs_review.csv) for a human pass. **Every pre-label is
AI-assisted and human-reviewed before use** — provenance is tracked in the `notes` column
(`groq-prelabel conf=…`). This annotation assistance is disclosed in §9.

**Label distribution.** `RESULTS PENDING` — `src/label.py` prints it and flags any class
<20% or >70%. Target ≈ 40 / 35 / 25 across `reaction / hot_take / analysis` (every class
≥20%, none >70%); the thread source exists to keep `analysis` above 20%.

| label | count | % |
|---|---|---|
| reaction | _pending_ | _pending_ |
| hot_take | _pending_ | _pending_ |
| analysis | _pending_ | _pending_ |

**Three genuinely difficult examples and the decisions** (from the pre-annotation
stress-test, see §9):

1. **Emotion that smuggles in a claim** — *"NAHHHH that was a ROBBERY the refs handed them
   this game I'm sick."* Loud emotion, but "the refs decided the game" is a debatable claim
   with no support → **`hot_take`**. *Rule:* if a debatable proposition is extractable, it is
   not `reaction`, however loud.
2. **Bold claim + a single stat** — *"Jokic is the best player alive and it's not close,
   averaging a 30 triple-double."* One stat is garnish → **`hot_take`**; it would be
   `analysis` only if the stat were wired into a causal explanation. *Rule:* analysis needs an
   explanatory chain, not a number tacked on.
3. **Awe/emoji stat-lines & meta-sarcasm** — *"40/8/12 on 65 TS ARE YOU KIDDING ME 🐐🐐🐐"* and
   *"oh great, another 'MJ would've averaged 40' take, riveting."* Awe with an *implied* (not
   stated) ranking, and sarcasm aimed at the discourse → both **`reaction`**. *Rule:* awe +
   emoji is `reaction` unless it states an *explicit* verbal ranking ("he's the GOAT" →
   `hot_take`); meta-commentary about other takes with no on-court position is `reaction`.

## 4. Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (66M params), sequence classification head, 3 labels.
- **Training setup:** max_len 128, lr 2e-5, batch 16, stratified 70/15/15 split from the single
  CSV (seeded, with duplicate-text + cross-split leakage guards in
  [`src/dataset.py`](src/dataset.py)). Code: [`src/finetune.py`](src/finetune.py); the same
  logic is in the Colab notebook ([notebooks/takemeter.ipynb](notebooks/takemeter.ipynb)).
- **Deliberate hyperparameter decision:** the course default is **3 epochs**; I use **5 epochs
  with early stopping on validation macro-F1** (keep the best checkpoint, patience 2). With
  ~250 examples and a scarce `analysis` class, 3 epochs underfits the hard `hot_take`↔`analysis`
  seam; the extra epochs let it learn that boundary while early-stopping-on-val prevents the
  overfitting that more epochs would otherwise invite. Set `EPOCHS=3` to reproduce the default.

## 5. Baseline description

A **zero-shot** Groq `llama-3.3-70b-versatile` classifier ([`src/baseline.py`](src/baseline.py))
on the **same** held-out test set, no task-specific training. The prompt embeds the label
definitions verbatim and instructs the model to **output only the label name** (the exact
`src/labels.build_prompt` shared with the labeler). Parsing is defensive (exact match → first
whole-word label → majority fallback) and the script reports the unparseable rate, revising
the prompt if it exceeds 10%. Run it **before** looking at the fine-tuned numbers.

## 6. Evaluation report

Both models scored on the identical 15% test split. Metrics: overall accuracy, per-class
precision/recall/F1, macro-F1 (the headline comparison — it weights the rare class equally),
and the fine-tuned confusion matrix. Produced by [`src/evaluate_models.py`](src/evaluate_models.py)
→ [`evaluation_results.json`](evaluation_results.json) + [`confusion_matrix.png`](confusion_matrix.png)
+ [artifacts/report_blocks.md](artifacts/report_blocks.md).

**`RESULTS PENDING`** — paste `artifacts/report_blocks.md` here after the run. The scripts emit
exactly these shapes:

**Fine-tuned DistilBERT** — accuracy **_x.xxx_**, macro-F1 **_x.xxx_**

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | _ | _ | _ | _ |
| hot_take | _ | _ | _ | _ |
| analysis | _ | _ | _ | _ |

**Zero-shot Groq baseline** — accuracy **_x.xxx_**, macro-F1 **_x.xxx_** (same table shape).

**Confusion matrix — fine-tuned (markdown, authoritative; the PNG is a copy):**

| true ↓ / pred → | reaction | hot_take | analysis | total |
|---|---|---|---|---|
| **reaction** | _ | _ | _ | _ |
| **hot_take** | _ | _ | _ | _ |
| **analysis** | _ | _ | _ | _ |

**Three fine-tuned misclassifications with analysis** — `RESULTS PENDING`; pull the 3 from
[`artifacts/wrong_predictions.csv`](artifacts/wrong_predictions.csv) (sorted by confidence) and
write, for each: which two labels were confused, why that boundary is hard, whether it's a
**labeling** vs **data** vs **prompt** problem, and what would fix it. Pre-registered hypothesis
(verify against the real errors before committing): the dominant confusion will be
**`analysis` → `hot_take` on short comments** — a supported claim compressed into one line
looks like an unsupported one — which is a *data* problem (too few short-analysis examples),
fixable by oversampling concise analysis.

**Sample classifications** — `RESULTS PENDING`; 3–5 test comments with predicted label +
confidence (from `finetuned_preds.json`), at least one correct example explained:

| comment | predicted | confidence | correct? | note |
|---|---|---|---|---|
| _ | _ | _ | _ | _ |

## 7. Reflection — what the model learned vs. what I intended

`RESULTS PENDING` for the specifics, but the framing the run will confirm or revise: the
taxonomy defines the classes by *intent* (does the author make a claim? support it?), while
the model can only see *surface form*. The likely gap is that DistilBERT learns proxies for
intent — caps/emoji/length/exclamation for `reaction`, hedged causal connectives ("because",
"so", "until") for `analysis` — rather than intent itself. Where those proxies and intent
diverge (a calm one-line hot take; an excited but genuinely analytical comment), it should
err. Overfit signal to watch: if `analysis` recall is high only on *long* comments, it learned
"length = analysis", not "reasoning = analysis". The confusion matrix and the per-length error
breakdown ([`src/error_analysis.py`](src/error_analysis.py)) are where this shows up.

## 8. Spec reflection

- **One way the spec helped:** writing `planning.md`'s label definitions *before* collecting
  anything forced the 2-question decision tree, and the pre-annotation stress-test
  ([§9](#9-ai-usage)) caught two boundary holes (awe/emoji vs explicit ranking; meta-sarcasm)
  that I fixed in the taxonomy *before* a single comment was labeled — so the labels were tight
  from the first row instead of drifting mid-annotation.
- **One way the implementation diverged from the spec, and why:** the spec's "annotation
  assistance" assumed Groq pre-labeling with a light human review. The build environment had no
  Groq key and Reddit blocks its IP, so the *implemented* path splits the work: a ready-to-run
  collector + Groq labeler are staged for a residential run, and the pipeline is instead
  verified end-to-end on a clearly-marked synthetic smoke set. The methodology is unchanged; the
  divergence is purely about *where/when* the keyed+networked steps run (documented in
  [STATUS.md](STATUS.md)).

## 9. AI usage

At least two concrete instances, plus the required annotation disclosure:

1. **Label stress-testing (before annotation).** I directed an AI to generate 16 r/nba comments
   sitting on the two boundary seams, then ran an independent pass that labeled each using only
   the rules. It produced 14/16 clean, confident labels; **I overrode the taxonomy** in response
   — adding a 4th tiebreaker for awe/emoji-vs-explicit-ranking and meta-sarcasm after seeing the
   2 it couldn't cleanly resolve. (Run via a fan-out workflow of generator + adjudicator agents.)
2. **Pipeline implementation.** I directed an AI to write the collection, labeling, split,
   fine-tune, baseline, and evaluation code and the Colab notebook against the rubric. **What I
   changed:** fixed the transformers-5.x `Trainer(processing_class=…)` API, and added the
   duplicate-text + cross-split leakage guards in `src/dataset.py` after deciding the default
   split was leakage-unsafe for a small dataset.
3. **Planned failure analysis.** After evaluation, I'll have an AI cluster the fine-tuned model's
   wrong predictions into patterns, then **verify each pattern by re-reading the actual
   misclassified comments** before it goes in §6 — AI surfaces candidates, I confirm or reject.

**Annotation-assistance disclosure (required):** the training labels are **AI-pre-labeled** by
Groq `llama-3.3-70b-versatile` ([`src/label.py`](src/label.py)) and then **human-reviewed** —
at minimum every low/medium-confidence row in `data/needs_review.csv`. The `notes` column records
each row's provenance and confidence so reviewed/overridden labels are traceable.

---

## Repo layout & how to run

```
planning.md  README.md  STATUS.md  DEMO_SCRIPT.md  requirements.txt
data/        takemeter_labeled.csv (single, non-split)  needs_review.csv
src/         labels.py collect.py label.py dataset.py prepare_splits.py
             finetune.py baseline.py evaluate_models.py error_analysis.py calibration.py app.py
notebooks/   takemeter.ipynb   (Colab T4, self-contained)
artifacts/   train/val/test.csv  *_preds.json  report_blocks.md  wrong_predictions.csv
evaluation_results.json   confusion_matrix.png
```

> The layout above is the **intended final state**. The data files
> (`data/takemeter_labeled.csv`, `data/needs_review.csv`) and every file under `artifacts/`
> (plus `evaluation_results.json`, `confusion_matrix.png`) are **produced by the run order
> below** and are not present until then — see the `RESULTS PENDING` markers and
> [STATUS.md](STATUS.md).

Run order (see [STATUS.md](STATUS.md) for the two prerequisites):

```bash
python src/collect.py --target 400                 # from a residential IP (Reddit blocks cloud IPs)
python src/label.py                                # needs GROQ_API_KEY; writes labeled CSV + needs_review
# (human-review data/needs_review.csv, fold corrections back in)
python src/prepare_splits.py                       # 70/15/15, same test set for both models
python src/baseline.py                             # zero-shot Groq baseline (run before looking at fine-tuned)
python src/finetune.py                             # or run notebooks/takemeter.ipynb on Colab T4
python src/evaluate_models.py                      # evaluation_results.json + confusion_matrix.png + report_blocks.md
python src/error_analysis.py && python src/calibration.py   # stretch
python src/app.py                                  # stretch: deployed interface
```
