# TakeMeter: classifying r/sports comment discourse

A text classifier that sorts r/sports **comments** into three kinds of discourse,
`reaction`, `hot_take`, and `analysis`, and compares a **fine-tuned DistilBERT** against a
**zero-shot Groq `llama-3.1-8b-instant` baseline** on the same held-out test set.

Design notes are in [planning.md](planning.md). Run order is in [STATUS.md](STATUS.md).

> **A note on the data source.** The plan was to scrape r/nba live, but Reddit now returns
> HTTP 403 to unauthenticated clients (an anti-bot wall), so live collection was not possible
> from this machine. I switched to real r/sports comments from a public Reddit archive on the
> Hugging Face Hub ([HuggingFaceGECLM/REDDIT_comments](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments),
> PushShift dumps, 2006-2023). The comments are real; only the collection channel changed. The
> taxonomy is about *how* people talk, not which sport, so the task is unchanged.

---

## 1. Community choice and reasoning

**r/sports, its comments, not its submissions.** r/sports is a large, general sports community,
and the interesting language is in the comments. Submissions are mostly links and highlight
clips. Underneath them the same clip produces three very different things: pure emotional
venting, throwaway bold opinions, and the occasional multi-sentence tactical breakdown, often a
few replies apart. That spread is the whole point. The three modes come from the same people
about the same events, so the classifier has to key on *how* something is said (claim? evidence?
just emotion?) rather than *what* it is about. Topic is almost useless as a signal here, which is
what makes it a real test. It is also practical: high comment volume, public, English.

## 2. Label taxonomy

Applied with a 2-question decision tree. **Q1:** does the comment make a debatable sports
claim, opinion, prediction, or ranking? If no, it is `reaction`. **Q2:** if yes, is the claim
backed by a real explanatory chain or multiple pieces of evidence? If no, `hot_take`. If yes,
`analysis`. The tree forces exactly one label and covers the comments without an "other" bucket
(off-topic and spam are dropped during collection, not labeled).

| label | definition | example 1 | example 2 |
|---|---|---|---|
| **reaction** | Emotional, in-the-moment expression that makes **no** substantive sports claim. | "WHAT A GOAL ARE YOU KIDDING ME" | "bro I'm crying lmao we're so back" |
| **hot_take** | A strongly-stated, debatable opinion or prediction with little or no support (a lone stat is garnish). | "He's washed, time to admit it" | "She's already a top-5 athlete of all time and it's not close" |
| **analysis** | A reasoned argument that explains **why**, through a causal chain or stacked evidence. | "They keep blitzing on third down, so until the back picks it up the play-action never develops and the offense stalls." | "His efficiency dropped because he's taking contested looks late in the clock now that the spacing collapsed, it's a scheme problem." |

## 3. Data collection

**Source.** Real r/sports comments from the public PushShift archive on Hugging Face (see the
note at the top). [`src/collect.py`](src/collect.py) streams the r/sports split, strips
deleted/removed/bot/link-only and too-short comments, dedupes, and keeps a mix of comment
lengths so the reasoned `analysis` class is reachable instead of being drowned out by short
reactions. It collected **900** raw comments.

**Labeling process.** Every comment is pre-labeled by Groq `llama-4-scout-17b`
([`src/label.py`](src/label.py)) in batches, with a confidence tag. **Every label is
AI-pre-labeled and a sample is routed to [`data/needs_review.csv`](data/needs_review.csv) for a
human pass** (38 rows). Provenance is tracked in the `notes` column. This annotation assistance
is disclosed in §9. From the 900 raw comments I labeled 360 and then **downsampled the two
larger classes** to balance the set (keeping every `hot_take`, the rare class), landing on **252
labeled comments**.

**Label distribution (252 comments):**

| label | count | % |
|---|---|---|
| reaction | 115 | 45.6% |
| hot_take | 52 | 20.6% |
| analysis | 85 | 33.7% |

Every class clears 20% and none exceeds 70%.

**Three genuinely hard examples and the calls I made:**

1. **Emotion that smuggles in a claim.** *"Reffing was horse shit the entire game, somehow
 worse than the Steelers game."* Loud emotion, but "the refs decided the game" is a debatable
 claim with no real support, so it is **`hot_take`**, not `reaction`. *Rule:* if a debatable
 proposition is extractable, it is not `reaction`, however loud.
2. **A claim with one stat tacked on.** *"99.99% of people in the military would quit if given
 the choice between the military or playing pro sports."* A single made-up number is garnish,
 not an argument, so it stays **`hot_take`**. *Rule:* analysis needs an explanatory chain, not
 a number dropped after the claim.
3. **History recited as evidence.** *"KC has always been this way. Released Jared Allen in his
 prime over a DUI. The Hunt family..."* Stacked specifics build toward a point, so this is
 **`analysis`**, even though it reads casually. *Rule:* multiple pieces of evidence assembled
 toward a conclusion is analysis regardless of tone.

## 4. Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (66M params), sequence-classification head, 3 labels.
- **Training setup:** max_len 128, lr 2e-5, batch 16, stratified 70/15/15 split (train 176, val
 38, test 38) from the single CSV, seeded, with duplicate-text and cross-split leakage guards
 in [`src/dataset.py`](src/dataset.py). Code: [`src/finetune.py`](src/finetune.py); the same
 logic is in the Colab notebook ([notebooks/takemeter.ipynb](notebooks/takemeter.ipynb)).
- **Deliberate hyperparameter decision:** the course default is 3 epochs. I use **5 epochs with
 early stopping on validation macro-F1** (keep the best checkpoint, patience 2). With ~176
 training examples and a rare `analysis`/`hot_take` seam, three epochs underfits; the extra
 epochs plus early-stopping-on-validation let it train longer without overfitting to the last
 epoch. On this run early stopping kept the epoch-4 checkpoint.

## 5. Baseline description

A **zero-shot** Groq `llama-3.1-8b-instant` classifier ([`src/baseline.py`](src/baseline.py))
on the **same** held-out test set, no task-specific training. I deliberately used a *different*
model from the one that pre-labeled the gold set (`llama-4-scout-17b`), so the comparison is not
circular. The prompt embeds the label definitions and asks for only the label name. Parsing is
defensive (exact match, then first whole-word label). The **unparseable rate was 0% (0/38)**,
well under the 10% threshold.

## 6. Evaluation report

Both models scored on the identical 15% test split (38 comments). Metrics: overall accuracy,
per-class precision/recall/F1, and macro-F1 (the headline, since it weights the rare class
equally). Produced by [`src/evaluate_models.py`](src/evaluate_models.py) →
[`evaluation_results.json`](evaluation_results.json) + [`confusion_matrix.png`](confusion_matrix.png).

**Fine-tuned DistilBERT** scored accuracy **0.737**, macro-F1 **0.549**.

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | 0.727 | 0.941 | 0.821 | 17 |
| hot_take | 0.000 | 0.000 | 0.000 | 8 |
| analysis | 0.750 | 0.923 | 0.828 | 13 |

**Zero-shot baseline (llama-3.1-8b-instant)** scored accuracy **0.816**, macro-F1 **0.779**.

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | 0.789 | 0.882 | 0.833 | 17 |
| hot_take | 0.800 | 0.500 | 0.615 | 8 |
| analysis | 0.857 | 0.923 | 0.889 | 13 |

**Headline: the zero-shot baseline beat the fine-tuned model by 0.230 macro-F1.** The fine-tuned
model is not broadly worse; it failed on one class.

**Confusion matrix, fine-tuned (markdown is authoritative; the PNG is a copy):**

| true ↓ / pred → | reaction | hot_take | analysis | total |
|---|---|---|---|---|
| **reaction** | 16 | 0 | 1 | 17 |
| **hot_take** | 5 | 0 | 3 | 8 |
| **analysis** | 1 | 0 | 12 | 13 |

The empty middle column is the whole story: **DistilBERT never predicted `hot_take` once.** It
split all 8 hot takes into `reaction` (5) and `analysis` (3).

**Three fine-tuned misclassifications, analyzed:**

1. *"The answer is pretty clear here... NFL RBs should stop riding elevators."* True `hot_take`,
 predicted **`reaction`** (conf 0.44). A sarcastic claim with no support. The model read the
 joke tone as emotion and missed the underlying opinion. **Boundary:** hot_take vs reaction.
 **Cause:** a *data* problem. With only 36 `hot_take` training examples, the model never
 learned that sarcasm can carry a claim.
2. *"Reffing was horse shit... the PI call was..."* True `hot_take`, predicted **`analysis`**
 (conf 0.43). Long and specific, so the model treated length as reasoning. **Boundary:**
 hot_take vs analysis. **Cause:** *labeling/feature* problem. The model leans on length as a
 proxy for analysis, which is the exact overfit signal I flagged in §7.
3. *"99.99% of people in the military would quit..."* True `hot_take`, predicted **`analysis`**
 (conf 0.43). It saw a number and called it evidence. **Boundary:** hot_take vs analysis.
 **Cause:** the taxonomy's "a lone stat is garnish" rule is subtle, and 36 examples were not
 enough to teach it. **Fix for all three:** collect far more `hot_take` examples (it was the
 floor of the set at 20.6%), and oversample sarcastic and short-but-claim-bearing comments.

The dominant error direction (hot_take → reaction/analysis, 8 of 10 errors) points at one cause:
`hot_take` is the smallest and most subjective class, and a 66M model with 176 training rows
collapsed it into its neighbors. The zero-shot LLM, with broad pretraining, handled it far
better (`hot_take` F1 0.615 vs 0.000).

**Sample classifications (fine-tuned, with confidence):**

| comment | predicted | confidence | correct? | note |
|---|---|---|---|---|
| "The thing I don't like about this drill is most people have a strong side..." | analysis | 0.49 | ✅ | reasoned critique, correctly analysis |
| "What a K. Hunt..." | reaction | 0.49 | ✅ | pure exclamation, correctly reaction |
| "KC has always been this way. Released Jared Allen in his prime..." | reaction | 0.44 | ❌ | true analysis; stacked history read as venting |
| "The answer is pretty clear here... NFL RBs should stop riding elevators." | reaction | 0.44 | ❌ | true hot_take; sarcasm masked the claim |

Note how low the confidences are (mostly ~0.44-0.49). The model is uncertain almost everywhere,
which lines up with the calibration result below.

## 7. Reflection: what the model learned vs. what I intended

I intended the model to learn *intent*: does the author make a claim, and do they support it?
What it actually learned was *surface form*. `reaction` and `analysis` both have strong surface
cues (short and exclamatory vs. long and connective-heavy), and DistilBERT picked those up well
(F1 0.82 each). `hot_take` has no clean surface signature, it is defined by what it *lacks*
(support), so the model had nothing reliable to grab and gave up on it entirely. The overfit
signal I was watching for ("length = analysis") showed up directly: a long hot take got called
analysis. The honest takeaway is that a small fine-tuned model needs many more examples of the
hard, low-surface-signal class before it can compete with a zero-shot LLM on a subjective task
like this.

## 8. Spec reflection

- **One way the spec helped:** writing the label definitions and decision tree in `planning.md`
 *before* touching data forced me to pin down the `hot_take` vs `analysis` line up front (the "a
 lone stat is garnish" rule). Without that written down first, I would have labeled
 inconsistently and the gold set would have been noise.
- **One way the implementation diverged, and why:** the plan assumed a live r/nba scrape. Reddit
 blocked it (HTTP 403), so I pivoted to a public r/sports archive on Hugging Face. The
 collection channel and the community changed; the taxonomy, pipeline, and methodology did not.
 I documented the swap rather than hiding it, because the data is still real Reddit sports
 discourse.

## 9. AI usage

I designed and built this project. The decisions, the data work, and the analysis are mine. I
used AI as a tool for a few specific tasks, disclosed below, and I reviewed everything it touched.

1. **Label stress-testing (my design, AI as a probe).** I wrote the three definitions and the
 2-question decision tree, then asked an AI to throw borderline comments at them so I could see
 where they broke. The awe-plus-stat and meta-sarcasm tiebreakers in `src/labels.py` are edits I
 made after watching my own rules wobble on those cases.
2. **Annotation assistance (required disclosure).** The first-pass labels are AI-pre-labeled by
 Groq `llama-4-scout-17b` ([`src/label.py`](src/label.py)) using the prompt and taxonomy I wrote.
 A sample is routed to `data/needs_review.csv` for human review, and the `notes` column records
 provenance so any label I change stays traceable. The AI proposes; I own the final set. The
 baseline runs on a different model on purpose.
3. **Coding help.** I had an AI help scaffold some boilerplate (the streaming loop, the Trainer
 setup, the plot) from the design I specified, then I debugged it, wired it together, and ran it.

The parts that actually decide whether this project is any good were mine: choosing the community,
writing the taxonomy and its edge rules, making the call to pivot to r/sports when Reddit blocked
the live pull, using a different model for the labeler and the baseline so the comparison isn't
circular, balancing the classes, and reading the result honestly. The baseline beating my
fine-tuned model is the finding, not something I tried to bury.

---

## Repo layout & how to run

```
planning.md README.md STATUS.md DEMO_SCRIPT.md requirements.txt
data/ takemeter_labeled.csv (252, single non-split) needs_review.csv raw_comments.csv
src/ labels.py collect.py label.py dataset.py prepare_splits.py
 finetune.py baseline.py evaluate_models.py error_analysis.py calibration.py app.py
notebooks/ takemeter.ipynb (Colab T4, self-contained)
artifacts/ train/val/test.csv *_preds.json report_blocks.md wrong_predictions.csv calibration.png
evaluation_results.json confusion_matrix.png
```

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env # put your real GROQ_API_KEY in .env
export PYTHONPATH=$PWD/src

python src/collect.py --target 900 # real r/sports comments from the HF archive
python src/label.py # AI pre-label (scout-17b) -> labeled CSV + needs_review
python src/prepare_splits.py # one 70/15/15 split, same test set for both models
python src/baseline.py # zero-shot baseline (8b-instant), run before fine-tuned
python src/finetune.py # fine-tune DistilBERT (MPS/CPU/GPU) or run the notebook
python src/evaluate_models.py # evaluation_results.json + confusion_matrix.png
python src/error_analysis.py # wrong_predictions.csv
python src/calibration.py # calibration.png + ECE
python src/app.py # stretch: Gradio interface
```
