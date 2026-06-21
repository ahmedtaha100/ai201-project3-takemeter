# TakeMeter — Demo Video Script (3–5 min)

Record after the pipeline has run on real data (see [STATUS.md](STATUS.md)). The rubric
requires four moments: (1) 3–5 posts classified by the **fine-tuned** model with **label +
confidence visible**, (2) one **correct** prediction narrated, (3) one **incorrect**
prediction narrated with *why*, (4) a brief walkthrough of the evaluation report.

Launch the interface first: `python src/app.py` (loads `models/distilbert-takemeter`).

---

### 0:00–0:30 — What it is
"This is TakeMeter. It classifies r/nba **comments** into three kinds of discourse:
**reaction** — pure emotion, no claim; **hot_take** — a bold opinion with no real support;
and **analysis** — an argument that actually explains *why*. The interesting part is that
these come from the same fans about the same games, so the model can't lean on topic — it
has to pick up on whether there's a claim and whether it's backed up."

### 0:30–2:00 — Live classifications (show label + confidence on screen)
Paste these into the app one at a time and read the predicted label + confidence aloud:

1. `WHAT A SHOT ARE YOU KIDDING ME` → expect **reaction** (high conf).
2. `Wemby is already a top-5 player, it's not close` → expect **hot_take**.
3. `They trap Brunson in the PnR and nobody else creates, so the half-court offense just dies` → expect **analysis**.
4. *(pick one real test comment your model got right)* — your **correct** example.
5. *(pick one real test comment your model got wrong)* — your **incorrect** example.

### 2:00–2:45 — Narrate ONE correct prediction
Use sample #4 (or a row from `finetuned_preds.json` where `true == pred`). Say:
"The model called this **\<label\>** at **\<confidence\>**, which is right — and you can see
*why*: \<the explicit claim / the 'because…so…' chain / the pure emotion\>. That's the
signal the taxonomy is built on."

### 2:45–3:30 — Narrate ONE incorrect prediction (the important moment)
Use a real miss from `artifacts/wrong_predictions.csv`. Say:
"Here it predicted **\<pred\>** but the true label is **\<true\>**. It's wrong because
\<e.g. the comment is a one-line *supported* claim, so it reads like an unsupported hot take —
the model is keying on length, not on the reasoning\>. This is the `analysis`↔`hot_take` seam,
the hardest boundary — and it's a **data** problem more than a model problem: too few short
analysis examples." *(Match this to your actual error pattern from §6 of the README.)*

### 3:30–4:30 — Evaluation report walkthrough
Show the README §6 / `evaluation_results.json`:
"Fine-tuned DistilBERT got **\<acc\> / macro-F1 \<f1\>**; the zero-shot Groq 70B baseline got
**\<acc\> / \<f1\>**. The fine-tuned model \<beat / didn't beat\> a model 1000× its size by
**\<Δ macro-F1\>**. The confusion matrix shows errors concentrated on \<the hot_take/analysis
seam\>, which is exactly where the taxonomy said the hard boundary was — not smeared randomly,
which is the sign it learned something real."

### 4:30–5:00 — Close
"So: a 66M-param model fine-tuned on \<N\> human-reviewed Reddit comments, evaluated against a
70B zero-shot baseline on the same test set, with the errors landing on the genuinely
ambiguous boundary. Thanks for watching."

---

**Checklist before you hit record:** model trained ✓, app launches ✓, you've picked one real
correct + one real incorrect example ✓, README §6 has the real numbers ✓, label + confidence
are visible on screen for every classification ✓.
