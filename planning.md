# TakeMeter — Planning

*Working notes, written before any data was collected. The polished writeup lives in
[README.md](README.md); this is the design thinking behind it.*

A classifier that sorts r/nba **comments** into the kind of discourse they are:
`reaction`, `hot_take`, or `analysis`.

---

## 1. Community

**r/nba** — specifically its **comments**, not its submissions.

r/nba is one of the largest sports communities on Reddit, and almost all of its actual
*talk* happens in comments: submissions are mostly news links, highlight clips, and
auto-posted Game/Post-Game Threads. Inside those threads the discourse is genuinely
varied — the same game produces pure emotional venting, throwaway bold opinions, and
multi-sentence tactical breakdowns, often three replies apart. That spread is what makes
the classification task interesting: the three modes are produced by the same people about
the same events, so the classifier has to key on *how* something is said (claim? evidence?
just emotion?) rather than *what* it is about. Topic is nearly useless as a signal here,
which is exactly the point.

It is also a practical fit: high comment volume (easy to clear 250), public, English, and
no login required to read it via the JSON endpoints.

---

## 2. Labels

Three labels, applied with a 2-question decision tree:

> **Q1.** Does the comment make a debatable basketball claim / opinion / prediction /
> ranking? **No → `reaction`.**
> **Q2.** If yes, is the claim backed by a real explanatory/causal chain or multiple pieces
> of evidence (a lone stat is garnish, not an argument)? **No → `hot_take`. Yes → `analysis`.**

- **`reaction`** — an emotional, in-the-moment expression responding to a play/game/news
  that makes no substantive basketball claim.
  - *"WHAT A SHOT ARE YOU KIDDING ME"*
  - *"bro I'm crying lmao we're so back"*
- **`hot_take`** — a strongly-stated, debatable basketball opinion/claim/prediction/ranking
  asserted with little or no supporting reasoning.
  - *"LeBron is washed, time to admit it"*
  - *"Wemby is already a top-5 player in the league"*
- **`analysis`** — a reasoned, evidence-based argument that explains *why*, via a causal
  chain or stacked evidence (stats, scheme, matchups, film).
  - *"Teams go under every screen he sets, so until he punishes it with the pull-up the
    second unit's spacing collapses and the drive-and-kick dies."*
  - *"His TS% dropped because he's taking more contested late-clock mid-range looks now that
    the corner shooters aren't hitting — it's a lineup problem, not effort."*

The labels are **mutually exclusive** (the tree forces exactly one) and **exhaustive**
enough that an "other" bucket is not needed: every basketball comment is either emotion,
an unsupported claim, or a supported claim. Off-topic/spam is dropped at collection, not
given a label.

---

## 3. Hard edge cases

The fuzziness is all on two seams: **reaction ↔ hot_take** (does an emotional comment also
make a claim?) and **hot_take ↔ analysis** (is there real reasoning, or just a stat?).

I stress-tested the definitions *before* annotating by having an AI generate 16 deliberately
borderline comments and a separate pass label each using only the rules (see
[§ AI Tool Plan](#ai-tool-plan)). 14/16 were cleanly and confidently labeled; the two that
weren't became the tiebreakers below. Three real difficult cases and the decisions:

1. **Emotional outburst that smuggles in a claim** — *"NAHHHH man that was a ROBBERY the
   refs handed them this game I'm sick."* Surface is pure emotion, but "the refs decided the
   game" is a debatable claim with no support → **`hot_take`**. **Rule:** if you can extract
   a debatable basketball proposition, it is *not* `reaction`, however loud it is.
2. **Bold claim with a single stat attached** — *"Jokic is the best player alive and it's
   not close, dude is averaging a 30 triple-double."* One stat is garnish, not an argument →
   **`hot_take`**. The same claim *becomes* `analysis` only when the stat is wired into a
   causal explanation. **Rule:** analysis needs an explanatory chain; remove the bold claim
   and there must still be a substantive "why" left over.
3. **Awe / emoji stat-lines and meta-sarcasm** — *"40/8/12 on 65 TS ARE YOU KIDDING ME 🐐🐐🐐"*
   and *"oh great, another 'MJ would've averaged 40' take, riveting stuff."* The first is awe
   with an *implied* ranking; the second mocks the discourse, not an on-court position. Both →
   **`reaction`**. **Rule (added after the stress test):** awe + stat + emoji is `reaction`
   unless it states an *explicit verbal* ranking ("he's the GOAT" → `hot_take`); sarcasm
   aimed at other takes/the discourse, with no on-court claim, is `reaction`.

During annotation, any comment that gives genuine pause goes to `data/needs_review.csv` with
the model's tentative label and confidence, and gets a human decision recorded there.

---

## 4. Data collection plan

**Where:** Reddit public JSON endpoints (`src/collect.py`, no API key, no PRAW):
- `r/nba/comments.json` — the recent-comment firehose across the sub (bulk; reaction-heavy).
- Top-level comments from discussion-style threads (Post Game Threads, Daily/Discussion
  threads) — where hot takes and analysis concentrate.

**How many:** target ~400 *raw* comments so there is slack to filter (drop
deleted/removed/bot/too-short/link-only, dedupe) and still land **≥250 labeled**.

**Per label:** the natural mix on r/nba skews toward `reaction`, with `analysis` scarce.
Plan: aim for a rough **40 / 35 / 25** split across `reaction / hot_take / analysis` — every
class **≥ 20%**, none **> 70%**. The thread-based source exists specifically to lift
`analysis` and `hot_take` above the firehose's reaction bias.

**If a label is underrepresented after 200:** keep collecting, but selectively — pull more
Post-Game/Daily threads (analysis-rich) and cap further `reaction` intake. `src/label.py`
prints the live distribution and flags any class under 20% or over 70% so this is a
checkable loop, not a guess. Balancing happens by collecting more of the thin class, never
by relabeling to hit a quota.

---

## 5. Evaluation metrics

Accuracy alone is not enough here for two reasons: the classes are **imbalanced** (a
model that always guesses `reaction` could score deceptively well), and the **cost of
confusions is uneven** (mixing up `hot_take`/`analysis` is the interesting failure;
`reaction` is the easy class). So:

- **Overall accuracy** — one headline number, on the held-out test set, for both models.
- **Per-class precision, recall, and F1** — this is the real report. Recall on `analysis`
  (the rare, valuable class) and the precision/recall trade on the `hot_take`↔`analysis`
  seam are what actually tell me whether the model learned the distinction or just learned
  the base rates.
- **Confusion matrix** (3×3, written as a markdown table) — shows *where* the errors go,
  which is the whole point of a discourse classifier: is it collapsing `analysis` into
  `hot_take`, or scattering randomly?
- **Macro-F1** as the single comparison number between the fine-tuned model and the
  baseline, because it weights the rare class equally and is not flattered by imbalance.

Both models are evaluated on the **same** held-out 15% test split.

---

## 6. Definition of success

A discourse classifier is genuinely useful (e.g. to auto-tag a thread, or surface the
`analysis` comments worth reading) if it clears the base-rate trap and handles the rare
class. Concrete thresholds:

- **Primary:** fine-tuned **macro-F1 ≥ 0.70** on the test set, and **beats the zero-shot
  Groq baseline's macro-F1**. (Beating a 70B zero-shot model with a fine-tuned 66M-param
  DistilBERT is the result that would justify the fine-tune.)
- **Per-class floor:** **`analysis` recall ≥ 0.60** — the rare, valuable class is the one
  worth protecting; a model that can't find analysis isn't useful no matter its accuracy.
- **Sanity ceiling:** if accuracy is **> 0.95**, treat it as a red flag (test leakage or
  labels too easy) and re-check, not a win.

"Good enough for deployment" = primary + per-class floor both met, with the confusion
matrix showing errors concentrated on the genuinely hard `hot_take`↔`analysis` seam rather
than smeared everywhere.

---

## AI Tool Plan

An explicit decision on each of the three required uses:

- **Label stress-testing — DONE, before annotating.** I gave the AI the label definitions
  and edge-case description and asked it to generate 16 comments sitting on the two boundary
  seams, then ran an independent pass that labeled each using only the rules. 14/16 were
  clean; the 2 unclear ones drove a 4th tiebreaker (awe/emoji vs explicit ranking;
  meta-sarcasm). This happened *before* any real data was labeled, exactly so the
  definitions would be tight first.

- **Annotation assistance — YES, with mandatory human review.** `src/label.py` pre-labels
  every comment with Groq `llama-3.3-70b-versatile` and tags each with a confidence. **Every
  pre-label is AI-assisted and must be human-reviewed**; at minimum every low/medium-confidence
  row (routed to `data/needs_review.csv`) gets a human decision before the dataset is used.
  Provenance is tracked in the `notes` column (`groq-prelabel conf=…`). This is disclosed in
  the README AI-usage section. *(Build-environment note: Groq was not keyed and Reddit blocks
  the build host's IP, so collection + labeling are staged for a residential run — see
  [STATUS.md](STATUS.md).)*

- **Failure analysis — PLANNED.** After evaluation, I feed the fine-tuned model's list of
  wrong predictions (with text, true label, predicted label) to an AI and ask it to cluster
  the errors into patterns ("collapses analysis→hot_take on short comments", "misses sarcasm"
  ). I then **verify each proposed pattern by re-reading the actual misclassified comments**
  before it goes in the README — the AI surfaces candidates, I confirm or reject them.
  `src/error_analysis.py` produces the wrong-prediction list for this.
