# TakeMeter: Planning

*Working notes, written before any data was collected or any model was run. The polished
writeup lives in [README.md](README.md); this is the design thinking behind it. Because it
is a plan, it states the metrics and the success bar but does not report results.*

A classifier that sorts r/sports **comments** into the kind of discourse they are:
`reaction`, `hot_take`, or `analysis`.

---

## 1. Community

**r/sports**, specifically its **comments**, not its submissions.

r/sports is one of the broad, high-traffic sports communities on Reddit, and almost all of
the real *talk* happens in comments. Submissions are mostly news links, highlight clips, and
match threads. The comments under those are where people actually argue. The appeal for this
task is that the same thread produces wildly different modes of discourse: pure emotional
venting, throwaway bold opinions, and the occasional multi-sentence breakdown, often a few
replies apart. Those three modes come from the same people about the same events, so the
classifier has to key on *how* something is said (is there a claim? is there evidence? or is
it just emotion?) rather than *what* sport or team it is about. Topic is nearly useless as a
signal here, and that is the point. A model that learns "this comment mentions the Lakers"
learns nothing about whether the comment is reaction or analysis.

It is also a practical fit. High comment volume so clearing 250 labeled rows is realistic,
public, English, and broad enough that the discourse is not all one sport's in-jokes.

---

## 2. Labels

Three labels, applied with a 2-question decision tree:

> **Q1.** Does the comment make a debatable sports claim / opinion / prediction / ranking?
> **No → `reaction`.**
> **Q2.** If yes, is the claim backed by a real explanatory/causal chain or multiple pieces
> of evidence (a lone stat is garnish, not an argument)? **No → `hot_take`. Yes → `analysis`.**

- **`reaction`**: an emotional, in-the-moment expression responding to a play/game/news
 that makes no substantive sports claim.
 - *"WHAT A FINISH ARE YOU KIDDING ME"*
 - *"bro I'm crying lmao we're so back"*
- **`hot_take`**: a strongly-stated, debatable sports opinion/claim/prediction/ranking
 asserted with little or no supporting reasoning.
 - *"He's washed, time to admit it"*
 - *"This kid is already a top-5 player in the league"*
- **`analysis`**: a reasoned, evidence-based argument that explains *why*, via a causal
 chain or stacked evidence (stats, scheme, matchups, tape).
 - *"Defenses keep sitting on the screen, so until he punishes it with the pull-up the
 second unit's spacing collapses and the whole drive-and-kick game dies."*
 - *"His efficiency dropped because he's taking more contested late-clock looks now that the
 shooters around him aren't hitting. It's a lineup problem, not effort."*

The labels are **mutually exclusive** (the tree forces exactly one) and **exhaustive** enough
that an "other" bucket is not needed: every on-topic sports comment is either emotion, an
unsupported claim, or a supported claim. Off-topic and spam get dropped at collection, not
given a label.

---

## 3. Hard edge cases

The fuzziness lives on two seams: **reaction ↔ hot_take** (does an emotional comment also
make a claim?) and **hot_take ↔ analysis** (is there real reasoning, or just a stat tossed
in?).

I plan to stress-test the definitions *before* annotating by having an AI generate a batch of
deliberately borderline comments and then, in a separate pass, label each using only the
rules (see [§ AI Tool Plan](#ai-tool-plan)). Anything the rules can't cleanly resolve becomes
a tiebreaker. Three real difficult cases and how I resolve them:

1. **Emotional outburst that smuggles in a claim.** *"NAHHHH man that was a ROBBERY the refs
 handed them this game I'm sick."* The surface is pure emotion, but "the refs decided the
 game" is a debatable claim with no support → **`hot_take`**. **Rule:** if you can extract a
 debatable sports proposition, it is *not* `reaction`, however loud it is.
2. **Bold claim with a single stat attached.** *"He's the best player alive and it's not
 close, dude is averaging a triple-double."* One stat is garnish, not an argument →
 **`hot_take`**. The same claim *becomes* `analysis` only when the stat is wired into a
 causal explanation. **Rule:** analysis needs an explanatory chain. Strip out the bold claim
 and there must still be a substantive "why" left standing.
3. **Awe / stat-line hype and meta-sarcasm.** *"40 burger on 65 percent ARE YOU KIDDING ME"*
 and *"oh great, another 'the old guys would've dominated' take, riveting stuff."* The first
 is awe with an *implied* ranking; the second mocks the discourse, not an actual on-field
 position. Both → **`reaction`**. **Rule:** awe plus a stat line is `reaction` unless it
 states an *explicit verbal* ranking ("he's the GOAT" → `hot_take`); sarcasm aimed at other
 takes or at the discourse, with no on-field claim of its own, is `reaction`.

During annotation, any comment that gives genuine pause goes to `data/needs_review.csv` with
the model's tentative label and confidence, and gets a human decision recorded there.

---

## 4. Data collection plan

**Original plan:** stream live comments from **r/nba** via Reddit's public JSON endpoints, no
API key, no PRAW.

**What broke:** Reddit's API now returns **HTTP 403** to unauthenticated clients, so the live
scrape failed outright. Rather than fake it, I pivoted on two axes and am documenting both
honestly.

**The pivot:**
- **Source:** instead of live Reddit, pull **real r/sports comments** from a public Reddit
 archive on Hugging Face: `HuggingFaceGECLM/REDDIT_comments` (the r/sports split, built from
 PushShift dumps covering 2006 to 2023). The comments are genuine Reddit comments. Only the
 *collection channel* changed, from a live endpoint to an archived dump.
- **Community:** r/nba → **r/sports**. The archive's r/sports split is well populated and the
 discourse is the same shape (reactions, hot takes, the occasional real breakdown), so the
 task and the taxonomy carry over without change. The decision tree never referenced
 basketball specifics on purpose, so nothing in the labels had to move.

**How:** `src/collect.py` streams the r/sports split from the HF dataset and writes raw rows
to `data/raw_comments.csv`. Target **~400 raw** comments so there is slack to filter
(drop deleted/removed/bot/too-short/link-only rows, dedupe) and still land **≥ 250 labeled**.

**Per label:** the natural mix on a general sports sub skews hard toward `reaction`, with
`analysis` scarce. Plan: aim for a rough **40 / 35 / 25** split across
`reaction / hot_take / analysis`, with every class **≥ 20%** and none **> 70%**. Because I am
sampling from an archive rather than a live thread feed, I can oversample longer, argument-
shaped comments to lift `analysis` and `hot_take` above the firehose's reaction bias.

**If a label is underrepresented after 200:** keep pulling, but selectively, biasing the
sample toward longer comments where analysis and hot takes concentrate, and capping further
`reaction` intake. `src/label.py` prints the live distribution and flags any class under 20%
or over 70%, so balancing is a checkable loop, not a guess. Balancing happens by collecting
more of the thin class, never by relabeling to hit a quota.

---

## 5. Evaluation metrics

Accuracy alone is not enough here, for two reasons. The classes are **imbalanced** (a model
that always guesses `reaction` could post a deceptively high number), and the **cost of
confusions is uneven** (mixing up `hot_take` and `analysis` is the interesting failure;
`reaction` is the easy class). So the plan reports:

- **Overall accuracy**: one headline number, on the held-out test set, for both models.
- **Per-class precision, recall, and F1**: this is the real report. Recall on `analysis`
 (the rare, valuable class) and the precision/recall trade on the `hot_take` ↔ `analysis`
 seam are what actually tell me whether the model learned the distinction or just learned the
 base rates.
- **Confusion matrix** (3x3): shows *where* the errors go, which is the whole point of a
 discourse classifier. Is it collapsing `analysis` into `hot_take`, or scattering at random?
- **Macro-F1 as the headline comparison number** between the fine-tuned model and the
 baseline. It weights all three classes equally, so the rare class counts as much as the
 common one, and it is not flattered by imbalance. A model can have decent accuracy while
 whiffing on `hot_take` entirely; macro-F1 catches that, accuracy hides it. That is exactly
 why macro-F1, not accuracy, is the number I lead with.

Both models are evaluated on the **same** held-out test split, produced by
`src/prepare_splits.py` with a stratified **70 / 15 / 15** train/val/test split and a leakage
guard so no comment appears in two splits.

---

## 6. Definition of success

A discourse classifier is genuinely useful (auto-tagging a thread, or surfacing the
`analysis` comments actually worth reading) only if it clears the base-rate trap and handles
the rare class. Concrete thresholds, set now so I can't move the goalposts later:

- **Primary:** fine-tuned **macro-F1 ≥ 0.70** on the test set, and it must **beat the
 zero-shot baseline's macro-F1**. Beating a zero-shot LLM with a small fine-tuned DistilBERT
 is the result that would justify the fine-tune at all.
- **Per-class floor:** **`analysis` recall ≥ 0.60**. The rare, valuable class is the one worth
 protecting; a model that can't find analysis is not useful no matter its accuracy.
- **Sanity ceiling:** if accuracy comes in **> 0.95**, treat it as a red flag (test leakage,
 or labels too easy) and re-check, not a win.

"Good enough" = primary plus the per-class floor both met, with the confusion matrix showing
errors concentrated on the genuinely hard `hot_take` ↔ `analysis` seam rather than smeared
everywhere.

A note on what the baseline is, since it matters for honesty: the baseline is a **zero-shot
Groq model** (`llama-3.3-70b-versatile`), deliberately a **different** model than the one used to
pre-label the data (`llama-4-scout-17b`). If the baseline and the labeler were the same model,
beating the baseline would just mean the test labels agreed with themselves. Using a different
model keeps the comparison from being circular.

---

## AI Tool Plan

An explicit decision on each of the three required uses.

- **Label stress-testing: do this first, before annotating.** Give the AI the label
 definitions and the edge-case writeup, ask it to generate a batch of comments sitting right
 on the two boundary seams, then run an independent pass that labels each using only the
 rules. Whatever the rules resolve cleanly confirms the taxonomy is tight; whatever they
 can't becomes a new tiebreaker rule (this is where the awe-plus-stat vs explicit-ranking
 rule and the meta-sarcasm rule come from). The whole point of doing this *before* touching
 real data is so the definitions are sharp before a single real comment gets a label.

- **Annotation assistance: yes, with review.** `src/label.py` pre-labels
 every comment with Groq `llama-4-scout-17b`, batched, and tags each with a confidence.
Every pre-label is AI-assisted and gets checked against the taxonomy before the dataset is used.
At minimum, every low/medium-confidence row gets routed to `data/needs_review.csv` (a ~38-row
review queue); the final pass is documented in `data/review_notes.md`. Provenance is tracked in
the `notes` column. This is disclosed in the README's AI-usage section. The AI proposes; I
decide. It never gets the final say on a label.

- **Failure analysis: planned, after evaluation.** Once both models are scored, feed the
 fine-tuned model's list of wrong predictions (text, true label, predicted label) to an AI
 and ask it to cluster the errors into named patterns (for example "collapses analysis into
 hot_take on short comments" or "misses sarcasm"). Then **verify each proposed pattern by
 re-reading the actual misclassified comments** before any of it goes in the writeup. The AI
 surfaces candidate patterns; I confirm or reject each one against the real rows.
 `src/error_analysis.py` produces the wrong-prediction list that feeds this step.
