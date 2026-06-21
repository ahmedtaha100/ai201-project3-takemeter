#!/usr/bin/env python3
"""
Pre-label raw r/nba comments with Groq llama-3.3-70b-versatile, assign a confidence to each,
and split out a small human-review queue for the boundary cases.

    python src/label.py --in data/raw_comments.csv --out data/takemeter_labeled.csv

Outputs:
  - data/takemeter_labeled.csv   columns: text,label,notes   (the single, non-split dataset)
  - data/needs_review.csv        low/medium-confidence rows for a human pass (review then
                                  fold corrections back into the labeled CSV)

DISCLOSURE: every row here is AI-pre-labeled. The course requires you to review and correct
the pre-assigned labels before using them — at minimum every row in needs_review.csv — and to
disclose this annotation assistance in the README's AI-usage section. The `notes` column
records the provenance ("groq-prelabel conf=high") so reviewed/overridden rows are traceable.

Needs GROQ_API_KEY (in .env or env). stdlib + groq + python-dotenv.
"""
import argparse
import csv
import os
import re
import sys

from dotenv import load_dotenv
from labels import LABELS, definitions_block

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"


def _key_ok() -> bool:
    k = GROQ_API_KEY.strip()
    return bool(k) and not k.startswith(("gsk_PLACEHOLDER", "your_groq"))


LABEL_PROMPT = (
    "You are an expert annotator labeling Reddit r/nba comments for a classifier's training "
    "set. Apply this decision tree strictly:\n"
    "Q1. Does the comment make a debatable basketball claim/opinion/prediction/ranking? "
    "If no -> reaction.\n"
    "Q2. If yes, is the claim backed by a real explanatory/causal chain or multiple pieces "
    "of evidence (a lone stat is garnish, not an argument)? If no -> hot_take. If yes -> analysis.\n\n"
    "Definitions:\n{defs}\n\n"
    'Comment: """{comment}"""\n\n'
    "Respond in EXACTLY this format:\n"
    "Label: <reaction|hot_take|analysis>\n"
    "Confidence: <high|medium|low>\n"
    "Reason: <one short clause>"
)


def parse(text: str):
    label, conf, reason = "", "low", ""
    for line in (text or "").splitlines():
        low = line.lower().strip()
        if low.startswith("label:"):
            cand = re.sub(r"[^a-z_]", "", line.split(":", 1)[1].strip().lower().split()[0] if line.split(":",1)[1].strip() else "")
            if cand in LABELS:
                label = cand
        elif low.startswith("confidence:"):
            c = line.split(":", 1)[1].strip().lower()
            if c in ("high", "medium", "low"):
                conf = c
        elif low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    if not label:
        for lbl in LABELS:
            if re.search(rf"\b{lbl}\b", (text or "").lower()):
                label, conf = lbl, "low"
                break
    return label, conf, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw_comments.csv")
    ap.add_argument("--out", default="data/takemeter_labeled.csv")
    ap.add_argument("--review", default="data/needs_review.csv")
    args = ap.parse_args()

    if not _key_ok():
        print("GROQ_API_KEY missing/placeholder. Put a real key in .env, then re-run.", file=sys.stderr)
        sys.exit(2)

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    defs = definitions_block()

    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Labeling {len(rows)} comments with {LLM_MODEL} ...")

    labeled, review = [], []
    dist = {l: 0 for l in LABELS}
    for i, row in enumerate(rows, 1):
        comment = row["text"]
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": LABEL_PROMPT.format(defs=defs, comment=comment)}],
                temperature=0,
                max_tokens=60,
            )
            label, conf, reason = parse(resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            label, conf, reason = "", "low", f"error:{type(e).__name__}"

        if label not in LABELS:
            review.append({"text": comment, "label": "", "confidence": "unparsed", "reason": reason})
            continue
        dist[label] += 1
        labeled.append({"text": comment, "label": label, "notes": f"groq-prelabel conf={conf}"})
        if conf in ("low", "medium"):
            review.append({"text": comment, "label": label, "confidence": conf, "reason": reason})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  dist={dist}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
        w.writeheader(); w.writerows(labeled)
    with open(args.review, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "confidence", "reason"])
        w.writeheader(); w.writerows(review)

    total = sum(dist.values())
    print(f"\nLabeled {total} -> {args.out}   (review queue: {len(review)} -> {args.review})")
    print("Distribution:")
    for l in LABELS:
        pct = 100 * dist[l] / total if total else 0
        flag = "  <-- <20%!" if pct < 20 else ("  <-- >70%!" if pct > 70 else "")
        print(f"  {l:<10} {dist[l]:>4}  ({pct:4.1f}%){flag}")
    if total < 250:
        print("NOTE: <250 labeled — collect more (src/collect.py) before fine-tuning.")
    if any(100*dist[l]/total < 20 for l in LABELS if total) or any(100*dist[l]/total > 70 for l in LABELS if total):
        print("NOTE: distribution out of bounds — collect more of the thin class(es) and re-balance.")


if __name__ == "__main__":
    main()
