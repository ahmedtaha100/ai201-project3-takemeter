#!/usr/bin/env python3
"""
Pre-label raw r/sports comments with Groq llama-4-scout-17b, assign a confidence to each,
and split out a small human-review queue for the boundary cases.

    python src/label.py --in data/raw_comments.csv --out data/takemeter_labeled.csv

Comments are labeled in BATCHES (many per request) so the whole set fits inside the free-tier
tokens-per-minute limit and finishes in a couple of minutes instead of one slow call per row.

Outputs:
  - data/takemeter_labeled.csv   columns: text,label,notes   (the single, non-split dataset)
  - data/needs_review.csv        low/medium-confidence + unparsed rows for a human pass (review,
                                  then fold corrections back into the labeled CSV)

DISCLOSURE: every row here is AI-pre-labeled. The course requires you to review and correct the
pre-assigned labels before using them, at minimum every row in needs_review.csv, and to disclose
this annotation assistance in the README. The `notes` column records provenance (model +
confidence) so reviewed/overridden rows are traceable.

Needs GROQ_API_KEY. stdlib + groq + python-dotenv.
"""
import argparse
import csv
import os
import re
import sys
import time

from dotenv import load_dotenv
from labels import LABELS, definitions_block

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
BATCH_SIZE = 12


def _key_ok() -> bool:
    k = GROQ_API_KEY.strip()
    return bool(k) and not k.startswith(("gsk_PLACEHOLDER", "your_groq"))


def chat_with_backoff(client, content, max_tokens, max_retries=10):
    """One chat completion, retrying on rate limits / transient errors with backoff."""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            is_rate = "RateLimit" in type(e).__name__ or "429" in str(e)
            time.sleep(delay if is_rate else 2.0)
            delay = min(delay * 1.5, 40.0)
    return ""


BATCH_PROMPT = (
    "You are an expert annotator labeling Reddit r/sports comments for a classifier's training "
    "set. Apply this decision tree to EACH comment:\n"
    "Q1. Does it make a debatable sports claim/opinion/prediction/ranking? If no -> reaction.\n"
    "Q2. If yes, is the claim backed by a real explanatory/causal chain or multiple pieces of "
    "evidence (a lone stat is garnish, not an argument)? If no -> hot_take. If yes -> analysis.\n\n"
    "Definitions:\n{defs}\n\n"
    "Comments to label:\n{numbered}\n\n"
    "For EACH numbered comment, output exactly one line in this format:\n"
    "<number>: <reaction|hot_take|analysis> <high|medium|low>\n"
    "Output only those lines, one per comment, nothing else."
)

LINE_RE = re.compile(r"^\s*(\d+)\s*[:.)\]]\s*(reaction|hot[ _-]?take|analysis)\b\s*(high|medium|low)?",
                     re.IGNORECASE)


def parse_batch(text: str, n: int):
    """Return {index: (label, confidence)} parsed from the model's batch response."""
    out = {}
    for line in (text or "").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        label = m.group(2).lower().replace(" ", "_").replace("-", "_")
        if label == "hottake":
            label = "hot_take"
        conf = (m.group(3) or "medium").lower()
        if 1 <= idx <= n and label in LABELS:
            out[idx] = (label, conf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw_comments.csv")
    ap.add_argument("--out", default="data/takemeter_labeled.csv")
    ap.add_argument("--review", default="data/needs_review.csv")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    if not _key_ok():
        print("GROQ_API_KEY missing/placeholder. Put a real key in .env, then re-run.", file=sys.stderr)
        sys.exit(2)

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    defs = definitions_block()

    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Labeling {len(rows)} comments with {LLM_MODEL} in batches of {args.batch} ...", flush=True)

    labeled, review = [], []
    dist = {l: 0 for l in LABELS}

    def flush_outputs():
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
            w.writeheader(); w.writerows(labeled)
        with open(args.review, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "confidence", "reason"])
            w.writeheader(); w.writerows(review)

    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        numbered = "\n".join(f'{i+1}. """{b["text"]}"""' for i, b in enumerate(batch))
        try:
            content = chat_with_backoff(
                client, BATCH_PROMPT.format(defs=defs, numbered=numbered),
                max_tokens=40 * len(batch),
            )
            parsed = parse_batch(content, len(batch))
        except Exception as e:  # noqa: BLE001
            parsed = {}
            print(f"  batch {start//args.batch} error: {type(e).__name__}", flush=True)

        for i, b in enumerate(batch, start=1):
            comment = b["text"]
            if i not in parsed:
                review.append({"text": comment, "label": "", "confidence": "unparsed", "reason": "no batch line"})
                continue
            label, conf = parsed[i]
            dist[label] += 1
            labeled.append({"text": comment, "label": label, "notes": f"groq-prelabel {LLM_MODEL} conf={conf}"})
            if conf in ("low", "medium"):
                review.append({"text": comment, "label": label, "confidence": conf, "reason": "boundary/low-conf"})

        done = start + len(batch)
        print(f"  {done}/{len(rows)}  dist={dist}  review={len(review)}", flush=True)
        flush_outputs()
        time.sleep(0.3)

    total = sum(dist.values())
    print(f"\nLabeled {total} -> {args.out}   (review queue: {len(review)} -> {args.review})")
    print("Distribution:")
    for l in LABELS:
        pct = 100 * dist[l] / total if total else 0
        flag = "  <-- <20%!" if pct < 20 else ("  <-- >70%!" if pct > 70 else "")
        print(f"  {l:<10} {dist[l]:>4}  ({pct:4.1f}%){flag}")


if __name__ == "__main__":
    main()
