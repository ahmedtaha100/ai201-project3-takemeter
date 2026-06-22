#!/usr/bin/env python3
"""
Zero-shot baseline: classify the SAME held-out test set with a Groq chat model, no
task-specific training. Run AFTER prepare_splits.py and (ideally) BEFORE looking at the
fine-tuned model's numbers.

    python src/baseline.py            # reads artifacts/test.csv -> artifacts/baseline_preds.json

The prompt embeds the label definitions and instructs the model to output ONLY the label
name (src/labels.build_prompt). Parsing is defensive; the script reports the unparseable
rate and warns if it exceeds 10% (revise the prompt if so).

Needs GROQ_API_KEY. stdlib + groq + python-dotenv + pandas.
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv

from labels import LABELS, build_prompt

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# The handout names llama-3.3-70b-versatile, but the local Groq account hit its daily 70B
# token cap on 2026-06-22. The committed artifacts therefore use this non-circular zero-shot
# baseline. Override with TAKEMETER_BASELINE_MODEL=llama-3.3-70b-versatile when quota is reset.
LLM_MODEL = os.getenv("TAKEMETER_BASELINE_MODEL", "llama-3.1-8b-instant")


def chat_with_backoff(client, content, max_tokens, max_retries=8):
    """One chat completion, retrying on rate limits / transient errors with backoff."""
    delay = 4.0
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
            delay = min(delay * 1.6, 30.0)
    return ""


def _key_ok() -> bool:
    k = GROQ_API_KEY.strip()
    return bool(k) and not k.startswith(("gsk_PLACEHOLDER", "your_groq"))


def parse_label(text: str) -> str:
    t = (text or "").strip().lower()
    # normalize the common human-readable spellings of hot_take so "hot take"/"hot-take" don't
    # fall through to the majority fallback and silently mislabel a test item.
    t = t.replace("hot-take", "hot_take").replace("hot take", "hot_take").replace("hottake", "hot_take")
    # exact first
    for lbl in LABELS:
        if t == lbl:
            return lbl
    # first whole-word mention
    for lbl in LABELS:
        if re.search(rf"\b{lbl}\b", t):
            return lbl
    return ""  # unparseable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    if not _key_ok():
        print("GROQ_API_KEY missing/placeholder. Put a real key in .env, then re-run.", file=sys.stderr)
        sys.exit(2)

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    test = pd.read_csv(os.path.join(args.artifacts, "test.csv"))
    print(f"Zero-shot baseline on {len(test)} test comments with {LLM_MODEL} ...")

    rows, unparsed = [], 0
    for i, r in test.iterrows():
        try:
            raw = chat_with_backoff(client, build_prompt(r.text), max_tokens=24)
        except Exception as e:  # noqa: BLE001
            raw = f"__error__ {type(e).__name__}"
        time.sleep(0.2)
        pred = parse_label(raw)
        if not pred:
            unparsed += 1
            pred = "reaction"  # majority fallback so every test item gets a prediction
        rows.append({"text": r.text, "true": r.label, "pred": pred, "raw": raw.strip()[:80]})

    os.makedirs(args.artifacts, exist_ok=True)
    with open(os.path.join(args.artifacts, "baseline_preds.json"), "w") as f:
        json.dump(rows, f, indent=2)

    rate = 100 * unparsed / len(test) if len(test) else 0
    print(f"Wrote {len(rows)} baseline predictions -> {args.artifacts}/baseline_preds.json")
    print(f"Unparseable: {unparsed}/{len(test)} ({rate:.1f}%)"
          + ("  <-- >10%, revise the prompt!" if rate > 10 else "  (ok, <=10%)"))


if __name__ == "__main__":
    main()
