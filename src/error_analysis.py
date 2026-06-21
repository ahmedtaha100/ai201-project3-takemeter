#!/usr/bin/env python3
"""
Stretch: systematic error-pattern analysis for the fine-tuned model.

    python src/error_analysis.py

Lists every test misclassification (text, true, pred, confidence), tallies the confusion
directions, and writes artifacts/wrong_predictions.csv — the input to the AI failure-analysis
pass described in planning.md. The AI proposes patterns; you verify them by re-reading these
rows before anything goes in the README.
"""
import csv
import json
import os
from collections import Counter

from labels import LABELS


def main():
    path = "artifacts/finetuned_preds.json"
    if not os.path.exists(path):
        raise SystemExit("Run finetune.py first (need artifacts/finetuned_preds.json).")
    with open(path) as f:
        rows = json.load(f)

    wrong = [r for r in rows if r["true"] != r["pred"]]
    directions = Counter(f'{r["true"]} -> {r["pred"]}' for r in wrong)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/wrong_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "true", "pred", "confidence"])
        for r in sorted(wrong, key=lambda x: -x.get("confidence", 0)):
            w.writerow([r["text"], r["true"], r["pred"], round(r.get("confidence", 0), 3)])

    print(f"{len(wrong)}/{len(rows)} wrong on the test set.")
    print("Confusion directions (true -> pred):")
    for k, v in directions.most_common():
        print(f"  {k:<26} {v}")
    print("\nMost confident mistakes (the worth-reading ones):")
    for r in sorted(wrong, key=lambda x: -x.get("confidence", 0))[:5]:
        print(f"  [{r['true']}→{r['pred']} @ {r.get('confidence',0):.2f}] {r['text'][:90]}")
    print("\nWrote artifacts/wrong_predictions.csv (feed to the AI failure-analysis pass).")


if __name__ == "__main__":
    main()
