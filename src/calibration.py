#!/usr/bin/env python3
"""
Stretch: confidence calibration for the fine-tuned model.

    python src/calibration.py

Are the softmax confidences meaningful — does a 90%-confident prediction actually beat a
60%-confident one? Bins the test predictions by confidence, reports accuracy per bin, and
computes Expected Calibration Error (ECE). Writes artifacts/calibration.png + prints a table.
"""
import json
import os

import numpy as np

from labels import LABELS  # noqa: F401  (kept for parity / future per-class calibration)


def main():
    path = "artifacts/finetuned_preds.json"
    if not os.path.exists(path):
        raise SystemExit("Run finetune.py first (need artifacts/finetuned_preds.json).")
    with open(path) as f:
        rows = json.load(f)

    conf = np.array([r["confidence"] for r in rows])
    correct = np.array([1.0 if r["true"] == r["pred"] else 0.0 for r in rows])

    bins = np.linspace(0.0, 1.0, 6)  # 5 bins
    print("confidence bin     n   acc    mean_conf")
    ece, n = 0.0, len(rows)
    bin_centers, bin_accs, bin_confs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        mc = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - mc)
        bin_centers.append((lo + hi) / 2); bin_accs.append(acc); bin_confs.append(mc)
        print(f"  [{lo:.2f},{hi:.2f})   {int(mask.sum()):>3}  {acc:.2f}   {mc:.2f}")
    print(f"\nExpected Calibration Error (ECE): {ece:.3f}  (lower = better calibrated)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(bin_confs, bin_accs, "o-", color="purple", label="model")
        ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability (ECE={ece:.3f})"); ax.legend()
        fig.tight_layout(); fig.savefig("artifacts/calibration.png", dpi=150)
        print("Wrote artifacts/calibration.png")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
