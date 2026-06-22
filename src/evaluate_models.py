#!/usr/bin/env python3
"""
Evaluate the fine-tuned model and the zero-shot baseline on the SAME test set and emit every
artifact the README needs.

    python src/evaluate_models.py

Reads (whichever exist):
  artifacts/finetuned_preds.json , artifacts/baseline_preds.json
Writes:
  evaluation_results.json     overall accuracy + per-class P/R/F1 + confusion matrix, both models
  confusion_matrix.png        fine-tuned model's 3x3 confusion matrix (committed image)
  artifacts/report_blocks.md  ready-to-paste markdown: confusion table, metrics tables, samples
"""
import argparse
import json
import os

import numpy as np

from labels import LABELS


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def confusion(rows):
    idx = {l: i for i, l in enumerate(LABELS)}
    m = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for r in rows:
        if r["true"] in idx and r["pred"] in idx:
            m[idx[r["true"]]][idx[r["pred"]]] += 1
    return m


def per_class_metrics(m):
    out = {}
    for i, label in enumerate(LABELS):
        tp = m[i][i]
        fp = m[:, i].sum() - tp
        fn = m[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                      "f1": round(f1, 4), "support": int(m[i, :].sum())}
    return out


def summarize(rows):
    m = confusion(rows)
    pc = per_class_metrics(m)
    total = m.sum()
    acc = float(np.trace(m) / total) if total else 0.0
    macro_f1 = float(np.mean([pc[l]["f1"] for l in LABELS]))
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": pc, "confusion_matrix": m.tolist(), "n": int(total)}


def md_confusion(m, title):
    head = "| true ↓ / pred → | " + " | ".join(LABELS) + " | **total** |"
    sep = "|" + "---|" * (len(LABELS) + 2)
    lines = [f"**{title}**", "", head, sep]
    for i, label in enumerate(LABELS):
        row = " | ".join(str(m[i][j]) for j in range(len(LABELS)))
        lines.append(f"| **{label}** | {row} | {int(m[i].sum())} |")
    return "\n".join(lines)


def md_metrics(name, s):
    lines = [f"**{name}** — accuracy **{s['accuracy']:.3f}**, macro-F1 **{s['macro_f1']:.3f}** (n={s['n']})",
             "", "| label | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for l in LABELS:
        p = s["per_class"][l]
        lines.append(f"| {l} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |")
    return "\n".join(lines)


def plot_confusion(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(m, cmap="Purples")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("TakeMeter — fine-tuned confusion matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(m[i][j]), ha="center", va="center",
                    color="white" if m[i][j] > m.max() / 2 else "black", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    ft = load(os.path.join(args.artifacts, "finetuned_preds.json"))
    bl = load(os.path.join(args.artifacts, "baseline_preds.json"))
    if not ft and not bl:
        raise SystemExit("No prediction files found. Run finetune.py and/or baseline.py first.")

    results, blocks = {}, []
    if ft:
        s = summarize(ft); results["fine_tuned"] = s
        plot_confusion(np.array(s["confusion_matrix"]), "confusion_matrix.png")
        blocks += [md_metrics("Fine-tuned DistilBERT", s), "",
                   md_confusion(np.array(s["confusion_matrix"]), "Fine-tuned confusion matrix"), ""]
    if bl:
        s = summarize(bl); results["baseline_zero_shot"] = s
        blocks += [md_metrics("Zero-shot Groq baseline (llama-3.1-8b-instant)", s), ""]
    if ft and bl:
        d = results["fine_tuned"]["macro_f1"] - results["baseline_zero_shot"]["macro_f1"]
        results["delta_macro_f1_ft_minus_baseline"] = round(d, 4)
        blocks.append(f"**Fine-tuned − baseline macro-F1: {d:+.3f}**")

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.artifacts, "report_blocks.md"), "w") as f:
        f.write("\n".join(blocks) + "\n")

    print("Wrote evaluation_results.json + artifacts/report_blocks.md")
    print("\n" + "\n".join(blocks))


if __name__ == "__main__":
    main()
