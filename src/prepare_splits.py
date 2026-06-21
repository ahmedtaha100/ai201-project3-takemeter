#!/usr/bin/env python3
"""
Make the one-time 70/15/15 split from the single labeled CSV and write it to artifacts/.
Run this once; the fine-tuner and the baseline both read artifacts/test.csv so they are
scored on the identical held-out set.

    python src/prepare_splits.py --in data/takemeter_labeled.csv
"""
import argparse
import os

from dataset import load_labeled, stratified_split
from labels import LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/takemeter_labeled.csv")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_labeled(args.inp)
    train, val, test = stratified_split(df)

    for name, part in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(args.outdir, f"{name}.csv")
        part.to_csv(path, index=False)
        dist = {l: int((part.label == l).sum()) for l in LABELS}
        print(f"{name:<5} {len(part):>4} rows  {dist} -> {path}")
    print(f"\ntotal {len(df)} unique labeled rows. Same test.csv feeds both models.")


if __name__ == "__main__":
    main()
