"""
Shared dataset loading + the 70/15/15 stratified split, with a leakage guard.

Both the fine-tuner and the baseline must evaluate on the SAME held-out test set, so the
split lives here behind a fixed seed and is written to disk once by prepare_splits.py.
"""
import pandas as pd

from labels import LABELS

SEED = 42


def load_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("labeled CSV must have at least 'text' and 'label' columns")
    df = df[["text", "label"] + ([c for c in df.columns if c == "notes"])].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # Leakage guard #1: drop exact-duplicate texts (a comment in train AND test would inflate
    # the score). Keep the first occurrence.
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    dropped = before - len(df)

    # Keep only in-taxonomy labels.
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    if dropped:
        print(f"  leakage guard: dropped {dropped} duplicate-text rows")
    return df


def stratified_split(df: pd.DataFrame, seed: int = SEED):
    """70/15/15 train/val/test, stratified by label, seeded. Returns three DataFrames."""
    train_parts, val_parts, test_parts = [], [], []
    for label in LABELS:
        sub = df[df["label"] == label].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(sub)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        test_parts.append(sub.iloc[:n_test])
        val_parts.append(sub.iloc[n_test:n_test + n_val])
        train_parts.append(sub.iloc[n_test + n_val:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Leakage guard #2: assert no text overlap across splits.
    s_train, s_test, s_val = set(train.text), set(test.text), set(val.text)
    assert not (s_train & s_test), "leakage: train/test text overlap"
    assert not (s_train & s_val), "leakage: train/val text overlap"
    assert not (s_val & s_test), "leakage: val/test text overlap"
    return train, val, test
