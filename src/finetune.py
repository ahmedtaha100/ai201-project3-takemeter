#!/usr/bin/env python3
"""
Fine-tune distilbert-base-uncased on the TakeMeter comment classes and predict the test set.

    python src/finetune.py            # trains on artifacts/train.csv + val.csv, predicts test.csv

Outputs:
  - models/distilbert-takemeter/           the saved model + tokenizer (gitignored)
  - artifacts/finetuned_preds.json         [{text, true, pred, confidence, probs}] on the test set

Defaults (course baseline): 3 epochs, lr 2e-5, batch size 16, max_len 128.

DELIBERATE HYPERPARAMETER DECISION (documented in the README):
  We raise epochs 3 -> 5 and add early stopping on val macro-F1. With only ~250 examples and
  three subjective classes, three epochs underfits the rare `analysis` class; the extra
  epochs + early-stopping-on-val let it learn the hard seam without overfitting (the run
  keeps the best val checkpoint, not the last). Set EPOCHS=3 below to reproduce the default.

Auto-detects GPU; falls back to CPU (fine for this dataset size).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from labels import LABELS, LABEL2ID, ID2LABEL

EPOCHS = 5            # deliberate change from the 3-epoch default (see module docstring)
LR = 2e-5
BATCH = 16
MAX_LEN = 128
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "models/distilbert-takemeter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(os.path.join(args.artifacts, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.artifacts, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.artifacts, "test.csv"))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(df):
        ds = Dataset.from_dict({"text": df.text.tolist(), "labels": [LABEL2ID[l] for l in df.label]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True)

    train_ds, val_ds, test_ds = encode(train_df), encode(val_df), encode(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    targs = TrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=args.epochs,
        learning_rate=LR, per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=10, report_to="none", seed=42,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    # Predict the held-out test set with softmax confidences.
    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = probs.argmax(axis=-1)

    rows = []
    for i, text in enumerate(test_df.text.tolist()):
        rows.append({
            "text": text,
            "true": test_df.label.iloc[i],
            "pred": ID2LABEL[int(pred_ids[i])],
            "confidence": float(probs[i].max()),
            "probs": {LABELS[j]: float(probs[i][j]) for j in range(len(LABELS))},
        })
    os.makedirs(args.artifacts, exist_ok=True)
    with open(os.path.join(args.artifacts, "finetuned_preds.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} test predictions -> {args.artifacts}/finetuned_preds.json")


if __name__ == "__main__":
    main()
