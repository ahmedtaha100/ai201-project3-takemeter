#!/usr/bin/env python3
"""
Stretch: deployed interface. Paste an r/sports comment -> predicted label + confidence.

    python src/app.py            # launches a local Gradio app

Loads the fine-tuned model from models/distilbert-takemeter (created by src/finetune.py).
If the model isn't present yet, the UI says so instead of crashing.
"""
import os

import gradio as gr

from labels import DEFINITIONS, ID2LABEL, LABELS

MODEL_DIR = "models/distilbert-takemeter"
_pipe = None


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe
    if not os.path.isdir(MODEL_DIR):
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    def predict(text):
        enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        ranked = sorted(((LABELS[i], float(probs[i])) for i in range(len(LABELS))),
                        key=lambda x: -x[1])
        return ranked

    _pipe = predict
    return _pipe


def classify(comment: str) -> str:
    if not comment.strip():
        return "Paste a comment first."
    predict = _load()
    if predict is None:
        return ("⚠️ No fine-tuned model found at models/distilbert-takemeter.\n"
                "Run the pipeline first: prepare_splits.py -> finetune.py.")
    ranked = predict(comment)
    top, conf = ranked[0]
    bars = "\n".join(f"  {l:<10} {'█' * int(p*20):<20} {p:.1%}" for l, p in ranked)
    return f"### {top}  ({conf:.1%})\n{DEFINITIONS[top]}\n\n```\n{bars}\n```"


with gr.Blocks(title="TakeMeter") as demo:
    gr.Markdown("# TakeMeter\nClassify an r/sports comment: **reaction**, **hot_take**, or **analysis**.")
    inp = gr.Textbox(label="r/sports comment", lines=3, placeholder="e.g. He's already a top-5 player and it's not close")
    out = gr.Markdown()
    gr.Button("Classify").click(classify, inputs=inp, outputs=out)
    gr.Examples(
        ["WHAT A GOAL ARE YOU KIDDING ME", "He's washed, time to admit it",
         "They keep blitzing on third down, so until the back picks it up the play-action never develops and the offense stalls"],
        inputs=inp,
    )

if __name__ == "__main__":
    demo.launch()
