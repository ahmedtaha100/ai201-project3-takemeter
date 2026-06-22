**Fine-tuned DistilBERT** — accuracy **0.632**, macro-F1 **0.479** (n=38)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | 0.632 | 0.750 | 0.686 | 16 |
| hot_take | 0.000 | 0.000 | 0.000 | 9 |
| analysis | 0.632 | 0.923 | 0.750 | 13 |

**Fine-tuned confusion matrix**

| true ↓ / pred → | reaction | hot_take | analysis | **total** |
|---|---|---|---|---|
| **reaction** | 12 | 0 | 4 | 16 |
| **hot_take** | 6 | 0 | 3 | 9 |
| **analysis** | 1 | 0 | 12 | 13 |

**Zero-shot Groq baseline (llama-3.3-70b-versatile)** — accuracy **0.921**, macro-F1 **0.901** (n=38)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | 0.889 | 1.000 | 0.941 | 16 |
| hot_take | 1.000 | 0.667 | 0.800 | 9 |
| analysis | 0.929 | 1.000 | 0.963 | 13 |

**Fine-tuned − baseline macro-F1: -0.423**
