**Fine-tuned DistilBERT** — accuracy **0.737**, macro-F1 **0.549** (n=38)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | 0.727 | 0.941 | 0.821 | 17 |
| hot_take | 0.000 | 0.000 | 0.000 | 8 |
| analysis | 0.750 | 0.923 | 0.828 | 13 |

**Fine-tuned confusion matrix**

| true ↓ / pred → | reaction | hot_take | analysis | **total** |
|---|---|---|---|---|
| **reaction** | 16 | 0 | 1 | 17 |
| **hot_take** | 5 | 0 | 3 | 8 |
| **analysis** | 1 | 0 | 12 | 13 |

**Zero-shot Groq baseline (llama-3.1-8b-instant)** — accuracy **0.816**, macro-F1 **0.779** (n=38)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| reaction | 0.789 | 0.882 | 0.833 | 17 |
| hot_take | 0.800 | 0.500 | 0.615 | 8 |
| analysis | 0.857 | 0.923 | 0.889 | 13 |

**Fine-tuned − baseline macro-F1: -0.230**
