# study_vanilla — the parameter-matched vanilla baseline (an A/B study)

A self-contained study module, **kept separate from Burel on purpose**. It exists to
answer one question, honestly and with numbers:

> Does Burel's nested memory (Continuum Memory System + Test-Time Training) buy
> *intelligence-per-parameter* over a plain Transformer — enough to justify its cost?

Burel bets that a small model can punch above its weight by **adapting at test time**
(updating a fast-weight memory while it reads) instead of relying only on attention.
That bet is only worth making if, **at equal parameters and equal data**, Burel beats a
textbook Transformer. This module is that textbook Transformer.

## Why a separate module
The comparison must be fair and legible to anyone. So the baseline is a clean,
dependency-light GPT (`model.py`) that does **not** import `burel`; it only reads the
same tokenized dataset (`data_cache/`). You can read, train and evaluate it on its own.

## What's inside
| file | role |
|---|---|
| `model.py` | `VanillaGPT` — decoder-only, pre-norm, full causal attention, tied weights. ~20.43M params (Burel ~20.17M, within ~1.3%). |
| `data.py` | minimal access to the shared `data_cache` (`*.bin` + `meta.pkl` + `tokenizer.json`). |
| `config.yaml` | hyper-parameters, parameter- and budget-matched to Burel's proven run. |
| `train.py` | nanoGPT-style trainer with Drive backup + auto-resume (Colab-safe). |
| `eval_loss_by_chunk.py` | standalone loss-per-position evaluation of the vanilla alone. |

The decisive three-way comparison (Burel-ON / Burel-OFF / Vanilla on identical windows)
lives one level up in [`scripts/compare_three.py`](../scripts/compare_three.py), because it
necessarily touches both models.

## The fairness contract (one variable: the architecture)
Same vocabulary (16k byte-level BPE), same data (TinyStories), same context (256),
same optimization budget (batch 16, lr 3e-4, ~20k iters). Because the tokenizer and data
are identical, **the validation losses of Burel and the vanilla are directly comparable** —
unlike a char-vs-BPE comparison. That single number is the headline of the A/B.

> Note: the vanilla could legitimately train with a much larger batch and fast attention
> kernels — that is part of its real efficiency advantage. We hold batch=16 here to isolate
> the architecture as the only variable; throughput is reported separately.

## Run it
```bash
# 1) train the baseline (uses the shared data_cache prepared by Burel)
python study_vanilla/train.py

# 2) evaluate it on its own (in-distribution + an unseen-domain file, e.g. code)
python study_vanilla/eval_loss_by_chunk.py --domain_file domain_code.txt

# 3) the real pivot — three-way A/B vs Burel on identical windows
python scripts/compare_three.py \
    --burel_ckpt checkpoints/burel_best.pt \
    --vanilla_ckpt checkpoints_vanilla/vanilla_best.pt \
    --domain_file domain_code.txt
```

## How to read the result
- **Headline:** lower best val at equal params wins.
- **Loss-per-chunk** on CONTROL (seen) and UNSEEN (code): if `Burel-ON` sits **below**
  `Vanilla`, the memory pays for itself. If the vanilla matches or beats `Burel-ON`, the
  honest conclusion is to scale the vanilla (and add MoE/retrieval), and treat the nested
  memory as a research branch — not the production path.

This study is deliberately falsifiable. A negative result is a result.
