# Experiment 02 — nested memory vs vanilla Transformer (equal parameters)

**Date:** 2026-06-16 · **Status:** ⏳ training in progress, results pending

## The question, in one line

At the **same parameter count and the same data**, does Burel's nested memory beat a
plain Transformer that adapts to context only through attention?

This is *the* test of Burel's whole thesis. [Experiment 01](../01_ttt_ablation/) showed the
memory does real work — but only compared to itself with the memory switched off, which is
a crippled baseline. The fair opponent is a normal Transformer with full attention.

## How we test it (one variable: the architecture)

We built a clean, textbook decoder-only Transformer — [`study_vanilla/`](../../study_vanilla/) —
kept deliberately **separate** from Burel so it's easy to read and trust. It is matched to
Burel as tightly as possible:

| | Burel | Vanilla baseline |
|---|---|---|
| parameters | ~20.17M | **~20.43M** (within ~1.3%) |
| vocabulary | 16k byte-level BPE | same |
| data | TinyStories | same `data_cache` |
| context | 256 | 256 |
| training budget | batch 16, lr 3e-4, ~20k iters | **same** |
| how it adapts to context | nested memory + Test-Time Training | full causal attention |

Everything is held equal except the one thing under test: **memory vs attention.**

Because the tokenizer and data are identical, the two models' **validation losses are
directly comparable** (unlike comparing a char model to a BPE model). That single number is
the headline.

## What we measure

1. **Headline:** best validation loss at equal parameters (lower wins). Burel's is 2.4831.
2. **Loss per chunk** on CONTROL (seen) and UNSEEN (code), three curves on identical
   windows: **Burel-ON**, **Burel-OFF**, **Vanilla**.
   - If `Burel-ON` sits **below** `Vanilla` → the memory pays for itself.
   - If `Vanilla` matches or beats `Burel-ON` → full attention already does the job; the
     honest move is to scale the plain Transformer (and add MoE / retrieval) rather than the
     nested memory.

Script: [`scripts/compare_three.py`](../../scripts/compare_three.py).

## A note on fairness

The vanilla baseline could legitimately train with a much bigger batch and fast attention
kernels — that's part of its real-world efficiency advantage. We hold batch=16 here so the
**only** variable is the architecture; the speed/cost advantage is reported separately. A
truly complete verdict weighs quality **and** cost (tokens/sec, FLOPs), not just loss.

## Results

> Pending — the vanilla is training now. Numbers and the three-curve table go here once the
> run finishes, with the same honest reading we gave experiment 01.

## Reproduce it

Put `burel_vanilla_ab.zip` and `Burel_Vanilla_AB_Colab.ipynb` (both in this folder) into
your Google Drive, open the notebook, pick a **GPU (T4)** runtime, and run the cells. It
trains the vanilla baseline (~1–1.5h, resumable) and then runs the three-way comparison
against your Burel checkpoint automatically.

Locally, after training the baseline:

```bash
python study_vanilla/train.py
python scripts/compare_three.py \
    --burel_ckpt checkpoints/burel_best.pt \
    --vanilla_ckpt checkpoints_vanilla/vanilla_best.pt \
    --domain_file your_code.txt
```
