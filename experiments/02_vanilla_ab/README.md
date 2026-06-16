# Experiment 02 — nested memory vs vanilla Transformer (equal parameters)

**Date:** 2026-06-16 · **Status:** ✅ done — **thesis falsified: vanilla wins at equal parameters**

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

The vanilla baseline trained to **val 2.0052** vs Burel's **2.4831** — at equal parameters,
data and budget. Lower is better, and these are directly comparable (same tokenizer/data).

| metric | Burel (ON) | Vanilla | winner |
|---|---:|---:|---|
| **best val loss** (headline) | 2.4831 (ppl 11.98) | **2.0052 (ppl 7.43)** | Vanilla, by 0.48 nats (≈ −38% perplexity) |
| CONTROL — mean loss/chunk | 2.478 | **2.007** | Vanilla, on **every** chunk (+0.11 … +0.61) |
| UNSEEN (code) — mean loss/chunk | 9.947 | **9.572** | Vanilla, on **every** chunk |
| in-context adaptation (chunk0→15) | +0.072 / +0.424 | **+0.575 / +0.694** | Vanilla adapts **more** |

The verdict is clean and points one way. The vanilla Transformer wins on **everything**:
absolute loss, per-chunk loss on both the seen and the unseen domain, and — the decisive
part — **in-context adaptation itself**. Full attention specializes to the context *as it
reads* better than Burel's Test-Time-Training memory does, for free and at lower cost (no
second-order gradients, fully parallel, fast attention kernels).

### Why

Burel is not "vanilla + memory". It is "vanilla with **amputated** attention (local to each
16-token chunk) + an expensive memory meant to stitch the cut back together." This result
says the stitching does not recover what the cut threw away. Burel traded a cheap, powerful
mechanism (full attention) for an expensive, weaker one (TTT).

### The one honest caveat (it does not rescue Burel here)

This test runs at a **256-token context**. The real case for memory architectures (Titans)
is **long context** — thousands of tokens — where full attention gets expensive (O(n²), the
KV-cache blows up). At 256 tokens attention is trivially cheap and clearly better, so Burel
had no opening here. The only regime where memory might still pay is the long-context
**cost-vs-quality frontier** (giving attention a limited context budget). That is a different,
narrow, and still-unproven experiment — not what Burel is today.

## Decision (pre-registered)

The plan said it in advance: **if the A/B falsifies the thesis, scale the vanilla, not
Burel.** So:

- the **vanilla (val 2.00)** is the better, cheaper base — growth continues from there;
- the levers for "capable + cheap to train + cheap to run" are **MoE + retrieval + data
  quality**, on top of the vanilla;
- **Test-Time Training becomes a research branch**, to be revisited only in the long-context
  regime (a possible future experiment 03).

A negative result is a result. The nested-learning core was a genuine, falsifiable bet; the
data did not back it at this scale, and we say so.

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
