# Experiment 01 — Test-Time Training ablation (memory ON vs OFF)

**Date:** 2026-06-16 · **Model:** `burel_best.pt` (20M, TinyStories BPE, val 2.4831)

## The question, in one line

Burel's selling point is that its memory keeps **learning while it reads** (Test-Time
Training, "TTT"). Does that adaptation actually help — or is it dead weight?

## How we tested it (and why it's clean)

We did the cheapest possible test: **no new model, one switch.** Burel already has a flag,
`update_memory_in_inference`. We run the exact same model on the exact same text windows,
twice:

- **ON** — the memory adapts after every chunk (normal Burel).
- **OFF** — the memory is frozen; it can still be *read*, but it no longer learns.

Why this isolates the thing we care about: in Burel, information passes from one chunk to
the next **only through the memory** (attention is local to each 16-token chunk). So the
only difference between ON and OFF is the test-time learning itself.

We measured the loss **per chunk** (lower = better prediction) as the model moves deeper
into a 256-token window, on two kinds of text:

- **CONTROL** — TinyStories validation (a domain the model was trained on);
- **UNSEEN** — Python source code (a domain it has *never* seen).

If the memory truly learns on the fly, the ON curve should drop *as it reads*, especially
on the unseen domain. Script: [`scripts/ablation_ttt.py`](../../scripts/ablation_ttt.py).

## What we found

| domain | loss ON | loss OFF | benefit (OFF−ON) | adaptation while reading (chunk0 → chunk15) |
|--------|--------:|---------:|-----------------:|---------------------------------------------|
| CONTROL (seen) | 2.478 | 2.812 | **+0.333** (≈12%) | ON: loss **improves** +0.07 · OFF: **worsens** −0.30 |
| UNSEEN (code) | 9.914 | 10.551 | **+0.638** (≈6%) | ON: loss **improves** +0.61 · OFF: **worsens** −0.10 |

Two clear signals:

1. **The mechanism is real.** With memory ON the loss gets *better* the further the model
   reads; with memory OFF it doesn't. Since memory is the only cross-chunk channel, this is
   direct evidence that the test-time learning is doing real work.
2. **It generalizes to an unseen domain.** It even kicks in on code, which the model was
   never trained on — so it's not just memorized behavior.

A built-in sanity check: at chunk 0 (before any update) ON and OFF are *identical*,
confirming the only difference is the memory updates.

## The honest caveats (read these)

- **Proof of mechanism, not of capability.** On code the loss is ~9.9 — the model is
  essentially clueless there (it's a 20M model trained on children's stories, with a
  story-trained tokenizer). TTT makes it *slightly less* clueless; it does **not** make it
  good at code.
- **In relative terms, the benefit is actually larger on the *seen* domain** (≈12% vs ≈6%).
  "Helps more on unseen" is true in absolute loss units but not after normalizing — don't
  oversell it.
- **OFF is not the real competitor.** "Burel with memory off" has no long-range channel at
  all (attention is chunk-local). A normal Transformer gets context adaptation for free via
  full attention. So this experiment proves *TTT ≫ Burel-without-TTT*, **not**
  *memory ≫ attention*. That comparison is [experiment 02](../02_vanilla_ab/).

## Reproduce it

**On Colab (recommended):** put `burel_ablation.zip` and `Burel_Ablation_Colab.ipynb`
(both in this folder) into your Google Drive, open the notebook, pick a **GPU (T4)** runtime,
and run the cells. The notebook finds your checkpoint and dataset on Drive automatically,
builds the "unseen" code file, and runs the ablation.

**Locally** (if you have the checkpoint and the BPE `data_cache`):

```bash
python scripts/ablation_ttt.py --ckpt checkpoints/burel_best.pt --domain_file your_code.txt
```
