# CHANGES — Burel

A concise, numbered list of every change to the project. Its purpose is to track
whether each change makes things better or worse. Update it on every change.

## Project changes (chronological)

1. **Birth of Burel** — took the HOPE model (Titans MAC + Nested Learning) from the
   financial project ModelMango and adapted it to language: input `Embedding` of tokens
   (instead of `Linear` over features), output `lm_head` over the vocabulary, cross-entropy
   loss, removed the financial heads. The nested-learning core (CMS, DeepOptimizer,
   Test-Time Training) was NOT touched.
2. **Causal masking** added in the augmented attention layer (required by the LM).
3. **Causality gap closed** — the memory retrieval query comes from the PREVIOUS chunk
   (not the current one); the first chunk uses `init_query`. Strictly causal, verified
   `max|delta| = 0`.
4. **Inference** — `inference/sampler.py` (`load_model`, `generate_text`) + `scripts/generate.py`.
5. **GPT-style weight init** (std 0.02) on the backbone, CMS excluded → initial loss from
   12.95 down to ~4.17.
6. **Progress bar** (tqdm) with %, ETA, current loss.
7. **Resume + checkpoint** — saves `burel_best.pt` and `burel_last.pt` (model+optimizer+iter),
   locally and on Drive; `resume: auto` resumes from where it stopped.
8. **Version-aware GradScaler** — compatible with multiple torch versions, fp16 behavior
   unchanged.
9. **Colab launcher** (`notebooks/Burel_Colab.ipynb`) — finds the zip, mounts Drive, installs,
   prepares, trains, generates.
10. **Enterprise refactor** — `burel/` package (model/data/training/inference) + scripts/,
    configs/, tests/, notebooks/, pyproject.toml. Model verified bit-identical to the pre-refactor.
11. **Early stopping** with `patience` — stops after N evals with no new best, tolerates noise.
12. **Phase 2 — TinyStories dataset + BPE tokenizer.** New module `burel/data/tinystories.py`
    (HuggingFace `roneneldan/TinyStories`) with **byte-level** BPE (the `tokenizers` library):
    no UNK, reusable as-is for code. Between one story and the next we insert
    `<|endoftext|>` -> the model learns the beginning-development-end boundaries. A token cap
    (`max_train_tokens`, default 100M) avoids over-provisioning data. **The core
    (`memory.py`, `hope.py`, `layers.py`) was NOT touched.** Same on-disk contract
    (train.bin/val.bin uint16 + meta.pkl) -> the trainer does not change (a 16k vocab fits in uint16).
13. **Decoupled text<->id codec** (`burel/data/codec.py`): `encode`/`decode` pick char-level
    or BPE from the `encoding` field in meta.pkl. `inference/sampler.py` now uses it instead
    of inline stoi/itos -> a single sampler valid for both datasets; Shakespeare stays
    bit-for-bit identical.
14. **Dataset selectable from config** (`data.name` in config.yaml) with dispatch in
    `burel/data/__init__.py`. `prepare(cfg)` is idempotent (rebuilds only if the data config
    changes); the trainer always calls it -> switching dataset = changing only the config.
15. **Dependencies**: added `tokenizers` and `datasets` (requirements.txt + pyproject.toml).
    New self-test `tests/test_tokenizer_roundtrip.py` (lossless round-trip, ids in uint16,
    EOS present; skipped if `tokenizers` is missing).
16. **Dataset cache on Drive** (`data.drive_cache_dir`). `tinystories.prepare` restores
    train.bin/val.bin/meta.pkl/tokenizer.json from Drive if the cache matches (cache_key),
    instead of re-downloading and re-encoding; on the first build it copies them to Drive.
    Prepare ONCE, every retrain reuses. The notebook points the cache at the user's Drive
    (`.../Burel_data`).
17. **FINDING — `batch_size` breaks nested-memory optimization.** First TinyStories run at
    batch 48 (to fill the T4): total stall at val ~5.97 (unigram baseline), train flat for
    4000 iters, INSENSITIVE to the LR (3e-4 and 6e-4 the same). Back to the proven v2 regime
    (**batch 16**, lr 3e-4) changing ONLY the dataset: the model learns immediately
    (val 5.34 at iter 500 and decreasing). Conclusion: the 2nd-order gradients of TTT do not
    tolerate a large batch the way a vanilla Transformer does. **The "fill VRAM with the
    batch" lever is OFF for this architecture**; batch scaling must be treated as its own
    experiment (one variable, with LR re-tuning), not as a freebie. Note for the nested-vs-
    vanilla thesis (step 2): this batch sensitivity is a characteristic of the architecture.

## Config experiments (lower val loss is better)

| version | parameters | key hyper-parameters | best val loss |
|---------|-----------|----------------------|---------------|
| **v1** (baseline) | 4,684,301 | d_model 256, layers 4, ff 1024, dropout 0.1, ctx 256 | **1.6196** (iter 4250) |
| **v2** | 14,052,877 | d_model 384, layers 6, ff 1536, dropout 0.2 | **1.5429** (iter 7500, full run 8000) → **BETTER than v1 (−0.077, ~5%)** |
| **v3a** | 20,171,917 | **TinyStories + BPE 16k**, arch = v2, **batch 48**, lr 3e-4→6e-4, block 256 | **STALL** at val ~5.97 (= unigram baseline), train flat for 4000 iters |
| **v3b** | 20,171,917 | like v3a but **batch 16** (proven v2 regime), lr 3e-4 | **PHASE 2 OK** — best val **2.4831** @ iter 19250 (from 9.67). Run stopped manually at ~iter 19550 (not a full plateau but a diminishing-returns tail, ~0.015 val/1000 iters). Samples at 2.48 visibly cleaner than at 2.58: stories with a complete arc, "The end.", named characters. Ceiling: multi-entity identity drift (cat→dog→Spot→Max), non-sequitur = the 20M limit, not fixable with more iters on TinyStories |

> ⚠️ **v3 — NOT a direct comparison with v1/v2.** It changes the dataset AND the tokenizer
> (char→BPE 16k): val loss in nats/BPE-token **is not comparable** with char-level val loss
> (different units, ~1 BPE token ≈ several characters). v3 is judged **in absolute terms**:
> (a) a loss curve that descends and plateaus, (b) **samples readable by eye** — real
> sentences, coherent characters, beginning-to-end. The verdict here is qualitative
> (meaning), not a numeric Δ against Shakespeare.

**v1→v2 outcome:** v2 wins clearly (1.5429 vs 1.6196, −0.077, ~5%), and the margin grew with
training (at the crossover it was ~1%, at the end ~5%): the extra capacity paid off over time.
Visibly better samples (more real words, fewer invented tokens). It did NOT break 1.5 (the
dataset's entropy floor, as expected). Final train/val gap ~0.21, similar to v1: dropout did
not shrink the final gap but allowed going lower without an overfitting blow-up. Cost: 3x
parameters, ~4x time. Lesson: capacity helps but with diminishing returns on Shakespeare; the
real jump is the DATASET. Note: the T4 was using only ~3/15 GB -> huge headroom for larger
models/batches in the next runs.

### Config changes v1 → v2 (12)

| parameter | v1 | v2 | why |
|-----------|----|----|-----|
| `d_model` | 256 | 384 | more capacity (lowers train and val) |
| `num_encoder_layers` | 4 | 6 | more depth |
| `dim_feedforward` | 1024 | 1536 | scales with d_model (~4x) |
| `dropout` | 0.1 | 0.2 | closes the train/val gap (~0.22 in v1) |
| `warmup_iters` | 100 | 200 | larger model, softer start |
| `max_iters` | 5000 | 8000 | higher cap; patience stops at the plateau |

Unchanged (nested-learning regime, not to be touched so results stay attributable): `chunk_size 16`,
`num_mem_levels 3`, `memory_depth 2`, `mem_lr 1e-4`, `persistent_length 4`, `max_memory_length 256`,
`block_size 256`, `batch_size 16`, `learning_rate 3e-4`, `min_lr 3e-5`, `grad_clip 1.0`.

**Expected v2:** ~1.50-1.55. Beyond that, on Shakespeare char-level, it's a wall: the dataset must change.

### Config changes v2 → v3 (phase 2)

| parameter | v2 | v3 | why |
|-----------|----|----|-----|
| `data.name` | (shakespeare char) | **tinystories** | the real jump: from "imitate the form" to "say things with meaning" |
| tokenizer | char-level (~65) | **byte-level BPE 16k** | semantic units, reusable for code; +~6M embedding params |
| `batch_size` | 16 | **16** (tried 48 → STALL, see #17) | batch 48 freezes the nested optimization; 16 is the proven regime |
| `max_iters` | 8000 | **40000** | much larger corpus; high cap, stopped by hand at the tail |
| `eval_interval` | 250 | **250** | fine reads for diagnosis |
| `sample.start` | `\n` | **"Once upon a time"** | natural opening for the stories |

**Architecture IDENTICAL to v2** (d_model 384, layers 6, ff 1536, dropout 0.2) and the
**nested-learning regime untouched** (`chunk_size 16`, `num_mem_levels 3`, `memory_depth 2`,
`mem_lr 1e-4`, `persistent_length 4`, `max_memory_length 256`, `block_size 256`, `lr 3e-4`).
Deliberate choice: we fill the T4 with the **batch**, not the parameters, so the only variable
versus v2 is the **dataset** → an attributable verdict. Scaling up the model (d_model/layers) is
deferred to a possible v4, after validating that TinyStories delivers the meaning jump
(validate-before-scaling).

**Expected v3:** readable samples (coherent sentences/characters, beginning-to-end). No
knowledge/facts, no instruction following, a children's-story vocabulary. This is "elementary
school": it validates meaning and breaks in the BPE/scale pipeline for code (step 3). The
stories are short → it does not yet stress the nested memory (which shines at long context, step 2/3).

### Phase 2 — CLOSED (outcome)

**Goal reached.** Burel makes the "imitate the form" → "say things with meaning" jump: real
English and grammar, named characters, a story template (beginning-development-end + moral),
working `<|endoftext|>` boundaries. Best val **2.4831** (perplexity ~12), stopped by hand at a
diminishing-returns tail (the goal was meaning, not squeezing the loss). Deliverable:
`burel_best.pt` on Drive (`Burel/Burel_checkpoints/`).

**Proven ceiling:** a 20M model cannot hold a multi-entity world state (cat→dog→Spot→Max) — a
capacity limit, not a training one: it does not break with more iters on TinyStories, it needs
more scale + a richer dataset. The pipeline (BPE, data, Drive cache, scale, proven nested regime)
is **road-tested and reusable**.

**Key lessons:**
- #17: a large `batch_size` breaks nested-memory optimization (2nd-order TTT). batch 16.
- Change ONE variable from the last-known-good: I had changed dataset + batch together → the
  stall was diagnosed only by returning to the exact v2 regime.
- BPE val is not comparable with char val (different units): phase 2 was judged on the SAMPLES.

**Next (step 3, decided):** a code corpus (The Stack/CodeParrot) + BPE (reuse the pipeline) +
building a vanilla Transformer at equal parameters for the nested-vs-vanilla A/B, where the long
context of code finally stresses the nested memory. Data-level distillation from Qwen as an
accelerator; RL with verifiable reward (passing tests) only at the end.
