# Burel

**A seed for a European, genuinely open AI.**

Burel is a language model built from scratch on an experimental architecture —
**HOPE** (Titans "Memory as Context" + **Nested Learning**) — with a stated and
ambitious goal: to show that serious language-model research can happen **outside the
big labs**, in a transparent, reproducible and shared way.

This repository is not a finished product. It is a **starting point**: code, model and
an experiment log made available to anyone who wants to contribute.

---

## Where we are and where we want to go

**Where we are.** Burel today is a ~20M-parameter model, **pretrained from scratch**.
It has learned to generate coherent short English stories (TinyStories dataset, BPE
tokenizer), with complete narrative arcs — beginning, development, "The end." — and
named characters. It is a competent newborn: small, but alive and genuinely learning.
The nested-learning core runs **at full strength**, not in a simplified version.

**Where we want to go.** The ambition is to build, over time and collaboratively,
**a European-level AI: open, inspectable, non-proprietary.** Not yet another clone
locked behind an API, but a model whose code, weights, choices and mistakes anyone can
see. Burel is the **seed** of this journey: an honest experiment, documented step by
step, asking to be picked up, criticized and grown by a community.

> **An honest position.** At this scale Burel is not, and will not be, GPT-4. The goal
> is not to compete on the absolute numbers of the giants, but to be the **best small,
> specialized model possible** — and, above all, an open base to build on together.

---

## What Nested Learning is (short and simple)

A normal Transformer learns once: you train it, **freeze the weights**, and from then
on, when it generates text, it learns nothing new — it only uses what it holds in fixed
memory.

**Nested Learning** flips this idea. The model has several **memory levels that keep
updating**, even *while it reads and generates* (this is called **Test-Time Training**).
It is like having, inside the model, "notebooks" running at different speeds:

- some change very slowly (general, stable knowledge);
- others update quickly, taking notes **on the fly** about what it is reading right now.

Technically: the memory updates via **second-order gradients** during inference, and
there is even a **learned optimizer** (the network learns *how* to update its own
memory, instead of using a fixed rule). Memory inside memory: "nested".

### The point — what I want to prove

The hypothesis Burel sets out to test is simple to state:

> **For the same number of parameters and the same data, does a model that keeps
> learning in the moment (nested / test-time training) understand and remember better
> than a classic Transformer with frozen weights?**

If the answer is yes — even just at small scale and in a measurable way — then it is an
interesting path toward **efficient** AI that does not need to be gigantic to be
capable: it learns a little even *while using*, the way we do. Burel exists to test this
intuition cleanly, **one variable at a time**, with the numbers on the table (see
`CHANGES.md`).

---

## The model (downloadable weights)

The trained weights are **not** in the repo tree (the file is too large for Git): they
are attached as a **GitHub Release**.

➡️ Download `modello_burel_best_v2.pt` (~231 MB) from the
[**Releases page**](../../releases) — see also [`model/README.md`](model/README.md).

| version | parameters | dataset | result |
|---------|-----------|---------|--------|
| v1 | 14M | TinyShakespeare (char) | best val loss **1.5429** |
| **v2** | **20M** | **TinyStories (BPE 16k)** | **best val loss 2.4831** — coherent stories, complete arc |

The full experiment log (what changed, why, and whether it improved or worsened things)
is in [`CHANGES.md`](CHANGES.md).

---

## Layout

```
Burel/
├── pyproject.toml          # installable package (pip install -e .)
├── requirements.txt
├── LICENSE                 # Apache 2.0
├── configs/
│   └── config.yaml         # hyper-parameters (tuned for Colab T4/L4 + small budget)
├── notebooks/
│   └── Burel_Colab.ipynb   # end-to-end Colab launcher
├── burel/                  # the package
│   ├── paths.py            # centralized path resolution
│   ├── model/
│   │   ├── layers.py       # Encoder, PositionalEncoding, AttentionPooling
│   │   ├── memory.py       # NESTED LEARNING: FunctionalMemory, DeepOptimizer, CMS
│   │   └── hope.py         # BurelLM (Transformer + CMS, causal)
│   ├── data/
│   │   ├── shakespeare.py  # phase 1: TinyShakespeare char-level
│   │   ├── tinystories.py  # phase 2: TinyStories + byte-level BPE
│   │   └── codec.py        # decoupled encode/decode (char or BPE from meta.pkl)
│   ├── training/
│   │   └── trainer.py      # loop, resume, best/last checkpoints + Drive backup
│   └── inference/
│       └── sampler.py      # load_model, generate_text
├── study_vanilla/         # SEPARATE parameter-matched vanilla Transformer (the A/B baseline)
│   ├── model.py           # VanillaGPT (~20.4M params, full attention, no memory)
│   ├── train.py           # nanoGPT-style trainer (Drive backup + resume)
│   └── eval_loss_by_chunk.py
├── experiments/           # open lab notebook: each test has a README + Colab notebook + zip
│   ├── 01_ttt_ablation/   # memory ON vs OFF (done, with results)
│   └── 02_vanilla_ab/     # nested vs vanilla at equal params (in progress)
├── make_bundles.sh        # rebuild the experiments' reproducibility zips
├── scripts/
│   ├── prepare_data.py     # python scripts/prepare_data.py
│   ├── train.py            # python scripts/train.py
│   ├── generate.py         # python scripts/generate.py --prompt "..."
│   ├── ablation_ttt.py     # experiment 01: Test-Time Training ON/OFF ablation
│   └── compare_three.py    # experiment 02: Burel-ON / Burel-OFF / Vanilla, same windows
└── tests/
    ├── test_causal.py              # strict-causality proof (max|delta|=0)
    └── test_tokenizer_roundtrip.py # lossless BPE round-trip
```

## What was adapted from the original architecture (and what was NOT)

The HOPE architecture was born in a financial model; Burel brings it to **language**.

**Unchanged** (`burel/model/memory.py`): the nested-learning core — multi-level
continuous memory with exponential update frequencies, a learned optimizer
(`DeepOptimizer`), fast weights, Test-Time Training with `create_graph=True`.

**Adapted to language** (`burel/model/hope.py`):
- input: `nn.Embedding` of discrete tokens instead of `nn.Linear` over continuous features;
- output: `lm_head` → logits over the vocabulary instead of price regression;
- loss: cross-entropy instead of Huber/SMAPE;
- removed the financial multi-task heads;
- added causal masking in the augmented attention layer;
- GPT-style init (std 0.02) on the backbone, leaving the CMS untouched.

## Local quickstart

```bash
pip install -e .                 # install Burel + dependencies
python scripts/prepare_data.py   # download and prepare the dataset (once)
python scripts/train.py          # train (GPU if available)
python scripts/generate.py --prompt "Once upon a time" --tokens 500 --temperature 0.8
```

Programmatic use of inference (after downloading the `.pt` from the Releases):

```python
from burel.inference import load_model, generate_text
model, meta = load_model("modello_burel_best_v2.pt")
print(generate_text(model, meta, prompt="Once upon a time", max_new_tokens=500))
```

## Colab

Open `notebooks/Burel_Colab.ipynb`, select a GPU runtime (T4), run the cells: it mounts
Drive, finds and unzips `Burel.zip`, runs `pip install -e .`, prepares, trains, generates.

### Checkpoints and resume (long training runs)

`scripts/train.py` saves at every `eval_interval`, locally **and** on Drive (if mounted):
`burel_best.pt` (best val loss) and `burel_last.pt` (full state for resuming).
With `resume: auto` (default), relaunching training resumes exactly where it stopped.
**Early stopping** via `patience: N` stops at the plateau.

## Causality — strict and verified

The model is **strictly causal**: a position never sees future tokens.
- intra-chunk attention and augmented layer: causal (masked);
- memory retrieval: chunk i's query comes from chunk i-1 (or from `init_query`),
  never from the current chunk → no intra-chunk leak;
- chunk i's memory update feeds only the subsequent chunks.

`python tests/test_causal.py`: scrambling the tokens at positions > t leaves the logits
at positions ≤ t bit-identical (`max|delta| = 0`).

## Roadmap

1. **[done]** architecture + training (resume, early stop) + inference, causality verified.
2. **[done]** Phase 1 — TinyShakespeare char-level: plausible English at character level.
3. **[done]** Phase 2 — TinyStories + BPE: coherent stories with a complete arc (v2, val 2.48).
4. **[done]** Experiment 01 — **Test-Time Training ablation** (memory ON vs OFF): the
   nested memory's test-time learning is confirmed to do real work, and it generalizes to
   an unseen domain. See [`experiments/01_ttt_ablation/`](experiments/01_ttt_ablation/).
5. **[in progress]** Experiment 02 — rigorous A/B **nested vs vanilla Transformer** at equal
   parameters/data → the central test of the hypothesis. See
   [`experiments/02_vanilla_ab/`](experiments/02_vanilla_ab/) and
   [`study_vanilla/`](study_vanilla/).
6. **[future]** larger corpora, code-gen, and community-driven growth.

## Experiments & reproducibility

Burel keeps an **open lab notebook** in [`experiments/`](experiments/). Each experiment
folder has a plain-English README (question → method → result, caveats included), a
**Colab notebook**, and a **self-contained zip** — drop the zip in your Google Drive, open
the notebook, press run. You don't even need to clone the repo to reproduce a result.

- [**01 — Test-Time Training ablation**](experiments/01_ttt_ablation/): is the nested
  memory's "learning while reading" real? We flip one switch (memory ON vs OFF) on the same
  model and windows. **Result:** yes — the memory adapts as it reads (it improves deeper
  into the text; with the memory frozen it doesn't), and it generalizes to unseen code. With
  honest caveats: it's a proof of *mechanism*, and the ON-vs-OFF gap isn't the final word.
- [**02 — nested vs vanilla**](experiments/02_vanilla_ab/) *(in progress)*: the decisive
  test. A plain Transformer with the **same parameter count and data** is the real opponent.
  If Burel's memory beats it, the bet pays off; if not, we scale the plain Transformer. We
  say which, on the numbers.

To rebuild the zips after changing the code: `bash make_bundles.sh`.

## Contributing

Burel is open **on purpose**, to be picked up. Issues, forks, experiments, critiques of
the numbers and PRs are all welcome. The project rule: **one change at a time, judged on
the data** — every change goes in `CHANGES.md`, saying whether it improves or worsens
things, and why.

## Contact

For information, collaboration, or to join the development:
**Giovanni Canclini** — [info@giovannicanclini.com](mailto:info@giovannicanclini.com)

## License

[Apache License 2.0](LICENSE) — free use, including commercial, with an explicit patent
grant. Code and weights are open: that is the whole point.
