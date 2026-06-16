# Experiments — a growing, reproducible log

This folder is Burel's **open lab notebook**. Every experiment lives in its own numbered
folder with three things, so anyone can read it and rerun it:

1. a **README** that explains, in plain English, *what we asked, how we tested it, and
   what we found* — including the honest caveats;
2. a **Colab notebook** (`*_Colab.ipynb`) that runs the experiment end to end;
3. a **self-contained zip** (`*.zip`) with exactly the code that experiment needs — drop
   it in your Google Drive, open the notebook, press run.

No experiment is "marketing": a negative result is a result, and we say so.

## The experiments so far

| # | question | status | result (short) |
|---|----------|--------|----------------|
| [01 — TTT ablation](01_ttt_ablation/) | Does test-time learning (memory adapting while it reads) actually do anything? | ✅ done | **Yes, mechanism confirmed.** Memory adapting (ON) beats frozen memory (OFF) on every chunk, and improves *while reading* — even on a domain (code) the model never saw. But ON-vs-OFF is not the final word (see below). |
| [02 — nested vs vanilla](02_vanilla_ab/) | At **equal parameters and data**, does Burel's memory beat a plain Transformer's attention? | ⏳ running | results pending |

## Why experiment 02 matters most

Experiment 01 shows that turning Burel's memory *off* hurts a lot. But "Burel with memory
off" is a crippled model — in this architecture, memory is the **only** channel that
carries information between chunks (attention is chunk-local). So of course removing it
hurts.

The fair question is different: a **normal Transformer** adapts to context for free,
through full attention over the whole window. So the real test (experiment 02) is
*memory vs attention* at **equal parameter count** — a plain GPT trained on the same data.
If Burel's memory beats it, the bet pays off. If the plain Transformer matches or wins,
the honest move is to scale the plain Transformer instead. That's what 02 measures.

## How the zips are made (keeping them fresh)

The zips are snapshots of the repo's source for convenience. To rebuild them after the
code changes, from the repo root run:

```bash
bash make_bundles.sh
```

This regenerates the zip in each experiment folder from the current `burel/`,
`study_vanilla/`, `scripts/` and `configs/`.

## How to add the next experiment

Copy the pattern: `experiments/NN_short_name/` with a README (question → method →
result → how to reproduce), a Colab notebook, and a zip built by `make_bundles.sh`.
One change at a time, judged on the data — same rule as the rest of the project.
