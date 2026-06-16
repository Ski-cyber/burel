# Model — Burel weights

The trained weights are too large to live in the Git tree, so they are distributed as
**GitHub Release assets** (no bandwidth limits, direct download).

## Download

➡️ Go to the [**Releases page**](../../../releases) and download:

| file | size | description |
|------|------|-------------|
| `modello_burel_best_v2.pt` | ~231 MB | **recommended model** — 20M params, TinyStories + BPE 16k, best val loss 2.48 |
| `modello_burel_best_v1.pt` | ~54 MB | previous version — 14M params, TinyShakespeare char-level, best val loss 1.54 |

## Usage

```python
from burel.inference import load_model, generate_text

# point at the .pt downloaded from the Releases
model, meta = load_model("modello_burel_best_v2.pt")
print(generate_text(model, meta, prompt="Once upon a time", max_new_tokens=500))
```

Or from the command line:

```bash
python scripts/generate.py --prompt "Once upon a time" --tokens 500 --temperature 0.8
```

> Note: the checkpoint contains `model` + `optimizer` + `iter` + `meta` (data config and
> tokenizer), so inference already knows the model's shape and how to decode the tokens.
