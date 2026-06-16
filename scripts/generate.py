#!/usr/bin/env python
"""Command-line inference.

    python scripts/generate.py
    python scripts/generate.py --prompt "ROMEO:" --tokens 800 --temperature 0.7 --top_k 100
"""

import argparse
import pathlib
import sys

# Make the repo root importable so `burel` resolves regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import yaml

from burel.inference import generate_text, load_model
from burel.paths import DEFAULT_CONFIG, resolve


def main():
    # First parse only --config, so the YAML can supply the defaults for the real args below.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()

    # Load sampling defaults from the config's [sample] section.
    cfg = yaml.safe_load(open(pre_args.config))
    sc = cfg["sample"]
    # By default load the best checkpoint saved under the training output directory.
    default_ckpt = resolve(cfg["train"]["out_dir"]) / "burel_best.pt"

    # Full parser: config-derived values act as defaults, overridable on the command line.
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--ckpt", default=str(default_ckpt))
    ap.add_argument("--prompt", default=sc["start"])
    ap.add_argument("--tokens", type=int, default=sc["max_new_tokens"])
    ap.add_argument("--temperature", type=float, default=sc["temperature"])
    ap.add_argument("--top_k", type=int, default=sc["top_k"])
    args = ap.parse_args()

    # Load the trained model plus its metadata (tokenizer/vocab info needed to decode).
    model, meta = load_model(args.ckpt)
    # Report which checkpoint we loaded (training iteration and its validation loss).
    info = model._ckpt_info
    print(f"checkpoint iter={info['iter']} val_loss={info['val_loss']:.4f}\n")

    # Autoregressively sample a continuation from the prompt and print it.
    text = generate_text(model, meta, prompt=args.prompt, max_new_tokens=args.tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(text)


if __name__ == "__main__":
    main()
