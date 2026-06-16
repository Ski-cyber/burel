#!/usr/bin/env python
"""Train Burel. Automatically resumes from burel_last.pt if it exists.

    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import pathlib
import sys

# Make the repo root importable so `burel` resolves regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from burel.paths import DEFAULT_CONFIG
from burel.training import main

if __name__ == "__main__":
    # CLI: accept an optional path to a YAML config, defaulting to the project config.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    # Delegate the whole training loop (data, model, optimizer, checkpointing) to burel.training.
    main(args.config)
