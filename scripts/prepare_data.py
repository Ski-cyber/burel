#!/usr/bin/env python
"""Prepare the dataset in data_cache/. Which dataset is chosen by configs/config.yaml
(data.name): "shakespeare" (char-level) or "tinystories" (BPE byte-level).
Idempotent: only regenerates if the data config has changed."""

import pathlib
import sys

# Make the repo root importable so `burel` resolves regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from burel.data import prepare

if __name__ == "__main__":
    # Download/tokenize the configured dataset and write the .bin + meta.pkl cache.
    prepare()
