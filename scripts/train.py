#!/usr/bin/env python
"""Addestra Burel. Riprende in automatico da burel_last.pt se esiste.

    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from burel.paths import DEFAULT_CONFIG
from burel.training import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    main(args.config)
