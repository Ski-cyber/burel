#!/usr/bin/env python
"""Prepara il dataset in data_cache/. Quale dataset lo decide configs/config.yaml
(data.name): "shakespeare" (char-level) o "tinystories" (BPE byte-level).
Idempotente: rigenera solo se la config dati e' cambiata."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from burel.data import prepare

if __name__ == "__main__":
    prepare()
