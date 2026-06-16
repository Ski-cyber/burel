# burel/data/ — data preparation and codec, decoupled from the model.
#
# prepare(cfg) picks the dataset based on configs/config.yaml -> data.name:
#   - "shakespeare": char-level (phase 1, baseline);
#   - "tinystories": byte-level BPE (phase 2).
# Both produce the same on-disk contract (train.bin/val.bin uint16 + meta.pkl),
# so trainer.py and inference DON'T change.

import yaml

from burel.paths import DEFAULT_CONFIG
from .codec import decode, encode
from .shakespeare import URL, load_meta, load_split
from .shakespeare import prepare as _prepare_shakespeare
from .tinystories import prepare as _prepare_tinystories

__all__ = ["prepare", "load_meta", "load_split", "encode", "decode", "URL"]


def prepare(cfg=None):
    """Prepare the dataset selected in config. Idempotent: the modules regenerate
    the cache only if the data config has changed. cfg=None -> load DEFAULT_CONFIG."""
    if cfg is None:
        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
    dc = (cfg or {}).get("data", {}) or {}
    name = dc.get("name", "shakespeare")

    # Dispatch to the right preparer, forwarding only the config keys it accepts.
    if name == "shakespeare":
        kw = {k: dc[k] for k in ("val_frac",) if k in dc}
        return _prepare_shakespeare(**kw)
    if name == "tinystories":
        keys = ("vocab_size", "max_train_tokens", "max_val_tokens",
                "tokenizer_sample_docs", "batch", "drive_cache_dir")
        kw = {k: dc[k] for k in keys if k in dc}
        return _prepare_tinystories(**kw)
    raise ValueError(f"data.name sconosciuto: {name!r} (usa 'shakespeare' o 'tinystories')")
