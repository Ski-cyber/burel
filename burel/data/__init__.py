# burel/data/ — preparazione dati e codec, disaccoppiati dal modello.
#
# prepare(cfg) sceglie il dataset in base a configs/config.yaml -> data.name:
#   - "shakespeare": char-level (fase 1, baseline);
#   - "tinystories": BPE byte-level (fase 2).
# Entrambi producono lo stesso contratto su disco (train.bin/val.bin uint16 +
# meta.pkl), quindi trainer.py e inference NON cambiano.

import yaml

from burel.paths import DEFAULT_CONFIG
from .codec import decode, encode
from .shakespeare import URL, load_meta, load_split
from .shakespeare import prepare as _prepare_shakespeare
from .tinystories import prepare as _prepare_tinystories

__all__ = ["prepare", "load_meta", "load_split", "encode", "decode", "URL"]


def prepare(cfg=None):
    """Prepara il dataset scelto in config. Idempotente: i moduli rigenerano la
    cache solo se la config dati e' cambiata. cfg=None -> carica DEFAULT_CONFIG."""
    if cfg is None:
        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
    dc = (cfg or {}).get("data", {}) or {}
    name = dc.get("name", "shakespeare")

    if name == "shakespeare":
        kw = {k: dc[k] for k in ("val_frac",) if k in dc}
        return _prepare_shakespeare(**kw)
    if name == "tinystories":
        keys = ("vocab_size", "max_train_tokens", "max_val_tokens",
                "tokenizer_sample_docs", "batch", "drive_cache_dir")
        kw = {k: dc[k] for k in keys if k in dc}
        return _prepare_tinystories(**kw)
    raise ValueError(f"data.name sconosciuto: {name!r} (usa 'shakespeare' o 'tinystories')")
