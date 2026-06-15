# burel/paths.py — risoluzione centralizzata dei percorsi del progetto.
# Tutto e' relativo alla radice del repo, indipendente dalla cwd da cui lanci.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../Burel
CACHE_DIR = ROOT / "data_cache"                      # dataset preparati (.bin, meta.pkl)
CHECKPOINT_DIR = ROOT / "checkpoints"                # checkpoint locali (.pt)
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"    # config di default


def resolve(path, base=ROOT):
    """Rende assoluto un percorso eventualmente relativo alla radice del progetto."""
    p = Path(path)
    return p if p.is_absolute() else base / p
