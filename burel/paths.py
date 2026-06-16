# burel/paths.py — centralized resolution of the project's paths.
# Everything is relative to the repo root, independent of the cwd you launch from.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../Burel
CACHE_DIR = ROOT / "data_cache"                      # prepared datasets (.bin, meta.pkl)
CHECKPOINT_DIR = ROOT / "checkpoints"                # local checkpoints (.pt)
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"    # default config


def resolve(path, base=ROOT):
    """Make a path absolute, treating relative paths as relative to the project root."""
    # Absolute paths are returned as-is; relative ones are anchored under `base`.
    p = Path(path)
    return p if p.is_absolute() else base / p
