# burel/data/shakespeare.py
#
# Simple starter dataset: TinyShakespeare, char-level.
# No external tokenizer: the vocabulary is the set of unique characters in the text.
# Produces train.bin / val.bin (uint16) + meta.pkl (stoi/itos) in data_cache/.

import pickle

import numpy as np
import requests

from burel.paths import CACHE_DIR

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(url=URL, cache_dir=CACHE_DIR, val_frac=0.1):
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "input.txt"

    # Download once and cache the raw text locally; reuse it on later runs.
    if not raw_path.exists():
        print(f"Scarico il dataset da {url} ...")
        text = requests.get(url, timeout=30).text
        raw_path.write_text(text)
    else:
        text = raw_path.read_text()

    # Build the char-level vocabulary and the char<->id maps (sorted for determinism).
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    # Encode the whole text as uint16 ids, then split off the last val_frac as validation.
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n_val = int(len(data) * val_frac)
    train_data, val_data = data[:-n_val], data[-n_val:]

    train_data.tofile(cache_dir / "train.bin")
    val_data.tofile(cache_dir / "val.bin")
    with open(cache_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": "char",
                     "stoi": stoi, "itos": itos}, f)

    print(f"OK: {len(text):,} caratteri, vocab={vocab_size}, "
          f"train={len(train_data):,}, val={len(val_data):,}")
    return vocab_size


# Load meta.pkl (vocab_size, encoding, and the codec maps). Shared by both datasets.
def load_meta(cache_dir=CACHE_DIR):
    with open(cache_dir / "meta.pkl", "rb") as f:
        return pickle.load(f)


# Memory-map a split's .bin (no full load into RAM); the trainer indexes into it.
def load_split(split, cache_dir=CACHE_DIR):
    return np.memmap(cache_dir / f"{split}.bin", dtype=np.uint16, mode="r")
