# study_vanilla/data.py
#
# Minimal, self-contained data access for the vanilla baseline. Reads the SAME on-disk
# contract Burel produces (data_cache/{train,val}.bin uint16 + meta.pkl + tokenizer.json),
# so the A/B is fair, but WITHOUT importing `burel` — this package stands on its own.

import pickle
from pathlib import Path

import numpy as np
import torch


def load_split(cache_dir, split):
    """Memory-map a token split ('train' or 'val') as a uint16 array."""
    path = Path(cache_dir) / f"{split}.bin"
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_meta(cache_dir):
    with open(Path(cache_dir) / "meta.pkl", "rb") as f:
        return pickle.load(f)


def get_batch(data, block_size, batch_size, device, generator=None):
    """Sample batch_size contiguous windows. x = block, y = block shifted by one.
    A shared generator makes the sampled windows reproducible across models (paired A/B)."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def encode_text(meta, text, cache_dir):
    """Encode raw text to token ids using the dataset's BPE tokenizer."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(cache_dir) / meta.get("tokenizer_file", "tokenizer.json")))
    return tok.encode(text).ids
