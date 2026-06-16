# burel/data/codec.py
#
# Text <-> id codec, DECOUPLED from the model. Lets us keep a single sampler that
# works for both char-level (Shakespeare) and BPE (TinyStories, and later code).
# The model doesn't change: it always sees integers; the translation happens here.
#
# The encoding type is recorded in meta.pkl ("encoding"):
#   - "char": uses stoi/itos (old behavior, bit-for-bit identical);
#   - "bpe" : loads data_cache/<tokenizer_file> via the `tokenizers` library.

from functools import lru_cache

from burel.paths import CACHE_DIR


def encode(meta, text):
    """Text -> list of integer ids, per the encoding declared in meta."""
    enc = meta.get("encoding", "char")
    if enc == "char":
        # Char-level: map each char via stoi, defaulting unknown chars to id 0.
        stoi = meta["stoi"]
        return [stoi.get(c, 0) for c in text]
    if enc == "bpe":
        tok = _load_tokenizer(meta.get("tokenizer_file", "tokenizer.json"))
        return tok.encode(text).ids
    raise ValueError(f"encoding sconosciuto in meta: {enc!r}")


def decode(meta, ids):
    """List of integer ids -> text, per the encoding declared in meta."""
    enc = meta.get("encoding", "char")
    if enc == "char":
        # Char-level: join the chars looked up via itos.
        itos = meta["itos"]
        return "".join(itos[int(i)] for i in ids)
    if enc == "bpe":
        tok = _load_tokenizer(meta.get("tokenizer_file", "tokenizer.json"))
        return tok.decode([int(i) for i in ids])
    raise ValueError(f"encoding sconosciuto in meta: {enc!r}")


# Cache loaded tokenizers so repeated encode/decode calls don't reread the file.
@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_file):
    # Local import: `tokenizers` is needed only for BPE datasets, not for the
    # char-level one. Always resolved inside CACHE_DIR -> portable across local/Drive/Colab.
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(CACHE_DIR / tokenizer_file))
