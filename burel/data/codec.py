# burel/data/codec.py
#
# Codec testo <-> id, DISACCOPPIATO dal modello. Serve a tenere un solo sampler
# valido sia per il char-level (Shakespeare) sia per il BPE (TinyStories, e poi
# il codice). Il modello non cambia: vede sempre interi; qui si traduce.
#
# Il tipo di codifica e' scritto in meta.pkl ("encoding"):
#   - "char": usa stoi/itos (vecchio comportamento, identico bit-per-bit);
#   - "bpe" : carica data_cache/<tokenizer_file> con la libreria `tokenizers`.

from functools import lru_cache

from burel.paths import CACHE_DIR


def encode(meta, text):
    """Testo -> lista di id interi, secondo l'encoding dichiarato in meta."""
    enc = meta.get("encoding", "char")
    if enc == "char":
        stoi = meta["stoi"]
        return [stoi.get(c, 0) for c in text]
    if enc == "bpe":
        tok = _load_tokenizer(meta.get("tokenizer_file", "tokenizer.json"))
        return tok.encode(text).ids
    raise ValueError(f"encoding sconosciuto in meta: {enc!r}")


def decode(meta, ids):
    """Lista di id interi -> testo, secondo l'encoding dichiarato in meta."""
    enc = meta.get("encoding", "char")
    if enc == "char":
        itos = meta["itos"]
        return "".join(itos[int(i)] for i in ids)
    if enc == "bpe":
        tok = _load_tokenizer(meta.get("tokenizer_file", "tokenizer.json"))
        return tok.decode([int(i) for i in ids])
    raise ValueError(f"encoding sconosciuto in meta: {enc!r}")


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_file):
    # Import locale: `tokenizers` serve solo per i dataset BPE, non per il
    # char-level. Risolto sempre dentro CACHE_DIR -> portabile tra locale/Drive/Colab.
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(CACHE_DIR / tokenizer_file))
