# burel/data/tinystories.py
#
# Dataset TinyStories (HuggingFace roneneldan/TinyStories) con tokenizer BPE
# ByteLevel. Stessa interfaccia su disco di shakespeare.py:
#   data_cache/train.bin, val.bin  (uint16)  + meta.pkl  + tokenizer.json
# => trainer.py e load_split/load_meta NON cambiano (vocab <= 16k entra in uint16).
#
# Scelte:
#   - BPE byte-level: nessun UNK (l'alfabeto copre tutti i 256 byte), riusabile
#     pari pari per il codice (passo 3) senza ridisegnare la pipeline;
#   - tra una storia e l'altra inseriamo <|endoftext|> -> il modello impara i
#     confini inizio-svolgimento-fine (esattamente cio' che vogliamo emergere);
#   - cap sui token (max_train_tokens): non sovra-approvvigioniamo dati che non
#     bruciamo in mezza giornata di T4, e teniamo leggeri download e backup Drive;
#   - cache_key in meta.pkl: prepare() e' idempotente, rigenera solo se la config
#     dati cambia (cosi' il trainer puo' chiamarlo sempre senza rifare il lavoro);
#   - cache su Drive (drive_cache_dir): si prepara UNA volta, e ogni retrain
#     ripristina gli artefatti da Drive invece di ri-scaricare e ri-encodare.

import pickle
import shutil
from pathlib import Path

import numpy as np

from burel.paths import CACHE_DIR

DATASET = "roneneldan/TinyStories"
EOS = "<|endoftext|>"
ARTIFACTS = ("train.bin", "val.bin", "meta.pkl", "tokenizer.json")


def prepare(cache_dir=CACHE_DIR, vocab_size=16000, max_train_tokens=100_000_000,
            max_val_tokens=1_000_000, tokenizer_sample_docs=200_000, batch=2000,
            drive_cache_dir=None):
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.pkl"
    tok_path = cache_dir / "tokenizer.json"

    cache_key = {
        "name": "tinystories", "vocab_size": vocab_size,
        "max_train_tokens": max_train_tokens, "max_val_tokens": max_val_tokens,
    }

    # 1) Cache locale gia' valida -> niente da fare.
    if _artifacts_valid(cache_dir, cache_key):
        vs = _vocab_from(cache_dir)
        print(f"data_cache TinyStories valido (vocab={vs}): salto la preparazione.")
        return vs

    # 2) Cache su Drive valida -> ripristino locale (niente ri-download/ri-encoding).
    drive = _drive_ready(drive_cache_dir)
    if drive and _artifacts_valid(drive, cache_key):
        print(f"Ripristino il dataset TinyStories da Drive ({drive}) ...")
        _copy_artifacts(drive, cache_dir)
        vs = _vocab_from(cache_dir)
        print(f"OK: dataset ripristinato da Drive (vocab={vs}), niente ri-encoding.")
        return vs

    # 3) Costruzione da zero.
    print(f"Alleno il tokenizer BPE (vocab~{vocab_size}) su <= {tokenizer_sample_docs:,} storie ...")
    tok = _train_tokenizer(vocab_size, _iter_text("train", tokenizer_sample_docs), tok_path)
    eos_id = tok.token_to_id(EOS)
    real_vocab = tok.get_vocab_size()
    assert eos_id is not None, "token EOS mancante nel tokenizer"
    assert real_vocab <= 65535, f"vocab {real_vocab} non entra in uint16"

    print("Encoding train ...")
    n_train = _encode_to_bin(_iter_text("train"), tok, eos_id,
                             cache_dir / "train.bin", max_train_tokens, batch)
    print("Encoding validation ...")
    n_val = _encode_to_bin(_iter_text("validation"), tok, eos_id,
                           cache_dir / "val.bin", max_val_tokens, batch)

    meta = {
        "vocab_size": real_vocab, "encoding": "bpe",
        "tokenizer_file": tok_path.name, "eos_id": eos_id,
        "cache_key": cache_key,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"OK TinyStories: vocab={real_vocab}, train={n_train:,} tok, val={n_val:,} tok")

    # 4) Backup su Drive: la prossima volta si ripristina, non si ricostruisce.
    if drive:
        print(f"Backup del dataset su Drive ({drive}) ...")
        _copy_artifacts(cache_dir, drive)
        print("Backup completato: il prossimo run riusa direttamente questi file.")

    return real_vocab


# --- streaming + encoding ---------------------------------------------------

def _iter_text(split, max_docs=None):
    """Itera i testi di uno split, in streaming (non scarica tutto in RAM)."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=split, streaming=True)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        t = ex["text"].strip()
        if t:
            yield t


def _train_tokenizer(vocab_size, texts, out_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # tutti i 256 byte -> niente UNK
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(out_path))
    return tok


def _encode_to_bin(texts, tok, eos_id, out_path, max_tokens, batch):
    """Encoda i testi a batch e li scrive in append come uint16. Inserisce EOS
    dopo ogni storia. Si ferma a max_tokens (None = nessun limite)."""
    n = 0
    buf = []

    def _flush(f, chunk):
        nonlocal n
        if not chunk:
            return False
        for enc in tok.encode_batch(chunk):
            ids = enc.ids
            ids.append(eos_id)
            a = np.asarray(ids, dtype=np.uint16)
            if max_tokens is not None and n + len(a) > max_tokens:
                a = a[:max_tokens - n]
                a.tofile(f)
                n += len(a)
                return True  # cap raggiunto
            a.tofile(f)
            n += len(a)
        return False

    with open(out_path, "wb") as f:
        for t in texts:
            buf.append(t)
            if len(buf) >= batch:
                stop = _flush(f, buf)
                buf = []
                if stop:
                    return n
        _flush(f, buf)
    return n


# --- cache locale / Drive ---------------------------------------------------

def _artifacts_valid(d, cache_key):
    """True se la cartella d contiene tutti gli artefatti e il cache_key combacia."""
    d = Path(d)
    if not all((d / n).exists() for n in ARTIFACTS):
        return False
    try:
        with open(d / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
    except Exception:
        return False
    return meta.get("cache_key") == cache_key


def _vocab_from(d):
    with open(Path(d) / "meta.pkl", "rb") as f:
        return pickle.load(f)["vocab_size"]


def _drive_ready(drive_cache_dir):
    """Ritorna il Path se Drive sembra montato (la cartella padre esiste), altrimenti None."""
    if not drive_cache_dir:
        return None
    p = Path(drive_cache_dir)
    return p if p.parent.is_dir() else None


def _copy_artifacts(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for n in ARTIFACTS:
        if (src / n).exists():
            shutil.copy2(src / n, dst / n)
