#!/usr/bin/env python
"""Self-test for the byte-level BPE tokenizer and codec, runnable wherever
`tokenizers` is installed (e.g. Colab). It checks the properties we rely on:

  1. lossless round-trip: decode(encode(x)) == x  (byte-level loses nothing);
  2. all ids fall within the vocabulary and fit in uint16;
  3. the <|endoftext|> token exists and has a valid id.

It does not download TinyStories: it trains a tiny BPE on synthetic text, so it
runs in a second. If `tokenizers` is missing, the test is skipped (not failed).

    python tests/test_tokenizer_roundtrip.py
"""

import pathlib
import sys

# Make the repo root importable so `burel` resolves regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError:
        # `tokenizers` is an optional dependency; skip gracefully when absent.
        print("SKIP: `tokenizers` non installato (pip install tokenizers).")
        return

    # Reuse the project's end-of-sequence special token so ids match production.
    from burel.data.tinystories import EOS

    # Small synthetic corpus repeated to give the BPE trainer enough material.
    corpus = [
        "Once upon a time there was a little cat named Tom.",
        "The cat liked to play in the garden every morning.",
        "One day, Tom found a red ball and was very happy.",
        "He played with the ball until the sun went down. The end.",
    ] * 50

    # Byte-level BPE: every byte is representable, so encoding can never fail.
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=2000, special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    eos_id = tok.token_to_id(EOS)
    vocab = tok.get_vocab_size()
    assert eos_id is not None, "EOS mancante"
    # Token ids are stored as uint16 in the .bin files, so the vocab must fit.
    assert vocab <= 65535, f"vocab {vocab} non entra in uint16"

    # Lossless round-trip on in-domain and out-of-domain text (byte-level => always ok).
    for s in ["Once upon a time, the cat ran!", "Zxq-42 \n\t £€ unseen chars 9876"]:
        ids = tok.encode(s).ids
        assert all(0 <= i < vocab for i in ids), "id fuori range"
        back = tok.decode(ids)
        assert back == s, f"round-trip rotto:\n  in : {s!r}\n  out: {back!r}"

    print(f"OK: vocab={vocab}, eos_id={eos_id}, round-trip lossless, id in uint16.")


if __name__ == "__main__":
    main()
