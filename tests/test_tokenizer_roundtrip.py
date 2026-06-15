#!/usr/bin/env python
"""Self-test del tokenizer BPE byte-level e del codec, eseguibile dove `tokenizers`
e' installato (es. Colab). Verifica le proprieta' che ci servono:

  1. round-trip lossless: decode(encode(x)) == x  (il byte-level non perde nulla);
  2. tutti gli id stanno nel vocabolario e entrano in uint16;
  3. il token <|endoftext|> esiste e ha un id valido.

Non scarica TinyStories: allena un BPE minuscolo su testo sintetico, cosi' gira
in un secondo. Se `tokenizers` manca, il test si salta (non fallisce).

    python tests/test_tokenizer_roundtrip.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError:
        print("SKIP: `tokenizers` non installato (pip install tokenizers).")
        return

    from burel.data.tinystories import EOS

    corpus = [
        "Once upon a time there was a little cat named Tom.",
        "The cat liked to play in the garden every morning.",
        "One day, Tom found a red ball and was very happy.",
        "He played with the ball until the sun went down. The end.",
    ] * 50

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
    assert vocab <= 65535, f"vocab {vocab} non entra in uint16"

    # round-trip lossless su testo dentro e fuori dominio (byte-level => sempre ok)
    for s in ["Once upon a time, the cat ran!", "Zxq-42 \n\t £€ unseen chars 9876"]:
        ids = tok.encode(s).ids
        assert all(0 <= i < vocab for i in ids), "id fuori range"
        back = tok.decode(ids)
        assert back == s, f"round-trip rotto:\n  in : {s!r}\n  out: {back!r}"

    print(f"OK: vocab={vocab}, eos_id={eos_id}, round-trip lossless, id in uint16.")


if __name__ == "__main__":
    main()
