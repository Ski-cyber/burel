#!/usr/bin/env python
# tests/test_causal.py
#
# Verifica che BurelLM sia STRETTAMENTE causale: cambiando i token a posizione > t,
# i logit alle posizioni <= t non devono cambiare (di un bit). E' la prova che il
# retrieval della memoria non introduce lookahead intra-chunk.
#
#   python tests/test_causal.py     (oppure: pytest tests/)

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from burel.model import BurelLM


def test_strictly_causal():
    torch.manual_seed(0)
    model = BurelLM(
        vocab_size=17, d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64,
        chunk_size=4, persistent_length=2, max_memory_length=32,
        num_mem_levels=2, memory_depth=2,
    )
    model.eval()  # dropout off -> deterministico

    vocab, T, t = 17, 12, 1  # t=1 dentro il primo chunk (chunk_size=4)
    x = torch.randint(0, vocab, (2, T))

    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[:, t + 1:] = (x2[:, t + 1:] + 5) % vocab  # stravolge tutto il futuro
        alt, _ = model(x2)

    past_delta = (base[:, :t + 1] - alt[:, :t + 1]).abs().max().item()
    changed_delta = (base[:, t + 1] - alt[:, t + 1]).abs().max().item()

    print(f"max|delta logit| su posizioni 0..{t} = {past_delta:.2e}  (deve essere ~0)")
    print(f"max|delta logit| a posizione {t + 1}  = {changed_delta:.2e}  (deve essere > 0)")

    assert past_delta < 1e-5, "LEAK: un token futuro influenza una posizione passata"
    assert changed_delta > 1e-4, "modello inerte: non reagisce ai token modificati"


if __name__ == "__main__":
    test_strictly_causal()
    print("OK: causalita' stretta verificata")
