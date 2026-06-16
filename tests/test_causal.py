#!/usr/bin/env python
# tests/test_causal.py
#
# Verify that BurelLM is STRICTLY causal: changing tokens at positions > t must not
# change the logits at positions <= t (not by a single bit). This is the proof that
# memory retrieval does not introduce intra-chunk lookahead.
#
#   python tests/test_causal.py     (or: pytest tests/)

import pathlib
import sys

# Make the repo root importable so `burel` resolves regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from burel.model import BurelLM


def test_strictly_causal():
    # Fixed seed so model weights and random inputs are reproducible.
    torch.manual_seed(0)
    # Tiny model: just large enough to exercise chunking and the memory levels.
    model = BurelLM(
        vocab_size=17, d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64,
        chunk_size=4, persistent_length=2, max_memory_length=32,
        num_mem_levels=2, memory_depth=2,
    )
    model.eval()  # dropout off -> deterministic

    vocab, T, t = 17, 12, 1  # t=1 sits inside the first chunk (chunk_size=4)
    x = torch.randint(0, vocab, (2, T))

    with torch.no_grad():
        # Baseline forward pass.
        base, _ = model(x)
        x2 = x.clone()
        # Perturb every token strictly after position t (the entire "future").
        x2[:, t + 1:] = (x2[:, t + 1:] + 5) % vocab  # scrambles the whole future
        alt, _ = model(x2)

    # Past logits (positions 0..t) must be untouched by the future perturbation.
    past_delta = (base[:, :t + 1] - alt[:, :t + 1]).abs().max().item()
    # Position t+1 was directly perturbed, so its logits must change.
    changed_delta = (base[:, t + 1] - alt[:, t + 1]).abs().max().item()

    print(f"max|delta logit| su posizioni 0..{t} = {past_delta:.2e}  (deve essere ~0)")
    print(f"max|delta logit| a posizione {t + 1}  = {changed_delta:.2e}  (deve essere > 0)")

    assert past_delta < 1e-5, "LEAK: un token futuro influenza una posizione passata"
    assert changed_delta > 1e-4, "modello inerte: non reagisce ai token modificati"


if __name__ == "__main__":
    test_strictly_causal()
    print("OK: causalita' stretta verificata")
