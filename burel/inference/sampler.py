# burel/inference/sampler.py
#
# API di inferenza importabile:
#   from burel.inference import load_model, generate_text
#   model, meta = load_model("checkpoints/burel_best.pt")
#   print(generate_text(model, meta, prompt="ROMEO:", max_new_tokens=500))

import torch

from burel.data import decode, encode, load_meta
from burel.model import BurelLM


def _configure_backend():
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def build_model(mc, vocab_size, device):
    return BurelLM(
        vocab_size=vocab_size,
        d_model=mc["d_model"], nhead=mc["nhead"], num_encoder_layers=mc["num_encoder_layers"],
        dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"],
        persistent_length=mc["persistent_length"], max_memory_length=mc["max_memory_length"],
        chunk_size=mc["chunk_size"], mem_lr=mc["mem_lr"], memory_depth=mc["memory_depth"],
        use_silu=mc["use_silu"], num_mem_levels=mc["num_mem_levels"], tie_weights=mc["tie_weights"],
    ).to(device)


def load_model(ckpt_path, device=None):
    """Carica un checkpoint e ritorna (model in eval, meta vocab)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure_backend()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"], ckpt["vocab_size"], device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model._ckpt_info = {"iter": ckpt.get("iter"), "val_loss": ckpt.get("val_loss")}
    return model, load_meta()


@torch.no_grad()
def generate_text(model, meta, prompt="\n", max_new_tokens=500, temperature=0.8, top_k=200):
    """Genera testo a partire da un prompt. Ritorna la stringa completa (prompt incluso)."""
    device = next(model.parameters()).device
    ids = encode(meta, prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(meta, out[0].tolist())
