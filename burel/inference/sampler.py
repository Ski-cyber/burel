# burel/inference/sampler.py
#
# Importable inference API:
#   from burel.inference import load_model, generate_text
#   model, meta = load_model("checkpoints/burel_best.pt")
#   print(generate_text(model, meta, prompt="ROMEO:", max_new_tokens=500))

import torch

from burel.data import decode, encode, load_meta
from burel.model import BurelLM


# Force the math SDP attention kernel on CUDA. Test-time training relies on
# second-order gradients, which the flash/mem-efficient kernels do not support;
# this keeps inference consistent with how the model was trained.
def _configure_backend():
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


# Instantiate a BurelLM from a model-config dict (mc) and move it to device.
# Mirrors the construction in the trainer so checkpoints load cleanly.
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
    """Load a checkpoint and return (model in eval mode, vocab meta)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure_backend()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Rebuild the architecture from the config stored in the checkpoint, then load weights.
    model = build_model(ckpt["config"]["model"], ckpt["vocab_size"], device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # Stash provenance (training iteration and val loss) for callers to inspect/print.
    model._ckpt_info = {"iter": ckpt.get("iter"), "val_loss": ckpt.get("val_loss")}
    return model, load_meta()


@torch.no_grad()
def generate_text(model, meta, prompt="\n", max_new_tokens=500, temperature=0.8, top_k=200):
    """Generate text from a prompt. Returns the full string (prompt included)."""
    device = next(model.parameters()).device
    # Encode the prompt to token ids; fall back to [0] if it encodes to nothing.
    ids = encode(meta, prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(meta, out[0].tolist())
