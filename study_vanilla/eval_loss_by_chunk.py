#!/usr/bin/env python
# study_vanilla/eval_loss_by_chunk.py
#
# Standalone evaluation of the vanilla baseline alone (no `burel` import): loss-per-chunk
# on the val split and, optionally, on an unseen-domain text file. Lets the study be read
# and run on its own. For the full three-way A/B vs Burel use scripts/compare_three.py.
#
#   python study_vanilla/eval_loss_by_chunk.py --domain_file domain_code.txt

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from study_vanilla.data import encode_text, get_batch, load_meta, load_split
from study_vanilla.model import build_from_config

ROOT = pathlib.Path(__file__).resolve().parents[1]


def per_chunk_loss(model, x, y, chunk_size):
    logits, _ = model(x)
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").reshape(B, T)
    per_pos = ce.mean(dim=0)
    nc = T // chunk_size
    return per_pos[: nc * chunk_size].reshape(nc, chunk_size).mean(dim=1).cpu()


@torch.no_grad()
def eval_stream(model, data, name, block, chunk, batch, windows, device, seed):
    nc = block // chunk
    acc = torch.zeros(nc)
    gen = torch.Generator().manual_seed(seed)
    nb = max(1, windows // batch)
    for _ in range(nb):
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix]).to(device)
        acc += per_chunk_loss(model, x, y, chunk)
    curve = acc / nb
    print(f"\n== {name} == (loss media {curve.mean().item():.4f})")
    for c in range(nc):
        print(f"  chunk {c:>2}: {curve[c].item():.4f}")
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints_vanilla" / "vanilla_best.pt"))
    ap.add_argument("--cache_dir", default=str(ROOT / "data_cache"))
    ap.add_argument("--domain_file", default=None)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16, help="dimensione bucket (= chunk_size di Burel)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_from_config(ck["config"]["model"], ck["vocab_size"], device)
    model.load_state_dict(ck["model"]); model.eval()
    block = ck["config"]["model"]["context"]
    print(f"device={device}  vanilla best_val={ck.get('best_val'):.4f}  block={block}")

    eval_stream(model, load_split(args.cache_dir, "val"), "CONTROL (val)",
                block, args.chunk, args.batch, args.windows, device, args.seed)
    if args.domain_file:
        meta = load_meta(args.cache_dir)
        ids = np.array(encode_text(meta, pathlib.Path(args.domain_file).read_text(errors="ignore"),
                                   args.cache_dir), dtype=np.uint16)
        eval_stream(model, ids, f"UNSEEN ({args.domain_file})",
                    block, args.chunk, args.batch, args.windows, device, args.seed)


if __name__ == "__main__":
    main()
