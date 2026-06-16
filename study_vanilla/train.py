#!/usr/bin/env python
# study_vanilla/train.py
#
# nanoGPT-style trainer for the vanilla baseline. Reads the SAME data_cache as Burel,
# saves vanilla_best.pt / vanilla_last.pt (with Drive backup + auto-resume on Colab).
# Because Burel and this baseline share vocabulary, tokenizer and data, their validation
# losses are DIRECTLY comparable — that single number is the headline of the A/B.
#
#   python study_vanilla/train.py
#   python study_vanilla/train.py --config study_vanilla/config.yaml

import argparse
import math
import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import yaml

# Make the repo root importable so `study_vanilla` resolves regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from study_vanilla.data import get_batch, load_split
from study_vanilla.model import VanillaGPT, count_parameters

ROOT = pathlib.Path(__file__).resolve().parents[1]


def lr_at(it, tc):
    warmup, max_it = tc["warmup_iters"], tc["max_iters"]
    lr, min_lr = tc["learning_rate"], tc["min_lr"]
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_it:
        return min_lr
    ratio = (it - warmup) / (max_it - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def drive_dir_if_mounted(drive_dir):
    if not drive_dir:
        return None
    parent = os.path.dirname(drive_dir.rstrip("/")) or "/"
    return drive_dir if os.path.isdir(parent) else None


def save_checkpoint(state, out_dir, drive_dir, fname):
    torch.save(state, os.path.join(out_dir, fname))
    if drive_dir:
        try:
            os.makedirs(drive_dir, exist_ok=True)
            torch.save(state, os.path.join(drive_dir, fname))
        except Exception as e:
            print(f"  -> backup Drive fallito: {e}")


def find_resume_path(resume, out_dir, drive_dir):
    if resume in (None, "none", "", False):
        return None
    if resume != "auto":
        return resume if os.path.exists(resume) else None
    for base in (out_dir, drive_dir):
        if base:
            cand = os.path.join(base, "vanilla_last.pt")
            if os.path.exists(cand):
                return cand
    return None


def configure_optimizer(model, weight_decay, lr):
    # Weight-decay on 2D weights (matmuls/embeddings); none on biases/LayerNorms.
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, splits, tc, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tc["eval_iters"])
        for k in range(tc["eval_iters"]):
            x, y = get_batch(data, tc["block_size"], tc["batch_size"], device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "study_vanilla" / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc, tc, dc = cfg["model"], cfg["train"], cfg["data"]
    torch.manual_seed(tc["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device={device}")

    cache_dir = dc["cache_dir"] if os.path.isabs(dc["cache_dir"]) else ROOT / dc["cache_dir"]
    with open(os.path.join(cache_dir, "meta.pkl"), "rb") as f:
        vocab_size = pickle.load(f)["vocab_size"]
    splits = {"train": load_split(cache_dir, "train"), "val": load_split(cache_dir, "val")}
    print(f"vocab_size={vocab_size}, cache={cache_dir}")

    assert tc["block_size"] <= mc["context"], "block_size deve essere <= context"

    model = VanillaGPT(
        vocab_size=vocab_size,
        d_model=mc["d_model"], n_head=mc["n_head"], n_layer=mc["n_layer"],
        d_ff=mc["d_ff"], context=mc["context"], dropout=mc["dropout"],
        activation=mc["activation"], tie_weights=mc["tie_weights"],
    ).to(device)
    print(f"VanillaGPT parametri addestrabili: {count_parameters(model):,}")

    optimizer = configure_optimizer(model, tc["weight_decay"], tc["learning_rate"])

    dtype = tc.get("dtype", "fp32")
    use_amp = device == "cuda" and dtype in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == "fp16"))

    out_dir = str(ROOT / tc["out_dir"]) if not os.path.isabs(tc["out_dir"]) else tc["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    drive_dir = drive_dir_if_mounted(tc.get("drive_backup_dir"))
    if drive_dir:
        print(f"Drive montato: backup in {drive_dir}")

    # --- RESUME (Colab-safe) ---
    start_iter, best_val, resumed = 0, float("inf"), False
    resume_path = find_resume_path(tc.get("resume", "auto"), out_dir, drive_dir)
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        if ck.get("vocab_size") == vocab_size and ck.get("config", {}).get("model") == mc:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_iter = ck.get("iter", -1) + 1
            best_val = ck.get("best_val", float("inf"))
            resumed = True
            print(f"RESUME da {resume_path}: iter {start_iter}, best_val {best_val:.4f}")
        else:
            print(f"ATTENZIONE: {resume_path} architettura/vocab diversi -> ignorato.")
    if not resumed:
        print("Training da zero.")

    def make_state(it, val_loss):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg, "vocab_size": vocab_size,
                "iter": it, "val_loss": val_loss, "best_val": best_val}

    patience = tc.get("patience", 0) or 0
    no_improve = 0
    model.train()

    try:
        from tqdm import tqdm
        rng = tqdm(range(start_iter, tc["max_iters"] + 1), initial=start_iter,
                   total=tc["max_iters"], desc="vanilla", dynamic_ncols=True)
        write = rng.write
    except ImportError:
        rng = range(start_iter, tc["max_iters"] + 1)
        write = print

    for it in rng:
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, tc)

        if it % tc["eval_interval"] == 0:
            losses = estimate_loss(model, splits, tc, device)
            msg = f"iter {it}: train {losses['train']:.4f} | val {losses['val']:.4f}"
            if losses["val"] < best_val:
                best_val = losses["val"]
                no_improve = 0
                save_checkpoint(make_state(it, losses["val"]), out_dir, drive_dir, "vanilla_best.pt")
                msg += "  -> nuovo best"
            else:
                no_improve += 1
            save_checkpoint(make_state(it, losses["val"]), out_dir, drive_dir, "vanilla_last.pt")
            write(msg)
            if patience and no_improve >= patience:
                write(f"early stop: nessun miglioramento da {patience} eval (best {best_val:.4f}).")
                break

        if it == tc["max_iters"]:
            break

        x, y = get_batch(splits["train"], tc["block_size"], tc["batch_size"], device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()
        if hasattr(rng, "set_postfix"):
            rng.set_postfix(loss=f"{loss.item():.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    print(f"fine. best val loss = {best_val:.4f}  (Burel best val = 2.4831, stesso vocab/dati)")


if __name__ == "__main__":
    main()
