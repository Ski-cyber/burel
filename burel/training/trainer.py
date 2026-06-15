# burel/training/trainer.py
#
# Training loop nanoGPT-style per BurelLM (HOPE / nested learning a pieno giro).
# Cross-entropy next-token. Logica invariata; percorsi centralizzati in burel.paths.
#
# Checkpoint (in <out_dir> e, se Drive montato, anche in drive_backup_dir):
#   burel_best.pt  -> miglior val loss
#   burel_last.pt  -> ultimo stato (model + optimizer + iter) per il RESUME
#
# Note Test-Time Training:
#   - Flash/mem-efficient attention disabilitate: i grad di 2 ordine
#     (create_graph=True) falliscono con quei kernel.
#   - dtype fp32 di default per stabilita'.

import math
import os
import pickle

import numpy as np
import torch
import yaml
from tqdm import tqdm

from burel.data import load_split, prepare
from burel.model import BurelLM, count_parameters
from burel.paths import CACHE_DIR, DEFAULT_CONFIG, resolve


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, cfg):
    warmup, max_it = cfg["warmup_iters"], cfg["max_iters"]
    lr, min_lr = cfg["learning_rate"], cfg["min_lr"]
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_it:
        return min_lr
    ratio = (it - warmup) / (max_it - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def drive_dir_if_mounted(drive_dir):
    """Ritorna drive_dir se il mount Drive (la sua cartella padre) esiste, altrimenti None."""
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
            return True
        except Exception as e:
            print(f"  -> backup Drive fallito: {e}")
    return False


def find_resume_path(resume, out_dir, drive_dir):
    if resume in (None, "none", "", False):
        return None
    if resume != "auto":
        return resume if os.path.exists(resume) else None
    # auto: cerca burel_last.pt prima in locale, poi su Drive
    for base in (out_dir, drive_dir):
        if base:
            cand = os.path.join(base, "burel_last.pt")
            if os.path.exists(cand):
                return cand
    return None


@torch.no_grad()
def estimate_loss(model, splits, cfg_t, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg_t["eval_iters"])
        for k in range(cfg_t["eval_iters"]):
            x, y = get_batch(data, cfg_t["block_size"], cfg_t["batch_size"], device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main(config_path=DEFAULT_CONFIG):
    cfg = load_config(config_path)
    mc, tc = cfg["model"], cfg["train"]
    torch.manual_seed(tc["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device}")

    # Idempotente: rigenera la cache solo se la config 'data' e' cambiata.
    # Cosi' cambiare dataset (shakespeare <-> tinystories) basta in config.
    prepare(cfg)
    with open(CACHE_DIR / "meta.pkl", "rb") as f:
        vocab_size = pickle.load(f)["vocab_size"]
    splits = {"train": load_split("train"), "val": load_split("val")}

    assert tc["block_size"] % mc["chunk_size"] == 0, "block_size deve essere multiplo di chunk_size"
    assert tc["block_size"] <= mc["max_memory_length"], "block_size deve essere <= max_memory_length"

    model = BurelLM(
        vocab_size=vocab_size,
        d_model=mc["d_model"], nhead=mc["nhead"], num_encoder_layers=mc["num_encoder_layers"],
        dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"],
        persistent_length=mc["persistent_length"], max_memory_length=mc["max_memory_length"],
        chunk_size=mc["chunk_size"], mem_lr=mc["mem_lr"], memory_depth=mc["memory_depth"],
        use_silu=mc["use_silu"], num_mem_levels=mc["num_mem_levels"], tie_weights=mc["tie_weights"],
    ).to(device)
    print(f"parametri addestrabili: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], betas=(0.9, 0.95))

    dtype = tc["dtype"]
    use_amp = device == "cuda" and dtype in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    # GradScaler solo per fp16 (fp32/bf16 -> disabilitato, no-op). Costruzione robusta alle
    # versioni di torch: API moderna torch.amp.GradScaler('cuda', ...) se disponibile,
    # altrimenti la classica torch.cuda.amp.GradScaler. Stesso oggetto, stessi default,
    # stessa matematica: nessuna differenza di comportamento nei training fp16.
    scaler_enabled = use_amp and dtype == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    out_dir = str(resolve(tc["out_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    drive_dir = drive_dir_if_mounted(tc.get("drive_backup_dir"))
    if drive_dir:
        print(f"Drive montato: backup in {drive_dir}")

    # --- RESUME ---
    start_iter = 0
    best_val = float("inf")
    resumed = False
    resume_path = find_resume_path(tc.get("resume", "auto"), out_dir, drive_dir)
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        # Compatibilita': stesso vocabolario E stessa architettura. Un checkpoint v1
        # (es. d_model diverso) NON viene caricato su un modello v2 -> niente crash.
        compatible = (ck.get("vocab_size") == vocab_size
                      and ck.get("config", {}).get("model") == mc)
        if compatible:
            model.load_state_dict(ck["model"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scaler" in ck and ck["scaler"] is not None:
                scaler.load_state_dict(ck["scaler"])
            start_iter = ck.get("iter", -1) + 1
            best_val = ck.get("best_val", ck.get("val_loss", float("inf")))
            resumed = True
            print(f"RESUME da {resume_path}: riparto da iter {start_iter}, best_val {best_val:.4f}")
        else:
            print(f"ATTENZIONE: {resume_path} ha architettura/vocabolario diversi dalla config "
                  f"attuale -> ignorato. Training da zero (i checkpoint v1 non si mischiano con v2).")
    if not resumed:
        print("Training da zero.")

    def make_state(it, val_loss):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "config": cfg, "vocab_size": vocab_size,
            "iter": it, "val_loss": val_loss, "best_val": best_val,
        }

    patience = tc.get("patience", 0) or 0  # 0 = early stop disattivo
    no_improve = 0  # eval consecutivi senza nuovo best

    model.train()
    pbar = tqdm(range(start_iter, tc["max_iters"] + 1), initial=start_iter,
                total=tc["max_iters"], desc="training", dynamic_ncols=True)
    for it in pbar:
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, tc)

        if it % tc["eval_interval"] == 0:
            losses = estimate_loss(model, splits, tc, device)
            msg = f"iter {it}: train {losses['train']:.4f} | val {losses['val']:.4f}"
            if losses["val"] < best_val:
                best_val = losses["val"]
                no_improve = 0
                save_checkpoint(make_state(it, losses["val"]), out_dir, drive_dir, "burel_best.pt")
                msg += "  -> nuovo best salvato"
            else:
                no_improve += 1
            # 'last' sempre, per il resume (perdi al massimo eval_interval iter su crash)
            save_checkpoint(make_state(it, losses["val"]), out_dir, drive_dir, "burel_last.pt")
            pbar.write(msg)  # stampa sopra la barra senza romperla

            if patience and no_improve >= patience:
                pbar.write(f"early stop: nessun miglioramento da {patience} eval "
                           f"(best val {best_val:.4f}). burel_best.pt ha il modello migliore.")
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

        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    pbar.close()
    print(f"fine. best val loss = {best_val:.4f}")
