#!/usr/bin/env python
"""A/B PERNO — Burel-ON vs Burel-OFF vs Vanilla, a parita' di parametri e dati.

Domanda che decide il destino di Burel: l'adattamento della MEMORIA (Test-Time Training)
batte l'adattamento dell'ATTENZIONE di un Transformer vanilla, a parita' di parametri?

L'ablazione precedente (scripts/ablation_ttt.py) ha mostrato TTT-ON >> Burel-OFF, ma OFF
e' Burel azzoppato (attenzione chunk-locale, niente canale cross-chunk). Il concorrente
vero e' il vanilla con attenzione PIENA. Qui le tre curve girano sulle STESSE finestre,
cosi' il confronto e' appaiato e una variabile sola (l'architettura).

Lettura
-------
  * confronto headline: best val (stesso vocab/dati -> direttamente comparabile).
  * loss-per-chunk su CONTROL (visto) e UNSEEN (codice): se Burel-ON sta SOTTO il vanilla,
    la memoria si ripaga; se il vanilla eguaglia/batte Burel-ON, l'amputazione non paga
    e la mossa onesta e' scalare il vanilla.

Uso
---
  python scripts/compare_three.py --vanilla_ckpt PATH/vanilla_best.pt \
      --burel_ckpt PATH/burel_best.pt --domain_file domain_code.txt
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from burel.data import encode, load_meta, load_split
from burel.inference import load_model as load_burel
from burel.paths import resolve
from study_vanilla.model import build_from_config


def sample_windows(token_array, block_size, batch_size, device, generator):
    n = len(token_array)
    assert n > block_size + 1, f"testo troppo corto: {n} token"
    ix = torch.randint(n - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(token_array[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(token_array[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def per_chunk_loss(model, x, y, chunk_size):
    """Loss media per posizione, aggregata per chunk -> tensore [num_chunks] su CPU.
    Funziona per Burel (chunked) e per il vanilla (bucket di chunk_size posizioni)."""
    logits, _ = model(x)
    logits = logits.detach()
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").reshape(B, T)
    per_pos = ce.mean(dim=0)
    nc = T // chunk_size
    return per_pos[: nc * chunk_size].reshape(nc, chunk_size).mean(dim=1).cpu()


def load_vanilla(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_from_config(ck["config"]["model"], ck["vocab_size"], device)
    model.load_state_dict(ck["model"])
    model.eval()
    model._best_val = ck.get("best_val", ck.get("val_loss"))
    return model


def evaluate_stream(burel, vanilla, token_array, name, block, chunk, batch, windows, device, seed):
    nc = block // chunk
    acc = {"burel_on": torch.zeros(nc), "burel_off": torch.zeros(nc), "vanilla": torch.zeros(nc)}
    gen = torch.Generator().manual_seed(seed)
    n_batches = max(1, windows // batch)
    print(f"\n[{name}] {n_batches} batch x {batch} finestre da {block} token ({nc} chunk) ...")
    for _ in range(n_batches):
        x, y = sample_windows(token_array, block, batch, device, gen)  # STESSE finestre per i tre
        burel.update_memory_in_inference = True
        acc["burel_on"] += per_chunk_loss(burel, x, y, chunk)
        burel.update_memory_in_inference = False
        acc["burel_off"] += per_chunk_loss(burel, x, y, chunk)
        acc["vanilla"] += per_chunk_loss(vanilla, x, y, chunk)
    return {k: v / n_batches for k, v in acc.items()}


def report(name, curves):
    on, off, van = curves["burel_on"], curves["burel_off"], curves["vanilla"]
    print(f"\n================  {name}  ================")
    print(f"{'chunk':>5} | {'Burel-ON':>9} | {'Burel-OFF':>9} | {'Vanilla':>9} | {'ON-Van':>8}")
    print("-" * 56)
    for c in range(len(on)):
        print(f"{c:>5} | {on[c].item():>9.4f} | {off[c].item():>9.4f} | "
              f"{van[c].item():>9.4f} | {on[c].item() - van[c].item():>+8.4f}")
    print("-" * 56)
    m_on, m_off, m_van = on.mean().item(), off.mean().item(), van.mean().item()
    print(f"loss media   Burel-ON={m_on:.4f}   Burel-OFF={m_off:.4f}   Vanilla={m_van:.4f}")
    margin = m_van - m_on  # >0 => Burel-ON meglio del vanilla
    print(f"margine (Vanilla - Burel-ON) = {margin:+.4f}   (>0 => la memoria batte l'attenzione)")
    slope = lambda t: (t[0] - t[-1]).item()
    print(f"adattamento in-contesto (chunk0-chunkN):  "
          f"ON={slope(on):+.4f}  Vanilla={slope(van):+.4f}")
    return {"margin": margin, "m_on": m_on, "m_van": m_van}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--burel_ckpt", default=str(resolve("checkpoints/burel_best.pt")))
    ap.add_argument("--vanilla_ckpt", default=str(resolve("checkpoints_vanilla/vanilla_best.pt")))
    ap.add_argument("--domain_file", default=None, help="dominio MAI VISTO (es. codice)")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--block", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    burel, meta = load_burel(args.burel_ckpt, device=device)
    vanilla = load_vanilla(args.vanilla_ckpt, device=device)
    print(f"Burel   best_val={burel._ckpt_info['val_loss']:.4f}")
    print(f"Vanilla best_val={vanilla._best_val:.4f}   "
          f"(stesso vocab/dati -> CONFRONTO HEADLINE diretto)")

    chunk = burel.chunk_size
    block = args.block or burel.max_memory_length
    assert block <= vanilla.context, "block > context del vanilla"

    results = {}
    print("\n# CONTROL = val in-distribution")
    results["control"] = report("CONTROL (in-distribution)",
                                 evaluate_stream(burel, vanilla, load_split("val"),
                                                 "CONTROL", block, chunk, args.batch,
                                                 args.windows, device, args.seed))
    if args.domain_file:
        text = pathlib.Path(args.domain_file).read_text(encoding="utf-8", errors="ignore")
        ids = np.array(encode(meta, text), dtype=np.uint16)
        print(f"\n# UNSEEN = {args.domain_file} ({len(ids)} token)")
        results["unseen"] = report("UNSEEN (dominio nuovo)",
                                   evaluate_stream(burel, vanilla, ids, "UNSEEN", block, chunk,
                                                   args.batch, args.windows, device, args.seed))

    print("\n\n############  VERDETTO A/B  ############")
    hb, hv = burel._ckpt_info["val_loss"], vanilla._best_val
    print(f"HEADLINE best val:  Burel={hb:.4f}  Vanilla={hv:.4f}  "
          f"-> {'Burel' if hb < hv else 'Vanilla'} vince a parita' di parametri")
    for k in ("control", "unseen"):
        if k in results:
            print(f"{k.upper():8}: margine Vanilla-BurelON = {results[k]['margin']:+.4f}")
    win_unseen = results.get("unseen", {}).get("margin", None)
    print("-" * 38)
    if hb < hv and (win_unseen is None or win_unseen > 0):
        print("=> La memoria nested SI RIPAGA: Burel batte il vanilla a parita' di parametri.")
        print("   Ha senso scalare Burel. Prossimo: misurare anche il costo (token/sec, FLOP).")
    else:
        print("=> Il vanilla EGUAGLIA/BATTE Burel a parita' di parametri. L'amputazione")
        print("   dell'attenzione cross-chunk non paga: la mossa onesta e' scalare il vanilla")
        print("   (+ MoE/retrieval), non Burel. Il TTT resta un ramo di ricerca.")
    print("######################################")


if __name__ == "__main__":
    main()
