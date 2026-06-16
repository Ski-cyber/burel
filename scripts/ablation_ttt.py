#!/usr/bin/env python
"""Ablazione del Test-Time Training (TTT) — UNA variabile sola: memoria che si adatta ON vs OFF.

Domanda che risponde: l'apprendimento a test-time (l'aggiornamento dei fast-weight
mentre il modello legge) aggiunge valore? E ne aggiunge DI PIU' su un dominio MAI VISTO?

Come isola la variabile
-----------------------
In BurelLM, tra un chunk e l'altro l'UNICO canale di informazione e' la memoria CMS
(l'attenzione e' chunk-locale). Quindi:
  - TTT ON  : i fast-weight della memoria si adattano ad ogni chunk (Test-Time Training).
  - TTT OFF : la memoria resta sui slow-weight (read intatto, ma niente adattamento).
Tutto il resto e' identico, stesse finestre, stesso modello, stessi token. La differenza
e' SOLO l'apprendimento a test-time. Lo switch e' gia' nel modello: update_memory_in_inference.

ATTENZIONE — cosa misura davvero: la memoria si azzera ad ogni forward (reset_state),
quindi questo misura l'adattamento DENTRO una finestra (<= max_memory_length token):
la "Tesi A" (adattamento in-contesto), che e' quella IMPLEMENTATA. NON misura
l'accumulo cross-documento ("Tesi B"), che non esiste ancora nel codice.

Lettura del risultato (per un quant)
------------------------------------
  * benefit = loss_OFF - loss_ON  (>0 => il TTT si paga da solo)
  * confronto UNSEEN vs CONTROL: la tesi "master of expertise on the fly" vuole
    benefit_UNSEEN > benefit_CONTROL (il TTT aiuta DI PIU' dove il dominio e' nuovo).
  * curva loss-per-chunk: se ON cala piu' ripido di OFF scendendo nella finestra,
    quella e' la prova DIRETTA che il modello si specializza mentre legge.

Uso
---
  python scripts/ablation_ttt.py --domain_file PATH_AL_TESTO_DOMINIO_NUOVO.txt
Opzioni: --ckpt --windows --batch --seed --block
"""

import argparse
import pathlib
import sys

# Rende importabile la repo root, qualunque sia la cwd (come in scripts/generate.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from burel.data import encode, load_split
from burel.inference import load_model
from burel.paths import CACHE_DIR, resolve


def sample_windows(token_array, block_size, batch_size, device, generator):
    """Estrae batch_size finestre contigue di block_size token. x e y (target shiftato di 1).
    Usa un generator dedicato cosi' le stesse finestre sono riproducibili e CONDIVISE da ON e OFF."""
    n = len(token_array)
    assert n > block_size + 1, (
        f"testo troppo corto: {n} token, servono > {block_size + 1}. Usa un file piu' grande.")
    ix = torch.randint(n - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(token_array[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(token_array[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def per_chunk_loss(model, x, y, chunk_size):
    """Loss media per posizione, aggregata per chunk. Ritorna un tensore [num_chunks] su CPU.

    Nota: il forward con TTT ON costruisce un grafo del 2o ordine (create_graph=True nel CMS).
    Stacchiamo i logits subito cosi' il grafo si libera e non calcoliamo gradienti inutili."""
    logits, _ = model(x)                      # [B, T, V], targets=None -> niente loss interna
    logits = logits.detach()
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").reshape(B, T)
    per_pos = ce.mean(dim=0)                   # [T] loss media sul batch, per posizione
    num_chunks = T // chunk_size
    per_pos = per_pos[: num_chunks * chunk_size].reshape(num_chunks, chunk_size).mean(dim=1)
    return per_pos.cpu()                       # [num_chunks]


def evaluate_stream(model, token_array, name, block_size, chunk_size, batch_size, windows, device, seed):
    """Valuta uno stream con TTT ON e OFF sulle STESSE finestre. Ritorna (curva_on, curva_off)."""
    num_chunks = block_size // chunk_size
    acc_on = torch.zeros(num_chunks)
    acc_off = torch.zeros(num_chunks)
    # Generator dedicato per-stream: ON e OFF vedono la STESSA sequenza di finestre (confronto appaiato).
    gen = torch.Generator().manual_seed(seed)
    n_batches = max(1, windows // batch_size)
    print(f"\n[{name}] {n_batches} batch x {batch_size} finestre da {block_size} token "
          f"({num_chunks} chunk) ...")
    for _ in range(n_batches):
        x, y = sample_windows(token_array, block_size, batch_size, device, gen)
        model.update_memory_in_inference = True
        acc_on += per_chunk_loss(model, x, y, chunk_size)
        model.update_memory_in_inference = False
        acc_off += per_chunk_loss(model, x, y, chunk_size)
    return acc_on / n_batches, acc_off / n_batches


def report(name, curve_on, curve_off):
    """Stampa la tabella per-chunk e le metriche-verdetto per uno stream."""
    mean_on, mean_off = curve_on.mean().item(), curve_off.mean().item()
    benefit = mean_off - mean_on                       # >0 => TTT aiuta
    # Pendenza di adattamento: quanto cala la loss dal primo chunk all'ultimo.
    slope_on = (curve_on[0] - curve_on[-1]).item()
    slope_off = (curve_off[0] - curve_off[-1]).item()

    print(f"\n================  {name}  ================")
    print(f"{'chunk':>5} | {'loss ON':>9} | {'loss OFF':>9} | {'OFF-ON':>8}")
    print("-" * 40)
    for c in range(len(curve_on)):
        d = curve_off[c].item() - curve_on[c].item()
        print(f"{c:>5} | {curve_on[c].item():>9.4f} | {curve_off[c].item():>9.4f} | {d:>+8.4f}")
    print("-" * 40)
    print(f"loss media     ON={mean_on:.4f}   OFF={mean_off:.4f}")
    print(f"benefit (OFF-ON) = {benefit:+.4f}   (>0 => il TTT si paga da solo)")
    print(f"pendenza adattamento (chunk0 - chunkN):  ON={slope_on:+.4f}   OFF={slope_off:+.4f}")
    print(f"  extra-adattamento del TTT = {slope_on - slope_off:+.4f}  "
          f"(>0 => ON migliora piu' di OFF mentre legge)")
    return {"benefit": benefit, "slope_on": slope_on, "slope_off": slope_off,
            "extra_adapt": slope_on - slope_off}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(resolve("checkpoints/burel_best.pt")),
                    help="checkpoint da valutare (default: checkpoints/burel_best.pt)")
    ap.add_argument("--domain_file", default=None,
                    help="file di testo del DOMINIO MAI VISTO (es. codice). Se assente, solo il control.")
    ap.add_argument("--windows", type=int, default=128, help="finestre totali per stream")
    ap.add_argument("--batch", type=int, default=8, help="finestre per batch (memoria: ON usa 2o ordine)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--block", type=int, default=None,
                    help="lunghezza finestra in token (default: max_memory_length del modello)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model, meta = load_model(args.ckpt, device=device)
    info = model._ckpt_info
    print(f"checkpoint: iter={info['iter']} val_loss={info['val_loss']:.4f}")

    chunk_size = model.chunk_size
    block_size = args.block or model.max_memory_length
    assert block_size % chunk_size == 0, "block deve essere multiplo di chunk_size"
    assert block_size <= model.max_memory_length, (
        f"block ({block_size}) > max_memory_length ({model.max_memory_length}): "
        "le pos-embedding dei chunk extra non sono mai state allenate.")

    results = {}

    # CONTROL: dominio IN-distribution (la val di training). Qui ci aspettiamo che il TTT aiuti POCO.
    print("\n# CONTROL = val in-distribution (data_cache/val.bin)")
    val = load_split("val")
    on, off = evaluate_stream(model, val, "CONTROL (in-distribution)", block_size, chunk_size,
                              args.batch, args.windows, device, args.seed)
    results["control"] = report("CONTROL (in-distribution)", on, off)

    # UNSEEN: dominio nuovo fornito dall'utente. Qui la tesi predice il beneficio PIU' grande.
    if args.domain_file:
        text = pathlib.Path(args.domain_file).read_text(encoding="utf-8", errors="ignore")
        ids = np.array(encode(meta, text), dtype=np.uint16)
        print(f"\n# UNSEEN = {args.domain_file}  ({len(text)} char -> {len(ids)} token col tokenizer del modello)")
        on, off = evaluate_stream(model, ids, "UNSEEN (dominio nuovo)", block_size, chunk_size,
                                  args.batch, args.windows, device, args.seed)
        results["unseen"] = report("UNSEEN (dominio nuovo)", on, off)

    # ---- VERDETTO ----
    print("\n\n############  VERDETTO  ############")
    c = results["control"]
    print(f"CONTROL: benefit={c['benefit']:+.4f}, extra-adattamento={c['extra_adapt']:+.4f}")
    if "unseen" in results:
        u = results["unseen"]
        print(f"UNSEEN : benefit={u['benefit']:+.4f}, extra-adattamento={u['extra_adapt']:+.4f}")
        print("-" * 36)
        thesis = u["benefit"] > 0 and u["benefit"] > c["benefit"] and u["extra_adapt"] > 0
        if thesis:
            print("=> La tesi RESPIRA: il TTT aiuta, e aiuta DI PIU' sul dominio mai visto,")
            print("   con adattamento visibile mentre legge. Ha senso il passo successivo (vanilla/LoRA).")
        elif u["benefit"] <= 0:
            print("=> Tesi FALSIFICATA a costo ~0: con TTT ON la loss NON migliora (anzi). L'adattamento")
            print("   a test-time qui e' peso morto: l'attenzione in-contesto fa gia' tutto.")
        else:
            print("=> Tesi DEBOLE: il TTT da' un beneficio, ma NON maggiore sul dominio nuovo (o senza")
            print("   adattamento visibile). Non e' il 'master of expertise on the fly' sperato.")
    else:
        print("(solo CONTROL: passa --domain_file per testare la tesi sul dominio nuovo)")
    print("###################################")


if __name__ == "__main__":
    main()
