# Modello — pesi di Burel

I pesi addestrati sono troppo grandi per stare nell'albero Git, quindi sono
distribuiti come **asset di una GitHub Release** (nessun limite di banda, download
diretto).

## Download

➡️ Vai alla [**pagina Releases**](../../../releases) e scarica:

| file | dimensione | descrizione |
|------|-----------|-------------|
| `modello_burel_best_v2.pt` | ~231 MB | **modello consigliato** — 20M param, TinyStories + BPE 16k, val loss best 2.48 |
| `modello_burel_best_v1.pt` | ~54 MB | versione precedente — 14M param, TinyShakespeare char-level, val loss best 1.54 |

## Uso

```python
from burel.inference import load_model, generate_text

# punta al .pt scaricato dalle Release
model, meta = load_model("modello_burel_best_v2.pt")
print(generate_text(model, meta, prompt="Once upon a time", max_new_tokens=500))
```

Oppure da riga di comando:

```bash
python scripts/generate.py --prompt "Once upon a time" --tokens 500 --temperature 0.8
```

> Nota: il checkpoint contiene `model` + `optimizer` + `iter` + `meta` (config dati e
> tokenizer), così l'inferenza sa già com'è fatto il modello e come decodificare i token.
