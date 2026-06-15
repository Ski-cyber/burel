# Burel

**Un seme per un'AI europea, realmente aperta.**

Burel è un language model costruito da zero su un'architettura sperimentale —
**HOPE** (Titans "Memory as Context" + **Nested Learning**) — con un obiettivo
dichiarato e ambizioso: dimostrare che si può fare ricerca seria su modelli di
linguaggio **fuori dai grandi laboratori**, in modo trasparente, riproducibile e
condiviso.

Questo repository non è un prodotto finito. È un **punto di partenza**: codice,
modello e diario degli esperimenti messi a disposizione di chiunque voglia
contribuire.

---

## Dove siamo e dove vogliamo andare

**Dove siamo.** Burel oggi è un modello da ~20M di parametri, **pretrainato da zero**.
Ha imparato a generare storie brevi in inglese coerenti (dataset TinyStories, tokenizer
BPE), con archi narrativi completi — inizio, svolgimento, "The end." — e personaggi
nominati. È un neonato competente: piccolo, ma vivo e che impara davvero. Il cuore
nested-learning gira **a pieno giro**, non in versione semplificata.

**Dove vogliamo andare.** L'ambizione è costruire, nel tempo e in modo collaborativo,
**un'AI di livello europeo: aperta, ispezionabile, non proprietaria.** Non l'ennesimo
clone richiuso dietro un'API, ma un modello di cui chiunque possa vedere il codice,
i pesi, le scelte e gli errori. Burel è il **seme** di questo percorso: un esperimento
onesto, documentato passo per passo, che chiede di essere ripreso, criticato e fatto
crescere da una comunità.

> **Posizione onesta.** A questa scala Burel non è e non sarà GPT-4. L'obiettivo non è
> competere sui numeri assoluti dei colossi, ma essere il **miglior modello piccolo e
> specializzato possibile** — e, soprattutto, una base aperta su cui costruire insieme.

---

## Cos'è il Nested Learning (in breve e semplice)

Un Transformer normale impara una volta sola: lo addestri, **congeli i pesi**, e da
quel momento in poi quando genera testo non impara più nulla — usa solo quello che ha
in memoria fissa.

Il **Nested Learning** ribalta questa idea. Il modello ha più **livelli di memoria che
continuano ad aggiornarsi**, anche *mentre legge e genera* (questo si chiama
**Test-Time Training**). È come avere, dentro al modello, dei "quaderni" a velocità
diverse:

- alcuni cambiano lentissimo (la conoscenza generale, stabile);
- altri si aggiornano in fretta, prendendo appunti **al volo** su ciò che sta leggendo
  in questo momento.

Tecnicamente: la memoria si aggiorna con dei **gradienti di secondo ordine** durante
l'inferenza, e c'è persino un **optimizer appreso** (la rete impara *come* aggiornare
la propria memoria, invece di usare una regola fissa). Memoria dentro la memoria:
"nested".

### Il senso — cosa voglio provare

L'ipotesi che Burel vuole mettere alla prova è semplice da enunciare:

> **A parità di parametri e di dati, un modello che continua a imparare al momento
> (nested / test-time training) capisce e ricorda meglio di un Transformer classico
> che ha i pesi congelati?**

Se la risposta è sì — anche solo su scala piccola e in modo misurabile — allora è una
strada interessante per costruire AI **efficienti**, che non hanno bisogno di essere
gigantesche per essere capaci: imparano un po' anche *usando*, come facciamo noi.
Burel esiste per verificare questa intuizione in modo pulito, **una variabile alla
volta**, con i numeri sul tavolo (vedi `MODIFICHE.md`).

---

## Il modello (pesi scaricabili)

I pesi addestrati **non** sono nell'albero del repo (file troppo grande per Git): sono
allegati come **GitHub Release**.

➡️ Scarica `modello_burel_best_v2.pt` (~231 MB) dalla
[**pagina Releases**](../../releases) — vedi anche [`modello/README.md`](modello/README.md).

| versione | parametri | dataset | risultato |
|----------|-----------|---------|-----------|
| v1 | 14M | TinyShakespeare (char) | val loss best **1.5429** |
| **v2** | **20M** | **TinyStories (BPE 16k)** | **val loss best 2.4831** — storie coerenti, arco completo |

Il diario completo degli esperimenti (cosa è cambiato, perché, e se ha migliorato o
peggiorato) è in [`MODIFICHE.md`](MODIFICHE.md).

---

## Struttura

```
Burel/
├── pyproject.toml          # package installabile (pip install -e .)
├── requirements.txt
├── LICENSE                 # Apache 2.0
├── configs/
│   └── config.yaml         # iperparametri (tarati per Colab T4/L4 + budget piccolo)
├── notebooks/
│   └── Burel_Colab.ipynb   # launcher Colab end-to-end
├── burel/                  # il package
│   ├── paths.py            # risoluzione centralizzata dei percorsi
│   ├── model/
│   │   ├── layers.py       # Encoder, PositionalEncoding, AttentionPooling
│   │   ├── memory.py       # NESTED LEARNING: FunctionalMemory, DeepOptimizer, CMS
│   │   └── hope.py         # BurelLM (Transformer + CMS, causale)
│   ├── data/
│   │   ├── shakespeare.py  # fase 1: TinyShakespeare char-level
│   │   ├── tinystories.py  # fase 2: TinyStories + BPE byte-level
│   │   └── codec.py        # encode/decode disaccoppiato (char o BPE da meta.pkl)
│   ├── training/
│   │   └── trainer.py      # loop, resume, checkpoint best/last + backup Drive
│   └── inference/
│       └── sampler.py      # load_model, generate_text
├── scripts/
│   ├── prepare_data.py     # python scripts/prepare_data.py
│   ├── train.py            # python scripts/train.py
│   └── generate.py         # python scripts/generate.py --prompt "..."
└── tests/
    ├── test_causal.py              # prova di causalita' stretta (max|delta|=0)
    └── test_tokenizer_roundtrip.py # round-trip BPE lossless
```

## Cosa è stato adattato dall'architettura originale (e cosa NO)

L'architettura HOPE nasce in un modello finanziario; Burel la porta al **linguaggio**.

**Immutato** (`burel/model/memory.py`): il cuore nested learning — memoria continua a
livelli con frequenze di update esponenziali, optimizer appreso (`DeepOptimizer`),
fast weights, Test-Time Training con `create_graph=True`.

**Adattato al linguaggio** (`burel/model/hope.py`):
- input: `nn.Embedding` di token discreti invece di `nn.Linear` su feature continue;
- output: `lm_head` → logit sul vocabolario invece della regressione di prezzo;
- loss: cross-entropy invece di Huber/SMAPE;
- rimosse le teste multi-task finanziarie;
- aggiunto masking causale nel layer di attention aumentata;
- init GPT-style (std 0.02) sul backbone, lasciando intatta la CMS.

## Quickstart locale

```bash
pip install -e .                 # installa Burel + dipendenze
python scripts/prepare_data.py   # scarica e prepara il dataset (una volta)
python scripts/train.py          # addestra (GPU se disponibile)
python scripts/generate.py --prompt "Once upon a time" --tokens 500 --temperature 0.8
```

Uso programmatico dell'inferenza (dopo aver scaricato il `.pt` dalle Release):

```python
from burel.inference import load_model, generate_text
model, meta = load_model("modello_burel_best_v2.pt")
print(generate_text(model, meta, prompt="Once upon a time", max_new_tokens=500))
```

## Colab

Apri `notebooks/Burel_Colab.ipynb`, seleziona Runtime GPU (T4), esegui le celle: monta
Drive, trova e scompatta `Burel.zip`, `pip install -e .`, prepara, addestra, genera.

### Checkpoint e resume (training lunghi)

`scripts/train.py` salva ad ogni `eval_interval`, in locale **e** su Drive (se montato):
`burel_best.pt` (miglior val loss) e `burel_last.pt` (stato completo per il resume).
Con `resume: auto` (default), rilanciare il training riprende esattamente da dove si era
interrotto. **Early stopping** via `patience: N` ferma al plateau.

## Causalità — stretta e verificata

Il modello è **strettamente causale**: una posizione non vede mai token futuri.
- attention intra-chunk e layer aumentato: causali (mask);
- retrieval dalla memoria: la query del chunk i nasce dal chunk i-1 (o da `init_query`),
  mai dal chunk corrente → nessun leak intra-chunk;
- l'update della memoria del chunk i alimenta solo i chunk successivi.

`python tests/test_causal.py`: stravolgendo i token a posizione > t, i logit delle
posizioni ≤ t restano bit-identici (`max|delta| = 0`).

## Roadmap

1. **[fatto]** architettura + training (resume, early stop) + inferenza, causalità verificata.
2. **[fatto]** Fase 1 — TinyShakespeare char-level: inglese plausibile a livello carattere.
3. **[fatto]** Fase 2 — TinyStories + BPE: storie coerenti con arco completo (v2, val 2.48).
4. **[prossimo]** A/B rigoroso **nested vs Transformer vanilla** a parità di parametri/dati →
   il test centrale dell'ipotesi (vedi sopra).
5. **[futuro]** corpus più ampi, code-gen, e crescita guidata dalla comunità.

## Contribuire

Burel è aperto **apposta** per essere ripreso. Issue, fork, esperimenti, critiche ai
numeri e PR sono benvenuti. La regola del progetto: **un cambiamento alla volta, con il
confronto sui dati** — ogni modifica va annotata in `MODIFICHE.md` dicendo se migliora o
peggiora, e perché.

## Contatti

Per informazioni, collaborazioni o per unirti allo sviluppo:
**Giovanni Canclini** — [info@giovannicanclini.com](mailto:info@giovannicanclini.com)

## Licenza

[Apache License 2.0](LICENSE) — uso libero, anche commerciale, con concessione esplicita
di brevetti. Codice e pesi sono aperti: questo è il punto.
