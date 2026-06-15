# MODIFICHE — Burel

Lista sintetica e numerata di tutte le modifiche al progetto. Serve a tracciare se,
cambiando, miglioriamo o peggioriamo. Aggiornare ad ogni cambiamento.

## Modifiche al progetto (cronologiche)

1. **Nascita di Burel** — preso il modello HOPE (Titans MAC + Nested Learning) dal progetto
   finanziario ModelMango e adattato al linguaggio: input `Embedding` di token (invece di
   `Linear` su feature), output `lm_head` sul vocabolario, loss cross-entropy, rimosse le
   teste finanziarie. Il cuore nested learning (CMS, DeepOptimizer, Test-Time Training) NON
   toccato.
2. **Masking causale** aggiunto nel layer di attention aumentata (richiesto dall'LM).
3. **Residuo di causalita' chiuso** — la query di retrieval della memoria nasce dal chunk
   PRECEDENTE (non dal corrente); primo chunk usa `init_query`. Strettamente causale,
   verificato `max|delta| = 0`.
4. **Inferenza** — `inference/sampler.py` (`load_model`, `generate_text`) + `scripts/generate.py`.
5. **Init pesi GPT-style** (std 0.02) sul backbone, CMS esclusa → loss iniziale da 12.95 a ~4.17.
6. **Barra di avanzamento** (tqdm) con %, ETA, loss corrente.
7. **Resume + checkpoint** — salva `burel_best.pt` e `burel_last.pt` (model+optimizer+iter),
   in locale e su Drive; `resume: auto` riprende da dove interrotto.
8. **GradScaler version-aware** — compatibile con piu' versioni di torch, comportamento fp16
   invariato.
9. **Launcher Colab** (`notebooks/Burel_Colab.ipynb`) — trova lo zip, monta Drive, installa,
   prepara, addestra, genera.
10. **Refactor enterprise** — package `burel/` (model/data/training/inference) + scripts/,
    configs/, tests/, notebooks/, pyproject.toml. Modello verificato bit-identico al pre-refactor.
11. **Early stopping** con `patience` — ferma dopo N eval senza nuovo best, tollera il rumore.
12. **Fase 2 — dataset TinyStories + tokenizer BPE.** Nuovo modulo `burel/data/tinystories.py`
    (HuggingFace `roneneldan/TinyStories`) con BPE **byte-level** (libreria `tokenizers`):
    nessun UNK, riusabile pari pari per il codice. Tra una storia e l'altra si inserisce
    `<|endoftext|>` -> il modello impara i confini inizio-svolgimento-fine. Cap sui token
    (`max_train_tokens`, default 100M) per non sovra-approvvigionare dati. **Il cuore
    (`memory.py`, `hope.py`, `layers.py`) NON e' toccato.** Stesso contratto su disco
    (train.bin/val.bin uint16 + meta.pkl) -> il trainer non cambia (vocab 16k entra in uint16).
13. **Codec testo<->id disaccoppiato** (`burel/data/codec.py`): `encode`/`decode` scelgono
    char-level o BPE dal campo `encoding` in meta.pkl. `inference/sampler.py` ora lo usa al
    posto dello stoi/itos inline -> un solo sampler valido per entrambi i dataset; Shakespeare
    resta identico bit-per-bit.
14. **Dataset selezionabile da config** (`data.name` in config.yaml) con dispatch in
    `burel/data/__init__.py`. `prepare(cfg)` e' idempotente (rigenera solo se la config dati
    cambia); il trainer lo chiama sempre -> cambiare dataset = cambiare solo la config.
15. **Dipendenze**: aggiunti `tokenizers` e `datasets` (requirements.txt + pyproject.toml).
    Nuovo self-test `tests/test_tokenizer_roundtrip.py` (round-trip lossless, id in uint16,
    EOS presente; si salta se `tokenizers` non c'e').
16. **Cache del dataset su Drive** (`data.drive_cache_dir`). `tinystories.prepare` ripristina
    train.bin/val.bin/meta.pkl/tokenizer.json da Drive se la cache combacia (cache_key), invece
    di ri-scaricare e ri-encodare; alla prima costruzione li copia su Drive. Si prepara UNA volta,
    ogni retrain riusa. Il notebook punta la cache sul Drive dell'utente (`.../Burel_data`).
17. **FINDING — il `batch_size` rompe l'ottimizzazione della memoria nested.** Primo run
    TinyStories a batch 48 (per riempire la T4): stallo totale a val ~5.97 (baseline unigram),
    train piatta per 4000 iter, INSENSIBILE al LR (3e-4 e 6e-4 uguali). Tornati al regime v2
    provato (**batch 16**, lr 3e-4) cambiando SOLO il dataset: il modello impara subito
    (val 5.34 a iter 500 e in discesa). Conclusione: i gradienti di 2 ordine del TTT non
    tollerano il batch grande come un Transformer vanilla. **La leva "riempi la VRAM col
    batch" e' OFF per questa architettura**; lo scaling del batch va trattato come esperimento
    a se' (una variabile, con ri-taratura del LR), non come freebie. Nota per la tesi nested-vs-
    vanilla (passo 2): questa sensibilita' al batch e' una caratteristica dell'architettura.

## Esperimenti su config (val loss = piu' bassa e' meglio)

| versione | parametri | iperparametri chiave | val loss best |
|----------|-----------|----------------------|---------------|
| **v1** (baseline) | 4.684.301 | d_model 256, layers 4, ff 1024, dropout 0.1, ctx 256 | **1.6196** (iter 4250) |
| **v2** | 14.052.877 | d_model 384, layers 6, ff 1536, dropout 0.2 | **1.5429** (iter 7500, run completo 8000) → **MEGLIO di v1 (−0.077, ~5%)** |
| **v3a** | 20.171.917 | **TinyStories + BPE 16k**, arch = v2, **batch 48**, lr 3e-4→6e-4, block 256 | **STALLO** a val ~5.97 (= baseline unigram), train piatta per 4000 iter |
| **v3b** | 20.171.917 | come v3a ma **batch 16** (regime v2 provato), lr 3e-4 | **FASE 2 OK** — best val **2.4831** @ iter 19250 (da 9.67). Run fermato manualmente a ~iter 19550 (non plateau pieno ma coda a rendimenti decrescenti, ~0.015 val/1000 iter). Sample a 2.48 visibilmente piu' puliti del 2.58: storie con arco completo, "The end.", personaggi nominati. Soffitto: deriva d'identita' multi-entita' (cat→dog→Spot→Max), non-sequitur = limite del 20M, non risolvibile con piu' iter su TinyStories |

> ⚠️ **v3 — confronto NON diretto con v1/v2.** Cambia il dataset E il tokenizer (char→BPE 16k):
> la val loss in nat/token BPE **non è comparabile** con la val loss char-level (unità diverse,
> ~1 token BPE ≈ più caratteri). v3 si giudica **in assoluto**: (a) curva di loss che scende e
> plateau, (b) **sample leggibili a occhio** — frasi vere, personaggi coerenti, inizio-fine. Il
> verdetto qui è qualitativo (significato), non un Δ numerico contro Shakespeare.

**Esito v1→v2:** v2 vince in modo netto (1.5429 vs 1.6196, −0.077, ~5%), e il margine e' cresciuto
col training (al sorpasso era ~1%, alla fine ~5%): la capacita' in piu' ha pagato man mano. Sample
visibilmente migliore (piu' parole vere, meno token inventati). NON ha rotto 1.5 (pavimento di
entropia del dataset, come previsto). Gap train/val finale ~0.21, simile a v1: il dropout non ha
ristretto il gap finale ma ha permesso di scendere piu' in basso senza esplosione di overfitting.
Costo: 3x parametri, ~4x tempo. Lezione: la capacita' aiuta ma con rendimenti decrescenti su
Shakespeare; il salto vero e' il DATASET. Nota: la T4 era usata solo ~3/15 GB -> margine enorme per
modelli/batch piu' grandi nei prossimi run.

### Modifiche config v1 → v2 (12)

| parametro | v1 | v2 | perche' |
|-----------|----|----|---------|
| `d_model` | 256 | 384 | piu' capacita' (abbassa train e val) |
| `num_encoder_layers` | 4 | 6 | piu' profondita' |
| `dim_feedforward` | 1024 | 1536 | scala con d_model (~4x) |
| `dropout` | 0.1 | 0.2 | chiude il gap train/val (~0.22 in v1) |
| `warmup_iters` | 100 | 200 | modello piu' grosso, partenza piu' morbida |
| `max_iters` | 5000 | 8000 | cap piu' alto; la patience ferma al plateau |

Invariati (regime nested learning, da non toccare per attribuire i risultati): `chunk_size 16`,
`num_mem_levels 3`, `memory_depth 2`, `mem_lr 1e-4`, `persistent_length 4`, `max_memory_length 256`,
`block_size 256`, `batch_size 16`, `learning_rate 3e-4`, `min_lr 3e-5`, `grad_clip 1.0`.

**Atteso v2:** ~1.50-1.55. Oltre, su Shakespeare char-level, e' muro: serve cambiare dataset.

### Modifiche config v2 → v3 (fase 2)

| parametro | v2 | v3 | perche' |
|-----------|----|----|---------|
| `data.name` | (shakespeare char) | **tinystories** | il salto vero: da "imita la forma" a "dice cose con senso" |
| tokenizer | char-level (~65) | **BPE byte-level 16k** | unita' semantiche, riusabile per il codice; +~6M param d'embedding |
| `batch_size` | 16 | **16** (provato 48 → STALLO, vedi #17) | il batch 48 congela l'ottimizzazione nested; 16 e' il regime provato |
| `max_iters` | 8000 | **40000** | corpus molto piu' grande; cap alto, fermato a mano in coda |
| `eval_interval` | 250 | **250** | read fini per diagnosi |
| `sample.start` | `\n` | **"Once upon a time"** | avvio naturale per le storie |

**Architettura IDENTICA a v2** (d_model 384, layers 6, ff 1536, dropout 0.2) e **regime nested
learning intoccato** (`chunk_size 16`, `num_mem_levels 3`, `memory_depth 2`, `mem_lr 1e-4`,
`persistent_length 4`, `max_memory_length 256`, `block_size 256`, `lr 3e-4`). Scelta deliberata:
riempiamo la T4 col **batch**, non coi parametri, cosi' l'unica variabile rispetto a v2 e' il
**dataset** → verdetto attribuibile. Lo scale-up del modello (d_model/layers) e' rimandato a un
eventuale v4, dopo aver validato che TinyStories da' il salto di senso (valida-prima-di-scalare).

**Atteso v3:** sample leggibili (frasi/personaggi coerenti, inizio-fine). Niente conoscenza/fatti,
niente istruzioni, vocabolario da storie per bambini. E' "le elementari": valida il significato e
roda la pipeline BPE/scala per il codice (passo 3). Le storie sono corte → non spreme ancora la
memoria nested (quella brilla a contesto lungo, passo 2/3).

### Fase 2 — CHIUSA (esito)

**Obiettivo raggiunto.** Burel fa il salto "imita la forma" → "dice cose con un senso": inglese e
grammatica veri, personaggi nominati, template storia (inizio-svolgimento-fine + morale), confini
`<|endoftext|>` funzionanti. Best val **2.4831** (perplexity ~12), fermato a mano in coda a
rendimenti decrescenti (l'obiettivo era il significato, non spremere la loss). Deliverable:
`burel_best.pt` su Drive (`Burel/Burel_checkpoints/`).

**Soffitto provato:** un 20M non tiene lo stato del mondo multi-entita' (cat→dog→Spot→Max) — limite
di capacita', non di training: non si rompe con piu' iter su TinyStories, serve piu' scala + dataset
piu' ricco. Pipeline (BPE, dati, cache Drive, scala, regime nested provato) **rodata e riusabile**.

**Lezioni chiave:**
- #17: il `batch_size` grande rompe l'ottimizzazione della memoria nested (TTT 2 ordine). batch 16.
- Cambiare UNA variabile dal last-known-good: avevo cambiato dataset + batch insieme → stallo
  diagnosticato solo tornando al regime v2 esatto.
- val BPE non comparabile con val char (unita' diverse): la fase 2 si e' giudicata sui SAMPLE.

**Prossimo (passo 3, deciso):** corpus di codice (The Stack/CodeParrot) + BPE (riusa la pipeline) +
costruzione di un Transformer vanilla a pari parametri per l'A/B nested-vs-vanilla, dove il contesto
lungo del codice mette finalmente sotto sforzo la memoria nested. Distillazione da Qwen sui DATI come
acceleratore; RL con reward verificabile (test che passano) solo alla fine.
