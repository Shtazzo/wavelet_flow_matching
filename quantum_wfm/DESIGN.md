# [ED] Quantum WFM — design

2026-11-24 · stato: **bozza** · branch `quantum_wfm` · nessun commit senza ok

Convenzione: ogni affermazione ha una fonte (file:riga, comando, misura). Quello che non è verificato è segnato
**da verificare**; quello che propongo io è segnato **proposta mia**.

## 1. Intento (dalle tue parole)

Costruire una rete di Quantum Wavelet Flow Matching (Q-WFM) basata sul WFM di Gabriele Accarino, autore del codice
originale, e confrontare i risultati con i suoi per capire se inserire il quantum porta un vantaggio.
Si parte da Turbulent Radiative Layer 2D (TRL); poi si estende agli altri dataset del paper (Rayleigh–Bénard,
Active Matter). Ora si lavora sul MacBook M3 (MPS); appena possibile, su un supercomputer con GPU NVIDIA (CUDA).

**Requisito**: il codice non dipende né dal dataset né dal device. Cambiare dataset significa cambiare config, non codice.

## 2. Scope

- **Deciso da te**: blocco quantum residuo nel bottleneck della U-Net (approccio B), con warm-start da un
  checkpoint classico (TRL: `WFM_3scale_haar`); PennyLane e Qiskit tramite i loro binding PyTorch.
- **Regola tua (2026-11-24)**: le funzioni di `wavelet_flow_matching/` si usano ma non si modificano mai. Se serve un
  comportamento diverso, si lavora su una copia in `quantum_wfm/` (versione quantum di x), con le righe cambiate
  marcate `[ED]`.
- **Proposte mie**: controllo classico, backbone congelato, baseline riprodotta in locale, criterio di vantaggio
  (§5.6). Fuori dalla prima fase: hardware IBM, simulazione con rumore, training da zero, variante con sostituzione
  dei ResBlock del bottleneck.

## 3. Fatti verificati (2026-11-24)

| Fatto | Fonte |
|---|---|
| Dati TRL in `data/datasets/turbulent_radiative_layer_2D/`: 128×384, 4 campi, scalare `tcool`, 72/9/9 traiettorie × 101 step, 6984 finestre train (218 batch da 32) | h5py, `WellDataset` |
| Checkpoint `WFM_3scale_haar` (TRL): 30.638.391 parametri, epoca 181, val_loss 0,0228; `base_ch` 64, `n_res` 3, `n_levels` 3, `cond_dim` 256, `num_cond` 1; optimizer `warmup_epochs` 20 | `config.json` su HF |
| La normalizzazione del checkpoint TRL coincide con `stats.yaml` (differenza massima 4e-7 nel primo confronto, 0,0 nel confronto di `check_normalizer`). Il modello normalizza con quella del checkpoint e il dataset è costruito con `use_normalization=False`: il controllo segnala solo checkpoint e dataset non corrispondenti | confronto eseguito, revisione finale; `modules/wavelet_fm.py:271-274`, `utils/well_data.py:83` |
| Su HF ci sono anche `WFM_3scale_db4` (RB) e `WFM_3scale_db2` (AM): sono le WFM 3-scale scelte dal paper | `list_repo_files`; `analysis/catalog.py:100-104` |
| Il checkpoint non salva le trasformazioni dei dati. Nei TOML a 3 scale: TRL `transformations = []`; RB `['log_scalars']` (haar, db2, db4, db6); AM `['log_scalars']` solo in db2, `[]` in haar, db4, db6 | `config.json`; `configs/*/WaveletFlow_*3scales*.toml:30` |
| Il bottleneck è una `nn.ModuleList` eseguita come `x = blk(x, cond)` | `models/wavelet_unet.py:337-340, 443-444` |
| Il bottleneck arriva dopo che l'encoder ha iniettato tutte le scale wavelet | `models/wavelet_unet.py:437-440` |
| TRL: bottleneck = 48,8% dei parametri, attivazione (B, 512, 8, 24) | conteggio, hook (config TOML TRL) |
| Step di training su MPS, solo rete con input casuali, fp32, config TRL, B = 32: 2,32 s | misurato |
| PennyLane `default.qubit` + `TorchLayer` su MPS fallisce: la coercizione device forza `cuda`. Vale per la 0.43.1 (`qml_env`; con patch → `Undefined type ComplexDouble`) e per la 0.42.3 installata in `wfm` | `pennylane/math/single_dispatch.py:762` (0.43.1), `:738-739` (0.42.3), eseguito |
| VQC 8 qubit, 4 layer, CPU, forward+backward: PennyLane backprop 0,03 s (B = 32), 2,0 s (B = 6144); `lightning.qubit` adjoint 42 s (B = 6144) | misurato in `qml_env` |
| Qiskit `EstimatorQNN` + `TorchConnector`, B = 32, forward+backward (parameter-shift): 144 s (Statevector), 163 s (Aer); forward Aer 0,087 s | misurato in `qml_env` |
| Wheel di `qiskit-aer-gpu` e `pennylane-lightning-gpu` solo per Linux | PyPI |
| In `wfm` si possono aggiungere pennylane 0.42.3, qiskit 2.6.2, qiskit-machine-learning 0.9.1, qiskit-aer 0.17.2 senza cambiare pacchetti esistenti. `pennylane-qiskit` è incompatibile (`sympy<1.13` contro `sympy==1.13.1` di torch 2.6) | `pip --dry-run` |
| Inferenza originale: `WaveletFlowModule.from_checkpoint` importato dentro `main()`; Euler di default; `ckpt_path` ha la precedenza su `ckpt_name`; override da CLI con `--override` | `inference_wavelet_fm.py:726, 797, 745, 740-741, 159` |
| Valori del paper, TRL `WFM_3scale_haar`, media ± std su 3 seed. VRMSE: 0,549 ± 0,007 / 0,824 ± 0,017 / 0,797 ± 0,014 (orizzonti 1–32 / 33–64 / 65–97). CRPS: 0,410 ± 0,003 / 1,018 ± 0,011 / 1,178 ± 0,030 | `artifacts/tables/csv/S12, S13` |
| CSV per-seed del paper non disponibili in locale, solo tabelle aggregate (`artifacts/tables/csv`); `experiments/` contiene solo i run locali | `ls` |

## 4. Approcci considerati

- **A — Q-cond**: VQC sugli input di condizionamento (t e scalari) al posto di `time_embed` e `scalar_proj`. È il più
  economico ed eseguibile su hardware, ma la parte quantum è piccola (TRL: 0,9% dei parametri).
- **B — Q-bottleneck (scelto)**: blocco quantum residuo su ogni posizione del bottleneck.
- **Scartati (valutazione mia)**: rete interamente quantum (l'uscita ha tanti valori quanti il campo, ~2×10⁵ per TRL);
  quanvolution sulle scale fini (troppi circuiti per step su CPU).

## 5. Design

### 5.1 Indipendenza dal dataset

| Cosa | Da dove arriva |
|---|---|
| Architettura (canali, scale, wavelet, numero di condizioni scalari, forme) | hyperparameter del checkpoint (`from_checkpoint`) |
| Normalizzazione | checkpoint |
| Impostazioni dati del training (`n_steps_input`/`output`, `batch_size`, `transformations`, `augmentations`) | TOML originale del dataset, indicato nel config Q |
| Valutazione | YAML originale del dataset più `--override` per path, checkpoint e seed |
| Checkpoint di partenza di default | **proposta mia**: la WFM 3-scale scelta dal paper (`catalog.BEST_WFM_PER_PDE`), sovrascrivibile da config |
| Dimensioni del blocco | larghezza da `net.chs[-1]`, condizionamento da `net._cond_dim`, posizioni dalla forma del tensore |

Il codice non contiene costanti del dataset. Il config Q contiene: path del TOML originale, nome del dataset, path
locali, sezioni nuove `[quantum]` e `[finetune]`.

### 5.2 Warm-start

Si carica il checkpoint classico già addestrato da Accarino e si aggiunge il blocco con uscita inizialmente zero:
all'avvio il modello è identico al suo. Si addestra solo il blocco.
Limite: si misura quanto un blocco quantum aggiunge a una WFM già ottimizzata, non come sarebbe una WFM nata quantum.

### 5.3 Blocco

```
x : (B, C_b, H_b, W_b)   attivazione del bottleneck (TRL: 512, 8, 24)     c : (B, cond_dim)
h = x per posizione → (B·H_b·W_b, C_b)
a = π · tanh(W_in h + W_c c)      n_q angoli; W_c c ripetuto su tutte le posizioni
z = f(a) ∈ [-1, 1]^{n_q}
y = x + W_out z                   W_out, b_out = 0 → identità all'avvio
```

- **f quantum**: n_q qubit, L layer. Ogni layer: `RX(a_i)` (re-uploading), poi `RZ·RY·RZ` per qubit con pesi
  allenabili, poi anello di CNOT. Uscita: ⟨Z_i⟩. Il circuito è scritto gate per gate (niente template), così è
  identico in PennyLane e in Qiskit.
- **f classica (controllo, proposta mia)**: MLP n_q → k → n_q con tanh, k scelto in modo che i pesi siano circa
  3·n_q·L (per n_q = 8, L = 4: k = 5, 93 pesi contro 96). Degli 96, gli 8 dell'ultimo `RZ` dell'ultimo layer hanno
  sempre gradiente zero (misurato): quelli effettivi sono 88.
- **Inserimento**: `net.bottleneck.append(block)`, dopo i ResBlock originali. Nessuna modifica alla U-Net.
- Default n_q = 8, L = 4 (configurazione misurata), nel config.

### 5.4 Training

- **Proposta mia**: backbone congelato, si allenano solo `W_in`, `W_c`, f, `W_out`. Così l'unica differenza tra
  Q-WFM e C-WFM è f. f viene inizializzata con un RNG separato (`torch.random.fork_rng`), quindi con lo stesso seed
  Q e C hanno gli stessi `W_in`, `W_c`, `W_out` iniziali e lo stesso ordine dei dati (`check_q_c_same_init`).
  Al primo step solo `W_out` riceve gradiente: è previsto (`W_out = 0`).
- Loss originale (`_compute_loss`) invariata.
- Optimizer dalla sezione `[finetune]`, non quello del checkpoint: con `warmup_epochs` 20 del checkpoint
  (`config.json`) un fine-tuning corto resterebbe nel warmup (`modules/wavelet_fm.py:608-617`).
  Default (proposta mia): AdamW, lr 1e-3, warmup 0, coseno da lr all'epoca 0, 10 epoche, uguale per Q e C.
  Lo scheduler originale fa il warmup per epoca partendo da 1e-3·lr: con warmup 1 l'epoca 0 gira a 1e-6, con warmup 0
  tutto il run (revisione finale). Con warmup 0 `QuantumWaveletFlowModule.configure_optimizers` usa solo il coseno
  (`check_finetune_lr_schedule`).
- Device: U-Net su MPS o CUDA. f quantum su CPU se l'input è su MPS, altrimenti sul device dell'input (CUDA:
  **da verificare** sul cluster). f sempre in float32.

### 5.5 PennyLane e Qiskit

- **PennyLane** (`qml.qnn.TorchLayer`, `default.qubit`, backprop): training e inferenza.
- **Qiskit** (`EstimatorQNN` + `TorchConnector`): stesso circuito, pesi copiati da PennyLane. Serve per il check di
  equivalenza e per l'inferenza su sottoinsiemi. Non per il training (misura in §3).
- Si sceglie con `backend = "pennylane" | "qiskit"` nel config.

### 5.6 Valutazione e confronto (per ogni dataset)

1. **Baseline**: inferenza originale con il checkpoint di partenza, seed 42/84/168, confrontata con le tabelle del
   paper (`artifacts/tables/csv`, che coprono tutti e tre i dataset).
2. **C-WFM e Q-WFM**: stessa inferenza, con `quantum_wfm/inference_qwfm.py`. È una copia di
   `inference_wavelet_fm.py` in cui cambiano, marcate `[ED]`, solo le righe che importano e caricano il modello.
3. **`compare.py --pde <nome>`**: legge i CSV con una sua funzione e calcola gli aggregati con le funzioni del paper
   (`catalog.add_features`, builder di `analysis/tables.py`) per baseline, C e Q, affiancati ai valori del paper.
   **Criterio (proposta mia)**: c'è vantaggio se Q batte sia C sia la baseline di oltre 2σ tra seed.

### 5.7 File (tutti in `quantum_wfm/`)

| File | Contenuto |
|---|---|
| `models/circuits.py`, `models/layers.py`, `models/bottleneck.py` | circuito (PennyLane e Qiskit), f quantum e classica, blocco residuo |
| `modules/qwfm.py` | `QuantumWaveletFlowModule` (sottoclasse di `WaveletFlowModule`, `from_checkpoint` che ricostruisce il blocco), freeze |
| `train_qwfm.py` | config Q + TOML originale → checkpoint (path o HF) → blocco → `WellDataModuleLightning` → `pl.Trainer(accelerator="auto")` |
| `inference_qwfm.py` | copia di `inference_wavelet_fm.py`, con import e caricamento di `QuantumWaveletFlowModule` (righe `[ED]`) |
| `analysis/compare.py` | tabella baseline / C / Q / paper per un dataset |
| `configs/` | un config Q e un config C per dataset (si parte da TRL) |
| `checks/check_qwfm.py` | check (§5.8) |
| `requirements.txt` | versioni delle librerie quantum, uguali in locale e sul cluster |

Output in `experiments/<pde>/` nella root: è già in `.gitignore` ed è il percorso letto da `analysis/paths.py`.

### 5.8 Check

- Zero-init: con stesso input e stesso rumore, Q-WFM dà esattamente l'output del modello originale.
- PennyLane e Qiskit, a parità di pesi, danno le stesse uscite (tolleranza 1e-5).
- Dopo uno step: backbone invariato, blocco cambiato.
- Salvataggio e ricaricamento del checkpoint Q: output identico.
- Gli stessi check passano su forme diverse: input finti con risoluzione e canali di RB e AM dalla tabella del README,
  `num_cond` e wavelet dai TOML `configs/<pde>`. Così l'indipendenza dal dataset è verificata e non solo dichiarata.
  L'ordine H/W di RB è da verificare; per il check non conta.

### 5.9 Da chiarire prima di estendere (fatti nei config originali)

- **AM**: il training usa `transformations = ['log_scalars']`
  (`configs/active_matter/WaveletFlow_active_matter_3scales_db2_v4.0.toml:30`), l'inferenza `transform: []`
  (`configs/active_matter/inference_active_matter_seed.yaml:38`). Le varianti AM haar, db4, db6 usano `[]`:
  `base_config` deve essere il TOML del checkpoint di warm-start, e il codice non lo controlla.
- **RB**: l'inferenza ha `num_sampling_steps: 50` (`configs/rayleigh_benard/inference_rayleigh_benard_seed.yaml:53`)
  e `ode_method: "heun"` (`:50`), mentre `scripts/inference_rayleigh_benard.sh:23` usa `euler`.
  Con il fix `[ED]` (`inference_wavelet_fm.py:798`) quel valore diventa effettivo. Quale valore era effettivo nei
  run del paper è **da verificare**.

## 6. Costi per TRL su M3

**Misurato** (smoke del Task 4, 2026-11-24; `step_times.csv` di `PerformanceTrackingCallback`, senza sync del device;
backbone congelato, B = 32, secondo step):

| Run | s/step |
|---|---|
| Q-WFM (8 qubit, 4 layer, f su CPU) | 7,85 |
| C-WFM | 0,67 |

**Stime** da queste misure, 218 step per epoca:
- training Q, 10 epoche: ≈ 4,8 h più validazione;
- training C: ≈ 25 min più validazione.

Inferenza su 3 seed (non misurata a batch 8): ~3 h per baseline e per C; per Q di più, perché il blocco quantum pesa
anche lì.

## 7. Lasciato al builder

Worker del DataLoader, `pin_memory` su MPS, nomi dei file metrica compatibili con `catalog.files_for`,
registrazione a runtime dei nuovi modelli in `catalog.MODEL_REGISTRY`, logging.
