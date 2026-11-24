# [ED] Quantum WFM — piano di implementazione

**Goal:** blocco quantum residuo nel bottleneck della WFM di Accarino, con warm-start dal suo checkpoint, controllo
classico e confronto con i suoi risultati. Si parte da TRL; il codice non dipende dal dataset.

**Architecture:** una libreria in `quantum_wfm/`: `models/` (circuito, f quantum o classica, blocco residuo) e
`modules/qwfm.py` (sottoclasse di `WaveletFlowModule`). Attorno: uno script di training (config = TOML originale del dataset più override), una copia
dello script d'inferenza (`inference_qwfm.py`, righe cambiate marcate `[ED]`) e uno script di confronto che riusa
`catalog.add_features` e i builder di `analysis/tables.py`. Il codice originale non viene modificato.

**Tech Stack:** Python 3.10.15, torch 2.6.0, pytorch-lightning 2.6.0, the_well 1.2.0 (env `wfm`); pennylane 0.42.3
(`TorchLayer`), qiskit 2.6.2 + qiskit-machine-learning 0.9.1 (`EstimatorQNN` + `TorchConnector`), qiskit-aer 0.17.2.

**Spec:** `quantum_wfm/DESIGN.md`

## Global Constraints

- Env `wfm`; librerie quantum con le versioni esatte di `quantum_wfm/requirements.txt`; niente `pennylane-qiskit`.
- File nuovi solo in `quantum_wfm/`. Dati in `data/` nella root, output in `experiments/<pde>/` nella root.
- `wavelet_flow_matching/` si usa (import, sottoclassi) ma non si modifica mai, né sui file né a runtime. Per un comportamento diverso si fa una copia in `quantum_wfm/` con le righe cambiate marcate `[ED]`. Solo i check possono sostituire oggetti a runtime (dataset finto, contatori), come fa già `check_sampling_steps.py`.
- Nessuna costante del dataset nel codice: architettura e normalizzazione dal checkpoint, impostazioni dati dal TOML originale (`base_config`), valutazione dallo YAML originale con `--override`.
- Nessun device fisso nel codice: `accelerator="auto"`. f gira su CPU se l'input è su MPS o se il backend è `qiskit`, altrimenti sul device dell'input. f calcola in float32 con autocast disattivato e restituisce device e dtype dell'input.
- Blocco residuo con `w_out` inizializzato a zero, inserito con `net.bottleneck.append`. Default `n_qubits = 8`, `n_layers = 4`.
- Optimizer del fine-tuning sempre dalla sezione `[finetune]`, mai quello del checkpoint. Default: AdamW, lr 1e-3, `warmup_epochs` 0 (coseno da lr all'epoca 0), 10 epoche. Uguale per Q e per C.
- Qiskit solo per equivalenza e inferenza: il training con `backend = "qiskit"` viene rifiutato.
- Check: funzioni `check_*` con `assert` in `quantum_wfm/checks/check_qwfm.py`. Si lanciano con `KMP_DUPLICATE_LIB_OK=TRUE conda run -n wfm python quantum_wfm/checks/check_qwfm.py [nome_check ...]` (senza nomi: tutti). Il runner stampa `PASS <nome>` o `FAIL <nome>: <errore>`; exit code 1 se qualcosa fallisce.
- Commit: mai senza aver mostrato file, diff e messaggio e aver ricevuto un ok esplicito. Messaggio `[ED][quantum] ...` scritto in `commit_message.txt` (già in `.gitignore`), poi `git commit -F commit_message.txt`. Nessuna riga `Co-Authored-By`. Niente push.
- Nei documenti, ogni affermazione ha una fonte oppure è segnata "da verificare" o "proposta mia".

## Review Focus

1. **bf16 sul cluster**: un input in mixed precision arriva a f. f deve calcolare in float32 e il blocco restituire il dtype dell'input → `check_block_autocast_bf16` (Task 3).
2. **Checkpoint classico** (HF) passato al loader Q: deve caricarsi come WFM normale, senza blocco → `check_classical_ckpt_loads_without_block` (Task 3).
3. **Normalizzazione diversa**: `stats.yaml` del dataset diverso dalla normalizzazione del checkpoint. Non riscala i dati (il modello usa quella del checkpoint, `wavelet_fm.py:271-274`; dataset con `use_normalization=False`, `well_data.py:83`), ma indica checkpoint e dataset non corrispondenti → `check_normalizer_guard` (Task 4).
4. **Optimizer del checkpoint** (`warmup_epochs` 20) che finisce nel fine-tuning e blocca il learning rate → `check_finetune_optimizer`; learning rate reale per epoca dei config Q e C → `check_finetune_lr_schedule` (Task 4).
5. **Backend Qiskit nel training**: ore per step → `check_qiskit_training_refused` (Task 4).

## Mappa dei file

| File | Responsabilità | Task |
|---|---|---|
| `quantum_wfm/requirements.txt` | versioni delle librerie quantum | 1 |
| `quantum_wfm/__init__.py`, `models/__init__.py`, `modules/__init__.py` | package (struttura di libreria come `wavelet_flow_matching/`, richiesta tua) | 3 |
| `quantum_wfm/models/circuits.py` | circuito variazionale, PennyLane e Qiskit | 1–2 |
| `quantum_wfm/models/layers.py` | `QuantumF`, `ClassicalF`, `classical_width`, `make_f`, `swap_backend` | 1–2 |
| `quantum_wfm/models/bottleneck.py` | `BottleneckBlock` | 3 |
| `quantum_wfm/modules/qwfm.py` | `QuantumWaveletFlowModule` | 3 |
| `quantum_wfm/checks/check_qwfm.py` | tutti i check | 1–6 |
| `quantum_wfm/train_qwfm.py` | training (fine-tuning del blocco) | 4 |
| `quantum_wfm/configs/{qwfm,cwfm}_trl_3scale_haar.toml` | config Q e C per TRL | 4 |
| `quantum_wfm/inference_qwfm.py` | copia di `inference_wavelet_fm.py` che carica `QuantumWaveletFlowModule` | 5 |
| `quantum_wfm/analysis/compare.py` (+ `analysis/__init__.py`) | confronto baseline / C / Q / paper | 6 |

---

### Task 1: Ambiente e funzione f (PennyLane e classica)

**Files:** Create `quantum_wfm/requirements.txt`, `quantum_wfm/models/circuits.py`, `quantum_wfm/models/layers.py`, `quantum_wfm/checks/check_qwfm.py`

**Interfaces — Produces:**
- `QuantumF(n_qubits: int, n_layers: int, backend: str = "pennylane", init_weights: torch.Tensor | None = None)`, sottoclasse di `nn.Module`.
  `forward(a: Tensor[N, n_qubits]) -> Tensor[N, n_qubits]`, valori in [-1, 1], stesso device e dtype di `a`.
  `get_weights() -> Tensor[n_layers, n_qubits, 3]` (copia staccata su CPU). Attributi `n_qubits`, `n_layers`, `backend`.
  `n_qubits < 2` → `ValueError`.
- `ClassicalF(n: int, k: int)`: `Linear(n, k) → Tanh → Linear(k, n) → Tanh`.
- `classical_width(n_qubits: int, n_layers: int) -> int` = `max(1, round((3·n·L − n) / (2n + 1)))`.
- `make_f(kind: str, n_qubits: int, n_layers: int, backend: str = "pennylane") -> nn.Module`, con `kind` ∈ {`"quantum"`, `"classical"`}.

- [ ] **Step 1: Registra lo stato dell'env, poi installa.**
  `conda run -n wfm python -m pip freeze > <scratchpad>/freeze_before.txt`. `requirements.txt` contiene esattamente:
  `pennylane==0.42.3`, `qiskit==2.6.2`, `qiskit-machine-learning==0.9.1`, `qiskit-aer==0.17.2`.
  Poi `conda run -n wfm python -m pip install -r quantum_wfm/requirements.txt`.
  Atteso: dal confronto con un nuovo `pip freeze` cambiano solo pacchetti aggiunti; torch 2.6.0, numpy 2.2.6, scipy 1.15.2 e sympy 1.13.1 restano invariati.
- [ ] **Step 2: Verifica il punto "da verificare" della spec.** `default.qubit` + `TorchLayer` con input su MPS, con PennyLane 0.42.3.
  Aggiorna la riga corrispondente in `DESIGN.md` §3 con il risultato, che sia un fallimento o meno.
- [ ] **Step 3: Scrivi i check che falliscono.**

```python
def check_f_shapes_and_range():
    for kind in ("quantum", "classical"):
        f = make_f(kind, n_qubits=4, n_layers=2)
        a = torch.randn(37, 4)
        z = f(a)
        assert z.shape == (37, 4) and z.dtype == a.dtype
        assert z.abs().max() <= 1 + 1e-6
        z.sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters())

def check_classical_width():
    assert classical_width(8, 4) == 5
    assert sum(p.numel() for p in ClassicalF(8, 5).parameters()) == 93

def check_quantum_f_on_mps():            # skipped (PASS with note) if MPS is unavailable
    z = QuantumF(4, 2)(torch.randn(6, 4, device="mps"))
    assert z.device.type == "mps"

def check_invalid_qubits():
    try:
        QuantumF(1, 2)
    except ValueError:
        return
    raise AssertionError("n_qubits=1 accepted")
```

- [ ] **Step 4: Esegui.** Atteso: `FAIL` con `ImportError` (`qwfm` non esiste).
- [ ] **Step 5: Implementa in `qwfm.py`.**
  Device `qml.device("default.qubit", wires=n)`, `qml.qnode(..., interface="torch", diff_method="backprop")`, `qml.qnn.TorchLayer` con pesi `{"weights": (n_layers, n_qubits, 3)}`.
  Inizializzazione: `init_weights` se c'è, altrimenti uniforme in [0, 2π). Ordine dei gate (identico in Qiskit, Task 2):

```python
for l in range(n_layers):
    for i in range(n): qml.RX(a[..., i], wires=i)                                  # re-uploading
    for i in range(n): qml.RZ(w[l, i, 0], wires=i); qml.RY(w[l, i, 1], wires=i); qml.RZ(w[l, i, 2], wires=i)
    for i in range(n): qml.CNOT(wires=[i, (i + 1) % n])
return [qml.expval(qml.PauliZ(i)) for i in range(n)]
```

  Regola di device dentro `forward`: CPU se `a.device.type == "mps"` o `backend == "qiskit"`, altrimenti `a.device`.
  Esecuzione dentro `torch.autocast(device_type=a.device.type, enabled=False)` (verificato funzionante su MPS e CPU con torch 2.6), con input `float32`; si ritorna `.to(a.device, a.dtype)`.
- [ ] **Step 6: Esegui.** Atteso: `PASS` per i 4 check.
- [ ] **Step 7: Proponi il commit** di `requirements.txt`, `qwfm.py` e `check_qwfm.py` (`[ED][quantum] Add quantum/classical f for the Q-WFM block`). Aspetta l'ok.

### Task 2: Backend Qiskit ed equivalenza con PennyLane

**Files:** Modify `quantum_wfm/models/circuits.py`, `quantum_wfm/models/layers.py`, `quantum_wfm/checks/check_qwfm.py`

**Interfaces — Consumes:** `QuantumF` (Task 1). **Produces:** `QuantumF(..., backend="qiskit")`; `swap_backend(f: QuantumF, backend: str) -> QuantumF` (nuova istanza con `init_weights=f.get_weights()`).

- [ ] **Step 1: Scrivi il check.**

```python
def check_pennylane_equals_qiskit():
    torch.manual_seed(0)
    f_pl = QuantumF(4, 2, "pennylane")
    f_qk = swap_backend(f_pl, "qiskit")
    a = torch.randn(16, 4)
    assert torch.equal(f_qk.get_weights(), f_pl.get_weights())
    assert torch.allclose(f_pl(a), f_qk(a), atol=1e-5)
```

- [ ] **Step 2: Esegui.** Atteso: `FAIL` (`swap_backend` non definita).
- [ ] **Step 3: Implementa il backend Qiskit.**
  `ParameterVector("x", n)` e `ParameterVector("w", n_layers·n·3)`, stessi gate del Task 1 (`rx`, `rz`, `ry`, `rz`, `cx`); il peso `w[l, i, j]` ha indice `(l·n + i)·3 + j`.
  Osservabili `SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n)`.
  `EstimatorQNN(circuit, observables, input_params, weight_params, estimator=StatevectorEstimator(), input_gradients=True)`, poi `TorchConnector(qnn, initial_weights=init.flatten())`.
  Se l'equivalenza non passa, l'errore sta nella mappatura qubit/osservabili: il check ha l'ultima parola, la tolleranza non si allarga.
- [ ] **Step 4: Esegui.** Atteso: `PASS` per tutti i check.
- [ ] **Step 5: Proponi il commit** (`[ED][quantum] Add Qiskit backend with PennyLane equivalence check`). Aspetta l'ok.

### Task 3: Blocco residuo e `QuantumWaveletFlowModule`

**Files:** Create `quantum_wfm/models/bottleneck.py`, `quantum_wfm/modules/qwfm.py`, `quantum_wfm/checks/check_qwfm.py`

**Interfaces — Consumes:** `make_f`, `swap_backend`. **Produces:**
- `BottleneckBlock(channels: int, cond_dim: int, f: nn.Module, n_qubits: int)`.
  `forward(x: Tensor[B, C, H, W], c: Tensor[B, cond_dim]) -> Tensor[B, C, H, W]`.
  Attributi `w_in = Linear(C, n)`, `w_c = Linear(cond_dim, n, bias=False)`, `w_out = Linear(n, C)` (peso e bias a zero), `f`.
  Formula: `y = x + w_out(f(π·tanh(w_in(h) + w_c(c) ripetuto su H·W)))`, con h = x per posizione.
- `QuantumWaveletFlowModule(WaveletFlowModule)`, con metodi:
  - `attach_block(qblock: dict) -> None`: chiavi `kind`, `n_qubits`, `n_layers`, `backend`; usa `channels = net.chs[-1]` e `cond_dim = net._cond_dim`; salva `hparams["qblock"] = {**qblock, "index": i}`.
  - `block` (property) `-> BottleneckBlock | None`.
  - `freeze_backbone() -> None`.
  - `set_backend(backend: str) -> None`: sostituisce `block.f` con `swap_backend`.
  - `from_checkpoint(path, map_location="cpu", stats=None, backend: str | None = None)` (classmethod).

- [ ] **Step 1: Scrivi i check.**
  Helper `tiny_module(data_shape, channels, num_cond, wavelet)` costruito come in `check_sampling_steps.py:53-63`, con `base_ch=8`, `n_res=1`, `n_levels=3`, `n_scales=3`, `cond_dim=16`, 3 frame di contesto.
  Helper `fake_net_inputs(m, B=2)` che restituisce i kwargs di `m.net(...)`. Blocco per i check: `QB = dict(kind="quantum", n_qubits=2, n_layers=1, backend="pennylane")`.

```python
SHAPES = {   # data_shape/canali: tabella README; num_cond/wavelet: TOML configs/<pde> (ordine H/W di RB da verificare)
    "trl": dict(data_shape=(128, 384), channels=4,  num_cond=1, wavelet="haar"),
    "rb":  dict(data_shape=(128, 512), channels=4,  num_cond=2, wavelet="db4"),
    "am":  dict(data_shape=(256, 256), channels=11, num_cond=3, wavelet="db2"),
}
def check_zero_init_identity():                       # per ogni SHAPES
    m = tiny_module(**s); kw = fake_net_inputs(m); ref = m.net(**kw)
    m.attach_block(QB); out = m.net(**kw)
    assert all(torch.equal(ref[i], out[i]) for i in ref)
def check_freeze_and_step():                          # backbone uguale, w_out cambiato dopo uno step AdamW
def check_checkpoint_roundtrip():                     # w_out ≠ 0, torch.save({"state_dict", "hyper_parameters": dict(m.hparams)}),
                                                      # from_checkpoint → stessi output di m.net; block non None
def check_classical_ckpt_loads_without_block():       # WFM classico salvato → from_checkpoint → block is None, stessi output
def check_block_autocast_bf16():                      # torch.autocast("cpu", dtype=torch.bfloat16): output float32 e finito
```

- [ ] **Step 2: Esegui.** Atteso: `FAIL` (`BottleneckBlock` non definita).
- [ ] **Step 3: Implementa `from_checkpoint`.**
  1. `torch.load(path, map_location=map_location)`.
  2. Senza `"qblock"` negli hyperparameter: `super().from_checkpoint(path, map_location, stats)`.
  3. Con `"qblock"`: togli le chiavi `net.bottleneck.{index}.*`, salva il checkpoint ridotto in un `io.BytesIO`, chiama `super().from_checkpoint(buf, ...)`, poi `attach_block`, poi `load_state_dict(state_dict_completo, strict=True)`.
  4. Se `backend` è dato e diverso dal backend salvato, `set_backend(backend)`.
- [ ] **Step 4: Esegui.** Atteso: `PASS` per tutti i check, sulle tre forme.
- [ ] **Step 5: Proponi il commit** (`[ED][quantum] Add residual bottleneck block and QuantumWaveletFlowModule`). Aspetta l'ok.

### Task 4: Training, config TRL e misura dei costi

**Files:** Create `quantum_wfm/train_qwfm.py`, `quantum_wfm/configs/qwfm_trl_3scale_haar.toml`, `quantum_wfm/configs/cwfm_trl_3scale_haar.toml`; Modify `quantum_wfm/checks/check_qwfm.py`

**Interfaces — Consumes:** `QuantumWaveletFlowModule`. **Produces:**
- `load_config(path) -> munch.Munch`: `toml.load(base_config)` con sopra il TOML Q (merge ricorsivo, vince il Q). I path relativi (`path.*`, `warm_start.ckpt_path`) si risolvono rispetto alla root del repo.
- `resolve_checkpoint(cfg) -> str`: `warm_start.ckpt_path`, altrimenti `hub.download_checkpoint(pde, warm_start.model)`. Se `model` manca: `catalog.wfm_key("3scale", catalog.BEST_WFM_PER_PDE[pde]["3scale"])`.
- `finetune_optimizer(ft) -> OptimizerConfig`: `cls="AdamW"`, `lr`, `weight_decay`, `betas=(0.9, 0.999)`, `warmup_epochs`, `scheduler_cls="CosineAnnealingLR"`, `eta_min`.
- `check_normalizer(a, b, atol=1e-5) -> None`: `ValueError` se mean, std, mean_delta o std_delta differiscono.
- `main(argv: list[str]) -> None`, CLI `-c CONFIG [--smoke]`. `--smoke` imposta `limit_train_batches=2`, `limit_val_batches=1`, 1 epoca, e stampa `s/step`.

Config TRL (Q; il C è identico con `kind = "classical"` ed `exp_name = "CWFM_turbulent_radiative_layer_2D_3scale_haar"`):

```toml
base_config = "configs/turbulent_radiative_layer_2D/WaveletFlow_turbulent_radiative_layer_2D_3scales_haar_ablation_v4.0.toml"
[main]
    exp_name = "QWFM_turbulent_radiative_layer_2D_3scale_haar"
[path]
    well_base_path   = "data/datasets"
    stats_path       = "data/datasets/turbulent_radiative_layer_2D/stats.yaml"
    experiments_path = "experiments/turbulent_radiative_layer_2D"
[warm_start]
    model = "WFM_3scale_haar"
[quantum]
    kind = "quantum"
    n_qubits = 8
    n_layers = 4
    backend = "pennylane"
[finetune]
    epochs = 10
    lr = 1e-3
    warmup_epochs = 1
    weight_decay = 1e-2
    eta_min = 1e-7
[lightning.datamodules]
    data_workers = 8
[lightning.trainer]
    accelerator = "auto"
    strategy = "auto"
    precision = "32-true"
```

Flusso di `main`:
1. Valida `[quantum]`: `backend = "qiskit"` → `ValueError` subito.
2. Seed, poi `resolve_checkpoint`, poi `from_checkpoint`. Se il checkpoint ha già `qblock`: `ValueError`, perché il resume non è previsto.
3. `attach_block`, `freeze_backbone`, `optimizer_cfg = finetune_optimizer(...)`, scritto anche in `hparams`.
4. `WellDataModuleLightning` come in `train_wavelet_fm.py:73-100`, con trasformazioni e augmentations dal TOML base.
5. `check_normalizer(model.normalizer, data_module.normalizer)`.
6. `ExperimentManager` e logger come in `train_wavelet_fm.py:131-168`.
7. `pl.Trainer(max_epochs=finetune.epochs, devices=1, ...)` e `fit`.

- [ ] **Step 1: Scrivi i check.**
  - `check_config_merge`: TOML base e Q temporanei. Il Q vince su `path` e `lightning.trainer.precision`; `train.transformations` e `train.n_steps_input` vengono dal base.
  - `check_finetune_optimizer`: `warmup_epochs == 1`, `lr == 1e-3`, `cls == "AdamW"`.
  - `check_normalizer_guard`: normalizzazioni uguali passano; con std diversa di 1e-3 → `ValueError`.
  - `check_qiskit_training_refused`: `main` con `backend = "qiskit"` → `ValueError` prima di toccare checkpoint e dati.
- [ ] **Step 2: Esegui.** Atteso: `FAIL` (`train_qwfm` non esiste).
- [ ] **Step 3: Implementa** `train_qwfm.py` e i due TOML.
- [ ] **Step 4: Esegui i check.** Atteso: `PASS`.
- [ ] **Step 5: Smoke su dati veri.**
  `conda run -n wfm python quantum_wfm/train_qwfm.py -c quantum_wfm/configs/qwfm_trl_3scale_haar.toml --smoke`, poi lo stesso con il config C.
  Atteso: exit 0, `s/step` stampato, un `*_portable.ckpt` in `experiments/turbulent_radiative_layer_2D/QWFM_.../artifacts/checkpoints/`.
  Quel file, caricato con `QuantumWaveletFlowModule.from_checkpoint`, ha `block` diverso da `None`.
  Scrivi i `s/step` misurati di Q e C in `DESIGN.md` §6, al posto delle stime.
- [ ] **Step 6: Proponi il commit** (`[ED][quantum] Add Q-WFM fine-tuning script and TRL configs`). Aspetta l'ok.

### Task 5: Inferenza (copia di `inference_wavelet_fm.py`)

**Files:** Create `quantum_wfm/inference_qwfm.py` (copia di `wavelet_flow_matching/inference_wavelet_fm.py`); Modify `quantum_wfm/checks/check_qwfm.py`

**Interfaces — Consumes:** `QuantumWaveletFlowModule.from_checkpoint(path, map_location, backend=...)`. **Produces:** stessa CLI e stessi output dell'originale (`--config YAML [--override K=V ...]`), più la chiave `backend` (`pennylane` | `qiskit`; se assente, quello salvato nel checkpoint).

Righe cambiate nella copia, ognuna con commento `# [ED] ...`:
- docstring: una riga che dice da quale file è copiata e cosa cambia;
- riga 726 dell'originale, `from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule` → `from quantum_wfm.modules.qwfm import QuantumWaveletFlowModule`;
- riga 797, `WaveletFlowModule.from_checkpoint(ckpt_path, map_location=device)` → `QuantumWaveletFlowModule.from_checkpoint(ckpt_path, map_location=device, backend=cfg.get("backend"))`.

`sys.path` non va toccato: la riga 108 dell'originale inserisce la cartella padre dello script, che anche per la copia
in `quantum_wfm/` è la root del repo.

- [ ] **Step 1: Scrivi i check.**
  - `check_inference_copy_diff`: con `difflib` contro l'originale, ogni riga aggiunta o cambiata nella copia contiene `[ED]`, e le righe cambiate sono al massimo 3 più il docstring.
  - `check_inference_driver`: riusa `TinyWell` e lo schema di `check_sampling_steps.py:53-89`. Modulo minimo con blocco (`n_qubits=2`, `n_layers=1`) e `w_out ≠ 0`; YAML con `num_sampling_steps: 2` e `physics.enabled: false`.
    Conta le chiamate a `BottleneckBlock.forward`. Atteso: `2 file × (8 − 4) frame × 2 step = 16` chiamate e un `*_metrics.csv` in `experiments/rayleigh_benard/results/`.
    Seconda esecuzione con `--override backend=qiskit`: `block.f.backend == "qiskit"` e CSV presente.
- [ ] **Step 2: Esegui.** Atteso: `FAIL` (la copia non esiste).
- [ ] **Step 3: Crea la copia** con `cp` e applica le 3 modifiche `[ED]`.
- [ ] **Step 4: Esegui.** Atteso: `PASS`.
- [ ] **Step 5: Proponi il commit** (`[ED][quantum] Add Q-WFM copy of the inference script`). Aspetta l'ok.

### Task 6: Confronto con il paper

**Files:** Create `quantum_wfm/analysis/compare.py`; Modify `quantum_wfm/checks/check_qwfm.py`

**Interfaces — Produces:** CLI `compare.py --pde PDE --paper-model MODEL --baseline LABEL=STEM [--c LABEL=STEM] [--q LABEL=STEM]`. Funzioni:
- `load_runs(pde: str, runs: dict[str, str]) -> DataFrame`: per ogni `label → stem` e ogni seed in `catalog.SEEDS` legge `experiments/<pde>/results/{stem}_nsampling*_nens*_seed{S}_metrics.csv` (esattamente un file per seed, altrimenti errore). Imposta `model=label`, `pde=pde` e `seed=S` se manca, toglie le righe `field == catalog.GLOBAL_FIELD` e restituisce `catalog.add_features(df)`. Niente copie di file e niente scritture in `catalog.MODEL_REGISTRY`.
- `tables_for(df, labels) -> dict[str, tuple[DataFrame, DataFrame]]`: `tables.across_seeds` per i quattro builder (prima verifica che i builder e `_hcat` non leggano `MODEL_REGISTRY`):

```python
TABLES = [("S12_vrmse", T.horizon_table, {"metric": "vrmse"}), ("S13_crps", T.horizon_table, {"metric": "crps"}),
          ("S14_spectral_bands", T.bands_table, {"time_agg": "horizon"}), ("T1_wavelet_selection", T.selection_table, {"time_agg": "horizon"})]
```

- `paper_values(name, model, pde)`: righe di `artifacts/tables/csv/{name}.csv` per quel modello e dataset.
- `advantage(mean, std, q, others) -> Series[bool]`. Più basso è meglio in tutte e quattro le tabelle. Criterio della spec (proposta mia):

```python
def advantage(mean, std, q, others):
    return pd.concat([(mean.loc[o] - mean.loc[q]) > 2 * np.maximum(std.loc[q], std.loc[o]) for o in others], axis=1).all(axis=1)
```

  Uscita: tabelle a schermo e `experiments/<pde>/comparison.csv`.

- [ ] **Step 1: Scrivi `check_compare`.**
  Root temporanea; `compare.py` va eseguito in un sottoprocesso con `WAVELET_FLOW_MATCHING_ROOT=<tmp>`, perché `analysis/paths.py` legge la variabile all'import.
  CSV sintetici per 3 label × 3 seed, step 0..96, colonne `traj`, `field`, `step`, `seed`, `vrmse`, `crps`, `rmse_c_low`, `rmse_c_mid`, `rmse_c_high`. Copia dei CSV del paper S12, S13, S14 e T1 nella root temporanea.
  - Caso 1: Q più basso di 0,1, rumore tra seed 0,001 → `advantage` tutto `True`.
  - Caso 2: Q uguale a C → tutto `False`.
- [ ] **Step 2: Esegui.** Atteso: `FAIL`.
- [ ] **Step 3: Implementa** `compare.py`.
- [ ] **Step 4: Esegui.** Atteso: `PASS`.
- [ ] **Step 5: Proponi il commit** (`[ED][quantum] Add comparison against the paper tables`). Aspetta l'ok.

### Task 7: Esperimenti TRL

I run lunghi vanno in background, con log in `experiments/turbulent_radiative_layer_2D/logs/`.
Variabili: `R=$PWD`; `CK` = path restituito da `wavelet_flow_matching.hub.download_checkpoint("turbulent_radiative_layer_2D", "WFM_3scale_haar")`.

- [ ] **Step 1: Baseline.** Per S in 42 84 168:
  `python wavelet_flow_matching/inference_wavelet_fm.py --config configs/turbulent_radiative_layer_2D/inference_turbulent_radiative_layer_2D_seed.yaml --override project_root=$R well_base_path=$R/data/datasets ckpt_path=$CK seed=$S run_tag=seed$S`
- [ ] **Step 2: Controllo della baseline.**
  `python quantum_wfm/analysis/compare.py --pde turbulent_radiative_layer_2D --paper-model WFM_3scale_haar --baseline WFM_3scale_haar_local=WFM_3scale_haar_turbulent_radiative_layer_2D`
  Atteso: ogni colonna entro 2σ dal valore del paper. **Se non lo è, ci si ferma** e si riporta la differenza: senza baseline riprodotta il confronto non vale.
- [ ] **Step 3: Training C e poi Q** con `train_qwfm.py` e i due TOML (senza `--smoke`).
  Per l'inferenza si usa il `*_portable.ckpt` con `val_loss` minima.
- [ ] **Step 4: Inferenza C e Q.** Per S in 42 84 168:
  `python quantum_wfm/inference_qwfm.py` con gli stessi argomenti dello Step 1, ma `ckpt_path` = checkpoint di C o di Q.
- [ ] **Step 5: Confronto completo.**
  `compare.py ... --c CWFM_trl=CWFM_turbulent_radiative_layer_2D_3scale_haar --q QWFM_trl=QWFM_turbulent_radiative_layer_2D_3scale_haar`
  Riporta all'utente le tabelle, la riga `advantage` e i tempi misurati. Nessun commit di risultati: `experiments/` è ignorata da git.

Prima di estendere a RB e AM vanno chiariti i punti di `DESIGN.md` §5.9. Poi servono solo config nuovi (`base_config`, `exp_name`, path, `warm_start.model`), senza codice nuovo.
