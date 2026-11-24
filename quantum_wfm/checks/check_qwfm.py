#!/usr/bin/env python3
"""
[ED] Checks for the Quantum WFM code in quantum_wfm/ (plan: quantum_wfm/PLAN.md, spec: quantum_wfm/DESIGN.md).

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n wfm python quantum_wfm/checks/check_qwfm.py [check_name ...]

No names: run every check_* function in file order. Prints PASS/FAIL per check, exit code 1 on any failure.
"""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from quantum_wfm.models import layers as QL  # noqa: E402
from quantum_wfm.models.bottleneck import BottleneckBlock  # noqa: E402
from quantum_wfm.modules.qwfm import QuantumWaveletFlowModule  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: f quantum (PennyLane) and classical
# ─────────────────────────────────────────────────────────────────────────────

def check_f_shapes_and_range():
    for kind in ("quantum", "classical"):
        f = QL.make_f(kind, n_qubits=4, n_layers=2)
        a = torch.randn(37, 4)
        z = f(a)
        assert z.shape == (37, 4) and z.dtype == a.dtype, (kind, tuple(z.shape), z.dtype)
        assert z.abs().max() <= 1 + 1e-6, (kind, z.abs().max().item())
        z.sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters()), kind


def check_classical_width():
    assert QL.classical_width(8, 4) == 5, QL.classical_width(8, 4)
    assert sum(p.numel() for p in QL.ClassicalF(8, 5).parameters()) == 93


def check_quantum_f_on_mps():
    if not torch.backends.mps.is_available():
        return "skipped: MPS not available"
    z = QL.QuantumF(4, 2)(torch.randn(6, 4, device="mps"))
    assert z.device.type == "mps", z.device


def check_invalid_qubits():
    try:
        QL.QuantumF(1, 2)
    except ValueError:
        return None
    raise AssertionError("n_qubits=1 accepted")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Qiskit backend == PennyLane
# ─────────────────────────────────────────────────────────────────────────────

def check_pennylane_equals_qiskit():
    torch.manual_seed(0)
    f_pl = QL.QuantumF(4, 2, "pennylane")
    f_qk = QL.swap_backend(f_pl, "qiskit")
    a = torch.randn(16, 4)
    assert f_qk.backend == "qiskit"
    assert torch.equal(f_qk.get_weights(), f_pl.get_weights())
    diff = (f_pl(a) - f_qk(a)).abs().max().item()
    assert diff <= 1e-5, diff
    return f"max |pennylane - qiskit| = {diff:.1e}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: residual bottleneck block and QuantumWaveletFlowModule
# ─────────────────────────────────────────────────────────────────────────────

# data_shape / channels: README data table; num_cond / wavelet: configs/<pde>/WaveletFlow_*3scales*.toml.
# Only the variety of shapes matters here (RB H/W order not verified).
SHAPES = {
    "trl": dict(data_shape=(128, 384), channels=4, num_cond=1, wavelet="haar"),
    "rb": dict(data_shape=(128, 512), channels=4, num_cond=2, wavelet="db4"),
    "am": dict(data_shape=(256, 256), channels=11, num_cond=3, wavelet="db2"),
}
QB = dict(kind="quantum", n_qubits=2, n_layers=1, backend="pennylane")


def tiny_module(data_shape, channels, num_cond, wavelet, cls=None):
    """Small 3-scale WFM with the given data layout (same construction as check_sampling_steps.py:53-63)."""
    from the_well.data.normalization import ZScoreNormalization
    from wavelet_flow_matching.utils.configs import (FlowMatchingConfig, OptimizerConfig, VelocityNetConfig,
                                                     WaveletConfig)
    cls = cls or QuantumWaveletFlowModule
    fields = [f"c{i}" for i in range(channels)]
    zero, one = {f: 0.0 for f in fields}, {f: 1.0 for f in fields}
    norm = ZScoreNormalization({"mean": zero, "std": one, "mean_delta": zero, "std_delta": one}, fields, [])
    vn = VelocityNetConfig(base_ch=8, n_res=1, n_levels=3, cond_dim=16, num_cond=num_cond,
                           context_in_channels=channels, num_context_frames=3)
    torch.manual_seed(0)
    return cls(in_channels=channels, field_names=fields, data_shape=data_shape, n_steps_output=1,
               scale_weights=[1.0, 1.0, 1.0], subband_weights=None,
               wavelet_cfg=WaveletConfig(use_wavelet=True, wavelet=wavelet, n_scales=3),
               velocity_net_cfg=vn, flow_matching_cfg=FlowMatchingConfig(), optimizer_cfg=OptimizerConfig(),
               normalizer=norm)


def fake_net_inputs(m, B=2):
    torch.manual_seed(1)
    (H, W), C = m.data_shape, m.in_channels
    c0 = m._to_coeff_dict_x0(torch.randn(B, C, 1, H, W))
    return dict(curr=c0, x_tau={i: torch.randn_like(v) for i, v in c0.items()}, t=torch.rand(B),
                scalar_conds=torch.randn(B, m.net.scalar_proj.in_features), context=torch.randn(B, C, 3, H, W))


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[i], b[i]) for i in a)


def _save(m, path):
    torch.save({"state_dict": m.state_dict(), "hyper_parameters": dict(m.hparams)}, path)


def check_zero_init_identity():
    for name, s in SHAPES.items():
        m = tiny_module(**s)
        kw = fake_net_inputs(m)
        with torch.no_grad():
            ref = m.net(**kw)
            m.attach_block(QB)
            out = m.net(**kw)
        assert m.block is m.net.bottleneck[-1], name
        assert _same(ref, out), name
    return "trl, rb, am"


def check_q_c_same_init():
    """Q and C differ only in f (DESIGN §5.4): same seed -> same w_in/w_c init and same RNG stream after attach."""
    got = {}
    for kind in ("quantum", "classical"):
        m = tiny_module(**SHAPES["trl"])
        torch.manual_seed(0)
        m.attach_block({**QB, "kind": kind})
        b = m.block
        got[kind] = (b.w_in.weight, b.w_in.bias, b.w_c.weight, torch.rand(4))
    assert all(torch.equal(q, c) for q, c in zip(got["quantum"], got["classical"])), "Q and C differ beyond f"


def check_freeze_and_step():
    m = tiny_module(**SHAPES["trl"])
    m.attach_block(QB)
    m.freeze_backbone()
    idx = m.hparams["qblock"]["index"]
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(f"net.bottleneck.{idx}.") for n in trainable), trainable
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    w_out = m.block.w_out.weight.detach().clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=1e-2)   # all params, as configure_optimizers does
    sum(v.square().mean() for v in m.net(**fake_net_inputs(m)).values()).backward()
    opt.step()
    assert all(torch.equal(p, frozen[n]) for n, p in m.named_parameters() if n in frozen)
    assert not torch.equal(m.block.w_out.weight, w_out)


def check_checkpoint_roundtrip():
    import tempfile
    m = tiny_module(**SHAPES["trl"])
    m.attach_block(QB)
    with torch.no_grad():
        m.block.w_out.weight.normal_()                                     # block output now matters
    kw = fake_net_inputs(m)
    path = Path(tempfile.mkdtemp()) / "q.ckpt"
    _save(m, path)
    m2 = QuantumWaveletFlowModule.from_checkpoint(str(path))
    assert m2.block is not None and m2.hparams["qblock"] == m.hparams["qblock"], m2.hparams.get("qblock")
    with torch.no_grad():
        assert _same(m.net(**kw), m2.net(**kw))


def check_classical_ckpt_loads_without_block():
    import tempfile
    from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule
    m = tiny_module(**SHAPES["trl"], cls=WaveletFlowModule)
    kw = fake_net_inputs(m)
    path = Path(tempfile.mkdtemp()) / "c.ckpt"
    _save(m, path)
    m2 = QuantumWaveletFlowModule.from_checkpoint(str(path))
    assert isinstance(m2, QuantumWaveletFlowModule) and m2.block is None
    with torch.no_grad():
        assert _same(m.net(**kw), m2.net(**kw))


def check_block_autocast_bf16():
    for kind in ("quantum", "classical"):
        block = BottleneckBlock(channels=16, cond_dim=8, f=QL.make_f(kind, 2, 1), n_qubits=2)
        with torch.no_grad():
            block.w_out.weight.normal_()
        x, c = torch.randn(2, 16, 3, 5), torch.randn(2, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = block(x, c)
        assert y.dtype == x.dtype and y.shape == x.shape and torch.isfinite(y).all(), (kind, y.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: fine-tuning script
# ─────────────────────────────────────────────────────────────────────────────

def _train_module():
    import quantum_wfm.train_qwfm as T
    return T


def check_config_merge():
    import tempfile
    import toml
    T = _train_module()
    tmp = Path(tempfile.mkdtemp())
    base = {"path": {"well_base_path": "/glade/d", "well_dataset_name": "pde_x", "stats_path": "/glade/s.yaml",
                     "experiments_path": "/glade/e"},
            "train": {"transformations": ["log_scalars"], "n_steps_input": 4},
            "lightning": {"trainer": {"precision": "bf16-mixed", "accelerator": "gpu"}}}
    q = {"base_config": str(tmp / "base.toml"),
         "path": {"well_base_path": "data/datasets", "stats_path": "data/s.yaml", "experiments_path": "experiments/x"},
         "lightning": {"trainer": {"precision": "32-true"}}}
    (tmp / "base.toml").write_text(toml.dumps(base))
    (tmp / "q.toml").write_text(toml.dumps(q))
    cfg = T.load_config(tmp / "q.toml")
    assert cfg.path.well_base_path == str(REPO / "data/datasets"), cfg.path.well_base_path      # Q wins, resolved
    assert cfg.path.experiments_path == str(REPO / "experiments/x")
    assert cfg.path.well_dataset_name == "pde_x"                                                  # from base
    assert cfg.train.transformations == ["log_scalars"] and cfg.train.n_steps_input == 4         # from base
    assert cfg.lightning.trainer.precision == "32-true" and cfg.lightning.trainer.accelerator == "gpu"


def check_finetune_optimizer():
    import munch
    T = _train_module()
    oc = T.finetune_optimizer(munch.Munch(epochs=10, lr=1e-3, warmup_epochs=1, weight_decay=1e-2, eta_min=1e-7))
    assert (oc.cls, oc.lr, oc.warmup_epochs, oc.scheduler_cls) == ("AdamW", 1e-3, 1, "CosineAnnealingLR"), oc
    assert (oc.weight_decay, oc.eta_min, tuple(oc.betas)) == (1e-2, 1e-7, (0.9, 0.999)), oc


def check_finetune_lr_schedule():
    """Shipped Q and C configs: lr from epoch 0, then cosine down. The original per-epoch LinearLR warmup
    (wavelet_fm.py:608-617) runs epoch 0 at 1e-3 * lr, and with warmup 0 the whole run (final review)."""
    import types
    T = _train_module()
    for name in ("qwfm_trl_3scale_haar.toml", "cwfm_trl_3scale_haar.toml"):
        ft = T.load_config(REPO / "quantum_wfm" / "configs" / name).finetune
        m = tiny_module(**SHAPES["trl"])
        m.attach_block(QB)
        m.freeze_backbone()
        m.optimizer_cfg = T.finetune_optimizer(ft)
        m._trainer = types.SimpleNamespace(max_epochs=int(ft.epochs))      # test-only: no Trainer needed
        out = m.configure_optimizers()
        opt, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
        lrs = []
        for _ in range(int(ft.epochs)):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert abs(lrs[0] - float(ft.lr)) < 1e-12, (name, lrs[:3])
        assert all(a > b for a, b in zip(lrs, lrs[1:])) and lrs[-1] >= float(ft.eta_min), (name, lrs)
    return "lr per epoch " + ", ".join(f"{x:.1e}" for x in lrs[:3]) + ", ..."


def check_normalizer_guard():
    from the_well.data.normalization import ZScoreNormalization
    T = _train_module()

    def norm(std):
        s = {"mean": {"a": 0.0}, "std": {"a": std}, "mean_delta": {"a": 0.0}, "std_delta": {"a": 1.0}}
        return ZScoreNormalization(s, ["a"], [])

    T.check_normalizer(norm(1.0), norm(1.0))
    try:
        T.check_normalizer(norm(1.0), norm(1.001))
    except ValueError:
        return None
    raise AssertionError("different std accepted")


def check_loader_workers():
    # the original DataLoader collate_fn is a lambda (utils/well_data.py:132-134): 'spawn' workers cannot pickle it
    T = _train_module()
    assert T.loader_workers(8, "spawn") == 0
    assert T.loader_workers(8, "forkserver") == 0
    assert T.loader_workers(8, "fork") == 8


def check_qiskit_training_refused():
    import tempfile
    import toml
    T = _train_module()
    tmp = Path(tempfile.mkdtemp())
    q = toml.load(REPO / "quantum_wfm/configs/qwfm_trl_3scale_haar.toml")
    q["quantum"]["backend"] = "qiskit"
    (tmp / "q.toml").write_text(toml.dumps(q))
    original = T.resolve_checkpoint

    def touched(cfg):
        raise AssertionError("checkpoint touched before the backend was validated")

    T.resolve_checkpoint = touched          # test-only patch of our own module
    try:
        T.main(["-c", str(tmp / "q.toml")])
    except ValueError as e:
        assert "qiskit" in str(e), e
        return None
    finally:
        T.resolve_checkpoint = original
    raise AssertionError("backend 'qiskit' accepted for training")


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: inference copy (quantum_wfm/inference_qwfm.py)
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_COPY = REPO / "quantum_wfm" / "inference_qwfm.py"


def check_inference_copy_diff():
    import difflib
    orig = (REPO / "wavelet_flow_matching" / "inference_wavelet_fm.py").read_text().splitlines()
    copy = INFERENCE_COPY.read_text().splitlines()
    diff = list(difflib.unified_diff(orig, copy, lineterm="", n=0))
    added = [d for d in diff if d.startswith("+") and not d.startswith("+++")]
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    assert added and all("[ED]" in d for d in added), [d for d in added if "[ED]" not in d]
    assert len(removed) <= 2, removed               # only the model import and the from_checkpoint call are replaced
    return f"{len(added)} added lines, all [ED]; {len(removed)} original lines replaced"


def check_inference_driver():
    import runpy
    import tempfile
    import yaml
    import wavelet_flow_matching.utils.well_data as wd
    from check_sampling_steps import N_FILES, T, T_IN, TinyWell      # same folder; synthetic 2-file dataset
    from the_well.data.normalization import ZScoreNormalization
    from wavelet_flow_matching.utils.configs import (FlowMatchingConfig, OptimizerConfig, VelocityNetConfig,
                                                     WaveletConfig)
    torch.manual_seed(0)
    fields = ["a", "b"]
    zero, one = {f: 0.0 for f in fields}, {f: 1.0 for f in fields}
    norm = ZScoreNormalization({"mean": zero, "std": one, "mean_delta": zero, "std_delta": one}, fields, [])
    vn = VelocityNetConfig(base_ch=8, n_res=1, n_levels=2, cond_dim=16, num_cond=0,
                           context_in_channels=2, num_context_frames=0)
    m = QuantumWaveletFlowModule(in_channels=2, field_names=fields, data_shape=(32, 16), n_steps_output=1,
                                 scale_weights=[1.0], subband_weights=None,
                                 wavelet_cfg=WaveletConfig(use_wavelet=True, wavelet="haar", n_scales=1),
                                 velocity_net_cfg=vn, flow_matching_cfg=FlowMatchingConfig(num_sampling_steps=2),
                                 optimizer_cfg=OptimizerConfig(), normalizer=norm)
    m.attach_block(QB)
    with torch.no_grad():
        m.block.w_out.weight.normal_()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / "rayleigh_benard" / "data" / "test").mkdir(parents=True)
    _save(m, tmp / "tinyq.ckpt")
    cfg = {"project_root": str(tmp), "well_base_path": str(tmp / "data"), "well_dataset_name": "rayleigh_benard",
           "ckpt_path": str(tmp / "tinyq.ckpt"), "n_steps_input": T_IN, "num_sampling_steps": 2, "n_ensemble": 1,
           "ode_method": "euler", "n_traj_per_file": 1, "transform": [], "physics": {"enabled": False}}
    (tmp / "inf.yaml").write_text(yaml.safe_dump(cfg))

    seen = []
    block_fwd, f_fwd, original_ds = BottleneckBlock.forward, QL.QuantumF.forward, wd.WellDataset

    def counted_block(self, *a, **k):
        seen.append("block")
        return block_fwd(self, *a, **k)

    def counted_f(self, *a, **k):
        seen.append(self.backend)
        return f_fwd(self, *a, **k)

    BottleneckBlock.forward, QL.QuantumF.forward, wd.WellDataset = counted_block, counted_f, TinyWell  # test-only
    argv = sys.argv
    try:
        for extra, backend in (([], "pennylane"), (["--override", "backend=qiskit"], "qiskit")):
            seen.clear()
            sys.argv = ["inference_qwfm.py", "--config", str(tmp / "inf.yaml"), *extra]
            runpy.run_path(str(INFERENCE_COPY), run_name="__main__")
            assert seen.count("block") == N_FILES * (T - T_IN) * 2, (backend, seen.count("block"))
            assert set(seen) - {"block"} == {backend}, (backend, set(seen))
            csvs = sorted(p.name for p in (tmp / "experiments" / "rayleigh_benard" / "results").glob("*_metrics.csv"))
            assert len(csvs) == (1 if backend == "pennylane" else 2), csvs        # the qiskit run never overwrites
            assert backend == "pennylane" or any("_seed0_qiskit_metrics" in c for c in csvs), csvs
    finally:
        sys.argv, BottleneckBlock.forward, QL.QuantumF.forward, wd.WellDataset = argv, block_fwd, f_fwd, original_ds
    return "pennylane and qiskit backends"


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: comparison with the paper tables (quantum_wfm/analysis/compare.py)
# ─────────────────────────────────────────────────────────────────────────────

def _run_compare(q_value):
    """Synthetic metric files for base/C/Q x 3 seeds in a temp root; run compare.py there; return comparison.csv."""
    import shutil
    import subprocess
    import tempfile
    import pandas as pd
    pde = "turbulent_radiative_layer_2D"
    root = Path(tempfile.mkdtemp())
    res = root / "experiments" / pde / "results"
    res.mkdir(parents=True)
    (root / "artifacts" / "tables" / "csv").mkdir(parents=True)
    for name in ("S12_vrmse", "S13_crps", "S14_spectral_bands", "T1_wavelet_selection"):
        shutil.copy(REPO / "artifacts" / "tables" / "csv" / f"{name}.csv", root / "artifacts" / "tables" / "csv")
    metrics = ["vrmse", "crps", "rmse_c_low", "rmse_c_mid", "rmse_c_high"]
    for stem, value in (("base_stem", 0.6), ("c_stem", 0.6), ("q_stem", q_value)):
        for k, seed in enumerate((42, 84, 168)):
            rows = [dict(traj=0, field="density", step=s, seed=seed, **{m: value + 0.001 * k for m in metrics})
                    for s in range(97)]
            rows.append(dict(traj=0, field="_global_", step=0, seed=seed))          # physics row, must be ignored
            pd.DataFrame(rows).to_csv(res / f"{stem}_nsampling20_nens8_seed{seed}_metrics.csv", index=False)
    cmd = [sys.executable, str(REPO / "quantum_wfm" / "analysis" / "compare.py"), "--pde", pde,
           "--paper-model", "WFM_3scale_haar", "--baseline", "BASE=base_stem", "--c", "C=c_stem", "--q", "Q=q_stem"]
    out = subprocess.run(cmd, env={**os.environ, "WAVELET_FLOW_MATCHING_ROOT": str(root)}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    return pd.read_csv(root / "experiments" / pde / "comparison.csv")


def check_compare():
    better = _run_compare(q_value=0.5)
    assert set(better["table"]) == {"S12_vrmse", "S13_crps", "S14_spectral_bands", "T1_wavelet_selection"}
    assert len(better) == 3 + 3 + 3 + 1, len(better)                    # 3 horizons, 3 horizons, 3 bands, 1 average
    assert better["q_advantage"].all(), better[["table", "q_advantage"]]
    assert better["paper_mean"].notna().all() and "baseline_within_2sd_of_paper" in better
    same = _run_compare(q_value=0.6)
    assert not same["q_advantage"].any(), same[["table", "q_advantage"]]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _checks():
    return {k: v for k, v in globals().items() if k.startswith("check_") and callable(v)}


if __name__ == "__main__":
    names = sys.argv[1:] or list(_checks())
    failed = []
    for name in names:
        try:
            note = _checks()[name]()
            print(f"PASS {name}" + (f" ({note})" if note else ""), flush=True)
        except Exception as e:  # noqa: BLE001 - report every failure, keep running the others
            failed.append(name)
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=4)
    print(f"{len(names) - len(failed)}/{len(names)} passed")
    sys.exit(1 if failed else 0)
