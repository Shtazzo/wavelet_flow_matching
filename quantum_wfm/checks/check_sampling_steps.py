#!/usr/bin/env python3
"""
[ED] Check: the inference YAML `num_sampling_steps` sets the ODE steps actually run.

Runs inference_wavelet_fm.py main() end-to-end on a tiny synthetic checkpoint
(saved with num_sampling_steps=3) and a fake 2-file dataset, counting the
velocity-net calls. No real data or checkpoint needed.

    conda run -n wfm python quantum_wfm/checks/check_sampling_steps.py
"""
import os
import runpy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402
from the_well.data.normalization import ZScoreNormalization  # noqa: E402

import wavelet_flow_matching.utils.well_data as wd  # noqa: E402
from wavelet_flow_matching.models.wavelet_unet import WaveletFlowUNet  # noqa: E402
from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule  # noqa: E402
from wavelet_flow_matching.utils.configs import (  # noqa: E402
    FlowMatchingConfig, OptimizerConfig, VelocityNetConfig, WaveletConfig)

CKPT_STEPS = 3
N_FILES, T, T_IN = 2, 8, 4
N_ROLLOUT_CALLS = N_FILES * (T - T_IN)          # T_out = 1 -> one call per predicted frame


class TinyWell(torch.utils.data.Dataset):       # 2 files x 1 trajectory of T steps
    def __init__(self, *a, **k):
        self.n_trajectories_per_file = [1] * N_FILES
        self.metadata = SimpleNamespace(n_steps_per_trajectory=[T] * N_FILES,
                                        constant_scalar_names=[], field_names={0: ["a", "b"]})

    def __len__(self):
        return N_FILES

    def __getitem__(self, i):
        return {"input_fields": torch.randn(T, 32, 16, 2) + 3.0}


def run(yaml_steps):
    """Run inference main() with the given YAML num_sampling_steps; return (ODE steps run, output files)."""
    torch.manual_seed(0)
    fields = ["a", "b"]
    zero, one = {f: 0.0 for f in fields}, {f: 1.0 for f in fields}
    norm = ZScoreNormalization({"mean": zero, "std": one, "mean_delta": zero, "std_delta": one}, fields, [])
    vn = VelocityNetConfig(base_ch=8, n_res=1, n_levels=2, cond_dim=16, num_cond=0,
                           context_in_channels=2, num_context_frames=0)
    m = WaveletFlowModule(in_channels=2, field_names=fields, data_shape=(32, 16), n_steps_output=1,
                          scale_weights=[1.0], subband_weights=None,
                          wavelet_cfg=WaveletConfig(use_wavelet=True, wavelet="haar", n_scales=1),
                          velocity_net_cfg=vn, flow_matching_cfg=FlowMatchingConfig(num_sampling_steps=CKPT_STEPS),
                          optimizer_cfg=OptimizerConfig(), normalizer=norm)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / "rayleigh_benard" / "data" / "test").mkdir(parents=True)
    ckpt = tmp / "tiny.ckpt"
    torch.save({"state_dict": m.state_dict(), "hyper_parameters": dict(m.hparams)}, ckpt)
    cfg = {"project_root": str(tmp), "well_base_path": str(tmp / "data"), "well_dataset_name": "rayleigh_benard",
           "ckpt_path": str(ckpt), "n_steps_input": T_IN, "num_sampling_steps": yaml_steps, "n_ensemble": 1,
           "ode_method": "euler", "n_traj_per_file": 1, "transform": [], "physics": {"enabled": False}}
    (tmp / "inf.yaml").write_text(yaml.safe_dump(cfg))

    calls = {"n": 0}
    original_fwd, original_ds = WaveletFlowUNet.forward, wd.WellDataset

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original_fwd(self, *args, **kwargs)

    WaveletFlowUNet.forward, wd.WellDataset = counted, TinyWell
    argv, sys.argv = sys.argv, ["inference_wavelet_fm.py", "--config", str(tmp / "inf.yaml")]
    try:
        runpy.run_path(str(REPO / "wavelet_flow_matching" / "inference_wavelet_fm.py"), run_name="__main__")
    finally:
        sys.argv, WaveletFlowUNet.forward, wd.WellDataset = argv, original_fwd, original_ds
    files = sorted(p.name for p in (tmp / "experiments" / "rayleigh_benard" / "results").glob("*"))
    return calls["n"] // N_ROLLOUT_CALLS, files


if __name__ == "__main__":
    results = {}
    for yaml_steps in (CKPT_STEPS, 7):          # = checkpoint (original behaviour) and != checkpoint
        steps, files = run(yaml_steps)
        results[yaml_steps] = steps
        print(f"checkpoint={CKPT_STEPS}  YAML={yaml_steps}  ->  ODE steps run={steps}  files={files}")
    ok = all(results[k] == k for k in results)
    print("PASS" if ok else "FAIL", {f"YAML {k}": f"{v} steps" for k, v in results.items()})
    sys.exit(0 if ok else 1)
