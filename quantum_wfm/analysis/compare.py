"""
[ED] Compare a locally reproduced baseline, C-WFM and Q-WFM with the paper tables of one dataset:
S12 (VRMSE per horizon), S13 (CRPS per horizon), S14 (spectral coherence RMSE per band), T1 (band average).

Aggregation is the paper's own: wavelet_flow_matching.analysis.catalog.add_features + the tables.py builders run per
seed by tables.across_seeds (used as they are; nothing is registered in catalog.MODEL_REGISTRY).

    python quantum_wfm/analysis/compare.py --pde turbulent_radiative_layer_2D --paper-model WFM_3scale_haar \
        --baseline WFM_3scale_haar_local=WFM_3scale_haar_turbulent_radiative_layer_2D \
        --c CWFM=CWFM_turbulent_radiative_layer_2D_3scale_haar --q QWFM=QWFM_turbulent_radiative_layer_2D_3scale_haar

Each LABEL=STEM reads experiments/<pde>/results/<STEM>_nsampling*_nens*_seed<S>_metrics.csv (written by
inference_wavelet_fm.py or quantum_wfm/inference_qwfm.py) for S in catalog.SEEDS, and the table is written to
experiments/<pde>/comparison.csv. Lower is better for every metric here.

Criteria (spec DESIGN.md §5.6, proposal): baseline reproduced if |baseline - paper| <= 2 * max(std); Q has an advantage
in a column if it beats every other run by more than 2 * max(std_Q, std_other). std = sample std across seeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wavelet_flow_matching.analysis import catalog as C  # noqa: E402
from wavelet_flow_matching.analysis import tables as T  # noqa: E402
from wavelet_flow_matching.analysis.paths import EXPERIMENTS, TABLES_CSV  # noqa: E402

TABLES = [("S12_vrmse", T.horizon_table, {"metric": "vrmse"}),
          ("S13_crps", T.horizon_table, {"metric": "crps"}),
          ("S14_spectral_bands", T.bands_table, {"time_agg": "horizon"}),
          ("T1_wavelet_selection", T.selection_table, {"time_agg": "horizon"})]


def load_runs(pde: str, runs: dict[str, str]) -> pd.DataFrame:
    """Per-field metric rows of every LABEL -> STEM run, one file per seed, with the paper's step/horizon features."""
    frames = []
    for label, stem in runs.items():
        for seed in C.SEEDS:
            files = sorted((EXPERIMENTS / pde / "results").glob(f"{stem}_nsampling*_nens*_seed{seed}_metrics.csv"))
            if len(files) != 1:
                raise FileNotFoundError(f"{label}: expected 1 metrics file for seed {seed}, found {[f.name for f in files]}")
            df = pd.read_csv(files[0])
            df["model"], df["pde"] = label, pde
            if "seed" not in df.columns or df["seed"].isna().all():
                df["seed"] = seed
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return C.add_features(df[df["field"] != C.GLOBAL_FIELD])


def tables_for(df: pd.DataFrame, labels: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {name: T.across_seeds(builder, df, models=labels, **kw)[:2] for name, builder, kw in TABLES}


def _index(idx) -> pd.MultiIndex:
    """(pde, column) for every table; T1 has one column per pde, labelled 'mean'."""
    return pd.MultiIndex.from_tuples([i if isinstance(i, tuple) else (i, "mean") for i in idx], names=["pde", "column"])


def paper_values(name: str, model: str, pde: str) -> tuple[pd.Series, pd.Series]:
    p = pd.read_csv(TABLES_CSV / f"{name}.csv")
    p = p[(p["model"] == model) & (p["col0"] == pde)]
    keys = list(zip(p["col0"], p["col1"])) if "col1" in p.columns else list(p["col0"])
    return (pd.Series(p["mean"].to_numpy(), index=_index(keys)), pd.Series(p["std"].to_numpy(), index=_index(keys)))


def advantage(mean: pd.DataFrame, std: pd.DataFrame, q: str, others: list[str]) -> pd.Series:
    return pd.concat([(mean.loc[o] - mean.loc[q]) > 2 * np.maximum(std.loc[q], std.loc[o]) for o in others],
                     axis=1).all(axis=1)


def _pair(text: str) -> tuple[str, str]:
    label, stem = text.split("=", 1)
    return label, stem


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description="Q-WFM vs C-WFM vs local baseline vs paper, one dataset")
    p.add_argument("--pde", required=True)
    p.add_argument("--paper-model", required=True, help="model key in artifacts/tables/csv, e.g. WFM_3scale_haar")
    p.add_argument("--baseline", required=True, type=_pair, help="LABEL=STEM of the locally reproduced baseline")
    p.add_argument("--c", type=_pair, help="LABEL=STEM of the classical control C-WFM")
    p.add_argument("--q", type=_pair, help="LABEL=STEM of Q-WFM")
    a = p.parse_args(argv)

    runs = dict(r for r in (a.baseline, a.c, a.q) if r)
    labels, base = list(runs), a.baseline[0]
    parts = []
    for name, (mean, std) in tables_for(load_runs(a.pde, runs), labels).items():
        out = pd.DataFrame({**{f"{m}_mean": mean.loc[m] for m in labels}, **{f"{m}_std": std.loc[m] for m in labels}})
        out.index = _index(out.index)
        out["paper_mean"], out["paper_std"] = paper_values(name, a.paper_model, a.pde)
        out["baseline_within_2sd_of_paper"] = ((out[f"{base}_mean"] - out["paper_mean"]).abs()
                                               <= 2 * np.maximum(out[f"{base}_std"], out["paper_std"]))
        if a.q:
            q_adv = advantage(mean, std, a.q[0], [m for m in labels if m != a.q[0]])
            out["q_advantage"] = q_adv.to_numpy()
        out.insert(0, "table", name)
        parts.append(out)
    result = pd.concat(parts)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(result)
    result.to_csv(EXPERIMENTS / a.pde / "comparison.csv")
    return result


if __name__ == "__main__":
    main()
