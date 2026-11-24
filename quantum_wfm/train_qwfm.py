"""
[ED] Fine-tune a quantum (Q-WFM) or classical-control (C-WFM) bottleneck block on top of a pretrained WFM checkpoint.

Quantum version of wavelet_flow_matching/train_wavelet_fm.py (spec: quantum_wfm/DESIGN.md §5.4). The config is the
paper's TOML for the dataset (`base_config`) overridden by the Q TOML, so nothing here depends on the dataset.

    conda run -n wfm python quantum_wfm/train_qwfm.py -c quantum_wfm/configs/qwfm_trl_3scale_haar.toml [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import munch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import toml  # noqa: E402
import torch  # noqa: E402

from quantum_wfm.modules.qwfm import QuantumWaveletFlowModule  # noqa: E402
from wavelet_flow_matching.utils.augmentations import compose_augmentation, compose_transformation  # noqa: E402
from wavelet_flow_matching.utils.configs import OptimizerConfig  # noqa: E402
from wavelet_flow_matching.utils.experiment import ExperimentManager  # noqa: E402
from wavelet_flow_matching.utils.well_data import WellDataModuleLightning  # noqa: E402

PATH_KEYS = (("path", "well_base_path"), ("path", "stats_path"), ("path", "experiments_path"),
             ("warm_start", "ckpt_path"))


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> munch.Munch:
    """Q TOML merged over its base_config TOML (Q wins); relative paths resolved against the repo root."""
    q = toml.load(path)
    cfg = _merge(toml.load(REPO / q["base_config"]), q)
    for section, key in PATH_KEYS:
        value = cfg.get(section, {}).get(key)
        if value:
            cfg[section][key] = str(REPO / value)          # an absolute value stays as it is
    return munch.munchify(cfg)


def resolve_checkpoint(cfg) -> str:
    """warm_start.ckpt_path, else the Hugging Face checkpoint warm_start.model (default: the paper's 3-scale WFM)."""
    ws = cfg.get("warm_start", {})
    if ws.get("ckpt_path"):
        return ws.ckpt_path
    from wavelet_flow_matching.analysis import catalog
    from wavelet_flow_matching.hub import download_checkpoint
    pde = cfg.path.well_dataset_name
    model = ws.get("model") or catalog.wfm_key("3scale", catalog.BEST_WFM_PER_PDE[pde]["3scale"])
    return download_checkpoint(pde, model)


def finetune_optimizer(ft) -> OptimizerConfig:
    """Optimizer for the fine-tuning, never the checkpoint's (its warmup_epochs would keep a short run in warmup)."""
    return OptimizerConfig(cls="AdamW", lr=float(ft.lr), weight_decay=float(ft.weight_decay), betas=(0.9, 0.999),
                           warmup_epochs=int(ft.warmup_epochs), scheduler_cls="CosineAnnealingLR",
                           eta_min=float(ft.eta_min))


def loader_workers(requested: int, start_method: str) -> int:
    """DataLoader workers to use: 0 unless worker processes are forked. The original DataLoader's collate_fn is a
    lambda (wavelet_flow_matching/utils/well_data.py:132-134) and 'spawn' (macOS default) cannot pickle it."""
    return requested if start_method == "fork" else 0


def check_normalizer(a, b, atol: float = 1e-5) -> None:
    """ValueError if two ZScoreNormalization objects differ (checkpoint vs dataset stats.yaml)."""
    for name in ("means", "stds", "means_delta", "stds_delta"):
        da, db = getattr(a, name), getattr(b, name)
        if da.keys() != db.keys() or any(
                (torch.as_tensor(da[k]) - torch.as_tensor(db[k])).abs().max() > atol for k in da):
            raise ValueError(f"normalizer '{name}' differs between the checkpoint and the dataset stats")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Q-WFM / C-WFM fine-tuning of a bottleneck block")
    p.add_argument("-c", "--config", required=True, help="Q TOML (quantum_wfm/configs/)")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 2 train batches, 1 val batch; print step times")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    qblock = dict(cfg.quantum)
    if qblock.get("backend") == "qiskit":
        raise ValueError("backend 'qiskit' is for equivalence checks and inference, not for training "
                         "(parameter-shift gradients, DESIGN.md §3/§5.5): use 'pennylane'")

    pl.seed_everything(cfg.random.seed)
    lt = cfg.lightning.trainer
    torch.set_float32_matmul_precision(lt.matmul_precision)

    model = QuantumWaveletFlowModule.from_checkpoint(resolve_checkpoint(cfg))
    if model.block is not None:
        raise ValueError("expected a classical WFM checkpoint to warm-start from (resuming is not supported)")
    model.attach_block(qblock)
    model.freeze_backbone()
    model.optimizer_cfg = finetune_optimizer(cfg.finetune)
    model.hparams["optimizer_cfg"] = model.optimizer_cfg.to_dict()

    tr = cfg.train                                          # data settings of the paper's run (base_config)
    workers = loader_workers(cfg.lightning.datamodules.data_workers, multiprocessing.get_start_method())
    if workers != cfg.lightning.datamodules.data_workers:
        print(f"[train_qwfm] DataLoader workers {cfg.lightning.datamodules.data_workers} -> {workers}: "
              f"start method '{multiprocessing.get_start_method()}' cannot pickle the original collate_fn")
    data = WellDataModuleLightning(
        well_base_path=cfg.path.well_base_path, well_dataset=cfg.path.well_dataset_name,
        stats_path=cfg.path.stats_path, n_steps_input=tr.n_steps_input, n_steps_output=tr.n_steps_output,
        max_rollout_steps=tr.max_rollout_steps, batch_size=tr.batch_size,
        num_workers=workers, train_shuffle=tr.train_shuffle,
        val_shuffle=tr.val_shuffle, test_shuffle=tr.test_shuffle,
        target_dtype=getattr(torch, tr.target_dtype.split(".")[-1]),
        augmentation=compose_augmentation(*tr.augmentations) if tr.augmentations else None,
        transformation=compose_transformation(*tr.transformations) if tr.transformations else None,
    )
    data.setup()
    check_normalizer(model.normalizer, data.normalizer)

    # --smoke writes to <exp_name>_smoke, so its checkpoints are never picked as the real run's best one
    exp_name = cfg.main.exp_name + ("_smoke" if args.smoke else "")
    em = ExperimentManager(base_dir=cfg.path.experiments_path, experiment_name=exp_name, overwrite=False,
                           loggers=cfg.loggers.loggers, callbacks=cfg.callbacks.callbacks,
                           log_every_n_epochs=cfg.loggers.log_every_n_epochs)
    model.experiment_manager = em
    em.copy_config_file(src=args.config)
    em.save_config(munch.unmunchify(cfg), name="config_merged.json")

    limits = dict(max_epochs=1, limit_train_batches=2, limit_val_batches=1) if args.smoke else dict(
        max_epochs=cfg.finetune.epochs, limit_train_batches=lt.limit_train_batches,
        limit_val_batches=lt.limit_val_batches)
    trainer = pl.Trainer(
        accelerator=lt.accelerator, strategy=lt.strategy, devices=1, precision=lt.precision,
        callbacks=em.get_callbacks(), logger=False, num_sanity_val_steps=0,
        gradient_clip_val=lt.gradient_clip_val if isinstance(lt.gradient_clip_val, float) else None,
        log_every_n_steps=lt.log_every_n_steps, **limits,
    )
    trainer._logger_connector.configure_logger(em.get_loggers())   # as train_wavelet_fm.py:165-168
    trainer.fit(model, datamodule=data)

    if args.smoke:
        steps = Path(em.dirs["performance"]) / "step_times.csv"     # PerformanceTrackingCallback (no device sync)
        rows = list(csv.reader(steps.open()))[1:] if steps.exists() else []
        print("[smoke] step times (s):", [round(float(r[1]), 2) for r in rows], "| checkpoints:",
              sorted(p.name for p in Path(em.dirs["checkpoints"]).glob("*_portable.ckpt")))


if __name__ == "__main__":
    main()
