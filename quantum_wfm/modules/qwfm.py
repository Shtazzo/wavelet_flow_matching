"""
[ED] QuantumWaveletFlowModule: WaveletFlowModule (wavelet_flow_matching/modules/wavelet_fm.py, used as is) plus an
optional BottleneckBlock appended to net.bottleneck.
"""
from __future__ import annotations

import io

import torch

from quantum_wfm.models.bottleneck import BottleneckBlock
from quantum_wfm.models.layers import QuantumF, make_f, swap_backend
from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule


class QuantumWaveletFlowModule(WaveletFlowModule):
    """Without a block it behaves exactly like WaveletFlowModule. The block configuration is stored in
    hparams["qblock"] (kind, n_qubits, n_layers, backend, index), so checkpoints rebuild it.
    """

    @property
    def block(self) -> BottleneckBlock | None:
        last = self.net.bottleneck[-1]
        return last if isinstance(last, BottleneckBlock) else None

    def attach_block(self, qblock: dict) -> None:
        if self.block is not None:
            raise ValueError("a bottleneck block is already attached")
        cfg = {k: qblock[k] for k in ("kind", "n_qubits", "n_layers", "backend")}
        with torch.random.fork_rng(devices=[]):  # f's init uses a forked RNG: Q and C get the same w_in/w_c/w_out
            f = make_f(cfg["kind"], cfg["n_qubits"], cfg["n_layers"], cfg["backend"])  # and the same RNG stream after
        block = BottleneckBlock(self.net.chs[-1], self.net._cond_dim, f, cfg["n_qubits"])
        self.net.bottleneck.append(block.to(self.device))
        self.hparams["qblock"] = {**cfg, "index": len(self.net.bottleneck) - 1}

    def freeze_backbone(self) -> None:
        """Train only the block: every original parameter gets requires_grad=False."""
        if self.block is None:
            raise ValueError("attach_block() first")
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.block.parameters():
            p.requires_grad_(True)

    def configure_optimizers(self):
        """warmup_epochs == 0: optimizer + CosineAnnealingLR from lr, stepped per epoch. The original
        (wavelet_fm.py:601-621, used when warmup_epochs > 0) steps its LinearLR warmup per epoch from 1e-3 * lr,
        so warmup_epochs=1 runs epoch 0 at 1e-3 * lr and warmup_epochs=0 would keep the whole run there."""
        oc = self.optimizer_cfg
        if not oc.scheduler_cls or oc.warmup_epochs > 0:
            return super().configure_optimizers()
        opt = getattr(torch.optim, oc.cls)(self.parameters(), **oc.optimizer_kwargs())
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.trainer.max_epochs, 1),
                                                           eta_min=oc.eta_min)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}

    def set_backend(self, backend: str) -> None:
        """Run the block's circuit on another backend, same weights (quantum blocks only)."""
        if self.block is None or not isinstance(self.block.f, QuantumF):
            raise ValueError("set_backend() needs an attached quantum block")
        self.block.f = swap_backend(self.block.f, backend).to(self.device)
        self.hparams["qblock"] = {**self.hparams["qblock"], "backend": backend}

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", stats: dict = None,
                        backend: str | None = None) -> "QuantumWaveletFlowModule":
        """Load a classical WFM checkpoint (no block) or a Q-WFM/C-WFM one (block rebuilt from hparams["qblock"]).

        The classical part is loaded by WaveletFlowModule.from_checkpoint itself, from an in-memory copy of the
        checkpoint without the block's keys; then the block is attached and the full state dict is loaded strictly.
        """
        ckpt = torch.load(path, map_location=map_location)
        q = ckpt["hyper_parameters"].get("qblock")
        if q is None:
            return super().from_checkpoint(path, map_location=map_location, stats=stats)
        prefix = f"net.bottleneck.{q['index']}."
        buf = io.BytesIO()
        torch.save({**ckpt, "state_dict": {k: v for k, v in ckpt["state_dict"].items() if not k.startswith(prefix)}}, buf)
        buf.seek(0)
        model = super().from_checkpoint(buf, map_location=map_location, stats=stats)
        model.attach_block(q)
        model.load_state_dict(ckpt["state_dict"])
        if backend is not None and backend != q["backend"]:
            model.set_backend(backend)
        return model
