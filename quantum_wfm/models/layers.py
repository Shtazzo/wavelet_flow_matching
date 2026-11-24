"""
[ED] f of the Q-WFM block: n_qubits angles -> n_qubits values in [-1, 1].

  QuantumF   : the circuit of circuits.py, backend "pennylane" (training, inference) or "qiskit" (equivalence,
               inference); always computed in float32, on CPU when the input is on MPS or the backend is Qiskit
  ClassicalF : classical control, tanh MLP with about the same number of weights
"""
from __future__ import annotations

import torch
import torch.nn as nn

from quantum_wfm.models.circuits import pennylane_layer, qiskit_layer

BACKENDS = ("pennylane", "qiskit")


def _in_float32(fn, a: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run fn on `device` in float32 with autocast off; return on a's device and dtype."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        return fn(a.to(device, torch.float32)).to(a.device, a.dtype)


class QuantumF(nn.Module):
    """(N, n_qubits) angles -> (N, n_qubits) expectation values <Z_i>, same device and dtype as the input."""

    def __init__(self, n_qubits: int, n_layers: int, backend: str = "pennylane",
                 init_weights: torch.Tensor | None = None):
        super().__init__()
        if n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2 (CNOT ring), got {n_qubits}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.n_qubits, self.n_layers, self.backend = n_qubits, n_layers, backend
        make = pennylane_layer if backend == "pennylane" else qiskit_layer
        self.layer = make(n_qubits, n_layers, init_weights)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # PennyLane default.qubit cannot run on MPS (pennylane/math/single_dispatch.py:738-739 forces cuda): CPU there.
        # TorchLayer moves its weights to the input's device (pennylane/qnn/torch.py:432), keeping gradients.
        # TorchConnector always computes on CPU (qiskit_machine_learning/connectors/torch_connector.py:167).
        cpu = a.device.type == "mps" or self.backend == "qiskit"
        return _in_float32(self.layer, a, torch.device("cpu") if cpu else a.device)

    def get_weights(self) -> torch.Tensor:
        """Detached CPU copy of the circuit weights, shape (n_layers, n_qubits, 3)."""
        w = self.layer.weights if self.backend == "pennylane" else self.layer.weight
        return w.detach().cpu().clone().reshape(self.n_layers, self.n_qubits, 3)


def swap_backend(f: QuantumF, backend: str) -> QuantumF:
    """New QuantumF with the same circuit and weights on another backend."""
    return QuantumF(f.n_qubits, f.n_layers, backend, init_weights=f.get_weights())


class ClassicalF(nn.Sequential):
    """Classical control for QuantumF: Linear(n, k) -> Tanh -> Linear(k, n) -> Tanh, run in float32."""

    def __init__(self, n: int, k: int):
        super().__init__(nn.Linear(n, k), nn.Tanh(), nn.Linear(k, n), nn.Tanh())

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return _in_float32(super().forward, a, a.device)


def classical_width(n_qubits: int, n_layers: int) -> int:
    """Hidden width k with k(2n+1)+n weights in ClassicalF, about the 3·n·L weights of QuantumF."""
    return max(1, round((3 * n_qubits * n_layers - n_qubits) / (2 * n_qubits + 1)))


def make_f(kind: str, n_qubits: int, n_layers: int, backend: str = "pennylane") -> nn.Module:
    if kind == "quantum":
        return QuantumF(n_qubits, n_layers, backend)
    if kind == "classical":
        return ClassicalF(n_qubits, classical_width(n_qubits, n_layers))
    raise ValueError(f"kind must be 'quantum' or 'classical', got {kind!r}")
