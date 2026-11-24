"""
[ED] The variational circuit of the Q-WFM block, written once per backend with the same gates in the same order.

  for layer l = 0..L-1:  RX(a_i) on every qubit                        (data re-uploading)
                         RZ(w[l,i,0]) RY(w[l,i,1]) RZ(w[l,i,2]) on every qubit
                         CNOT ring i -> (i+1) mod n
  read-out: <Z_i>, i = 0..n-1

pennylane_layer : qml.qnn.TorchLayer, weights shape (L, n, 3)
qiskit_layer    : qiskit-machine-learning TorchConnector, flat weights, w[l,i,j] has index (l*n + i)*3 + j
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn


def pennylane_layer(n_qubits: int, n_layers: int, init_weights: torch.Tensor | None = None) -> nn.Module:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)
            for i in range(n_qubits):
                qml.RZ(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # None -> TorchLayer default: uniform in [0, 2*pi) (pennylane/qnn/torch.py:515)
    init = None if init_weights is None else {"weights": init_weights.detach().clone().float()}
    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)}, init_method=init)


def qiskit_layer(n_qubits: int, n_layers: int, init_weights: torch.Tensor | None = None) -> nn.Module:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    x = ParameterVector("x", n_qubits)
    w = ParameterVector("w", n_layers * n_qubits * 3)
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.rx(x[i], i)
        for i in range(n_qubits):
            k = (layer * n_qubits + i) * 3
            qc.rz(w[k], i)
            qc.ry(w[k + 1], i)
            qc.rz(w[k + 2], i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    observables = [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits) for i in range(n_qubits)]
    estimator = StatevectorEstimator()
    # default_precision=0.0: EstimatorQNN defaults to 0.015625 (estimator_qnn.py:113), and StatevectorEstimator then
    # adds Gaussian noise to every expectation value (statevector_estimator.py, rng.normal). Exact values here.
    qnn = EstimatorQNN(circuit=qc, observables=observables, input_params=list(x), weight_params=list(w),
                       estimator=estimator, gradient=ParamShiftEstimatorGradient(estimator), input_gradients=True,
                       default_precision=0.0)
    if init_weights is None:                                    # same default as TorchLayer: uniform in [0, 2*pi)
        init_weights = torch.rand(n_layers, n_qubits, 3) * (2 * math.pi)
    return TorchConnector(qnn, initial_weights=init_weights.detach().flatten().cpu().numpy())
