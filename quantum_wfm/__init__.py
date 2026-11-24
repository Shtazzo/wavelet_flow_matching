"""
[ED] quantum_wfm — Quantum Wavelet Flow Matching, built on wavelet_flow_matching/ (used, never modified).

Layout (mirrors wavelet_flow_matching/):
  models/      circuits.py (variational circuit, PennyLane + Qiskit), layers.py (f: quantum / classical),
               bottleneck.py (residual block for the U-Net bottleneck)
  modules/     qwfm.py (QuantumWaveletFlowModule, Lightning module)
  analysis/    compare.py (comparison with the paper tables)
  train_qwfm.py, inference_qwfm.py   entry points
  configs/     TOML configs per dataset            checks/   runnable checks

Spec: quantum_wfm/DESIGN.md · plan: quantum_wfm/PLAN.md
"""
