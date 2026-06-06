"""Decoder shootout: BCG belief-propagation vs DQI-Circuit Gauss-Jordan.

The BP decoder is the source of DQI's gate-count blowup (deep reversible
message passing + ancillas). GJE replaces it with a fixed sequence of CX/SWAP
row operations and no ancillas. Same instance, same IQM CZ basis, head to head.

Run:  uv run python scripts/bp_vs_gje.py
"""

import numpy as np
from qiskit import transpile

from dqi_portfolio import build_dqi_circuit, build_dqi_circuit_gje, gate_stats
from dqi_portfolio.bases import IQM_BASIS

# Sparse (LDPC-like) max-XORSAT instances: m constraints x n variables.
INSTANCES = {
    "3x3": (np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
            np.array([1, 0, 1])),
    "6x4": (np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
                      [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]),
            np.array([0, 1, 1, 0, 1, 0])),
}


def _stats_iqm(qc):
    return gate_stats(transpile(qc, basis_gates=IQM_BASIS, optimization_level=3))


def main():
    print(f"{'instance':>8}  {'decoder':>7} | {'qubits':>6} {'2q(CZ)':>7} {'depth':>6} {'tot':>6}")
    print("-" * 52)
    for name, (B, v) in INSTANCES.items():
        bp = build_dqi_circuit(B, v, ell=2, bp_iters=1)
        gje = build_dqi_circuit_gje(B, v, ell=2)
        for tag, qc in (("BP", bp), ("GJE", gje)):
            s = _stats_iqm(qc)
            print(f"{name:>8}  {tag:>7} | {s['qubits']:>6} {s['2q_gates']:>7} "
                  f"{s['depth']:>6} {s['gates']:>6}")
        # reduction line
        b, g = _stats_iqm(bp), _stats_iqm(gje)
        dq = 100 * (b["2q_gates"] - g["2q_gates"]) / b["2q_gates"]
        dd = 100 * (b["depth"] - g["depth"]) / b["depth"]
        print(f"{name:>8}  {'Δ':>7} | "
              f"{b['qubits'] - g['qubits']:>+6} {-dq:>+6.0f}% {-dd:>+5.0f}% (2q / depth)")
        print("-" * 52)


if __name__ == "__main__":
    main()
