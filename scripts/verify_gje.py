"""Sanity-check the GJE-decoder DQI circuit: does it concentrate on good solutions?

Simulate, read the solution off the syndrome register, and compare the satisfied-
constraint count of DQI's sampled strings against uniformly random strings.

Run:  uv run python scripts/verify_gje.py
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from dqi_portfolio import build_dqi_circuit_gje
from dqi_portfolio.bases import SIM_BASIS
from dqi_portfolio.readout import score_counts

B = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
              [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
v = np.array([0, 1, 1, 0, 1, 0])
m, n = B.shape


def satisfied(x):
    """Number of constraints (B x == v) mod 2 met by bitstring x (len n)."""
    return int(np.sum((B @ x) % 2 == v))


def brute_force_max():
    best = 0
    for k in range(2 ** n):
        x = np.array([(k >> i) & 1 for i in range(n)])
        best = max(best, satisfied(x))
    return best


def main():
    qc = build_dqi_circuit_gje(B, v, ell=2, with_measurements=True)
    sim = AerSimulator()
    qc = transpile(qc, sim, basis_gates=SIM_BASIS)
    counts = sim.run(qc, shots=8192).result().get_counts()

    # measure_all packs all qubits; syndrome register is qubits [m, m+n).
    # Shared readout reverses each bitstring and reads x off offset m.
    r = score_counts(counts, B, v, x_offset=m)
    sat_hist, total = r["hist"], r["total"]

    opt = brute_force_max()
    rand_expected = 0.5 * m + 0.5 * 0  # each constraint met w.p. 1/2 for random x
    print(f"instance: {m} constraints x {n} variables, v={v.tolist()}")
    print(f"brute-force max satisfiable : {opt}/{m}")
    print(f"random-string expectation   : {rand_expected:.2f}/{m}")
    print(f"DQI-GJE mean satisfied      : {r['mean']:.2f}/{m}")
    print("\nDQI-GJE satisfied-count distribution (sat: share):")
    for s in sorted(sat_hist, reverse=True):
        print(f"  {s}/{m}: {100 * sat_hist[s] / total:5.1f}%")


if __name__ == "__main__":
    main()
