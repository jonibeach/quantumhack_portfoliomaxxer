"""Read-only Q50 access check + on-device routed gate count.

DOES NOT submit any quantum job. Just:
  1. builds the m=3 immunization (= 7-bond / BCH) DQI circuit
  2. transpiles it against the REAL Q50 coupling map exposed by
     iqm.qiskit_iqm.fake_backends.IQMFakeAphrodite (Aphrodite == Q50 chip,
     54 lattice sites, 90 CZ edges; QB32 is excluded on the physical device)
  3. reports CZ count, depth, qubit footprint after routing.

Run locally (requires `pip install qiskit-iqm`) or on LUMI after
    module use /appl/local/quantum/modulefiles
    module load Local-quantum fiqci-vtt-qiskit-JQH/1.0
"""

import warnings

from qiskit import transpile

from dqi_portfolio.bases import IQM_BASIS
from dqi_portfolio.dqi_algebraic import (
    build_bch_instance,
    build_dqi_circuit_algebraic,
)
from dqi_portfolio.metrics import gate_stats
from dqi_portfolio.immunization import build_immunization_instance

warnings.filterwarnings("ignore")


def main():
    # build the m=3, 7-bond instance (the existing "86 raw CZ / 158 routed
    # generic square lattice" baseline from immunization_prototype.py).
    m = 3
    B, v, code, ladder = build_immunization_instance(m_field=m, seed=0)

    qc = build_dqi_circuit_algebraic(
        code, B, v, ell=1
    )
    raw = gate_stats(qc)
    print(f"Raw circuit (logical, all-to-all):")
    print(f"  qubits {raw['qubits']}  2q {raw['2q_gates']}  depth {raw['depth']}")

    # baseline: IQM native basis but no coupling map (= what the project
    # currently reports as "IQM 86 CZ").
    tq = transpile(qc, basis_gates=IQM_BASIS, optimization_level=3)
    s = gate_stats(tq)
    print(f"\nIQM basis, no coupling map (all-to-all):")
    print(f"  qubits {s['qubits']}  CZ {s['2q_gates']}  depth {s['depth']}")

    # routed against the real Q50 coupling map.
    try:
        from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
    except ImportError:
        print("\nqiskit-iqm not installed - skip on-device routing.")
        print("On LUMI: module load fiqci-vtt-qiskit-JQH/1.0")
        return

    backend = IQMFakeAphrodite()
    print(f"\nIQMFakeAphrodite (Q50 chip family):")
    print(f"  num_qubits = {backend.num_qubits}  ops = {backend.operation_names}")
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    print(f"  undirected CZ edges = {len(edges)}  (real device has QB32 excluded "
          f"and QB10-QB18 coupler broken)")

    # 3 transpile seeds to get a feel for routing variance.
    print("\nTranspiled against Q50 coupling map (optimization_level=3):")
    for seed in (0, 1, 2):
        tq2 = transpile(
            qc,
            backend=backend,
            optimization_level=3,
            seed_transpiler=seed,
        )
        s2 = gate_stats(tq2)
        print(f"  seed={seed}: qubits {s2['qubits']}  CZ {s2['2q_gates']}  "
              f"depth {s2['depth']}")


if __name__ == "__main__":
    main()
