"""Throwaway generator: build 7-bond DQI immunization circuit, validate readout
on AerSimulator, compute baselines, export portable OpenQASM3 + meta sidecar.

Run: uv run python artifacts/iqm/circuit/gen_circuit.py
"""
import json
import warnings
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
import qiskit.qasm3 as q3

from dqi_portfolio import (
    build_immunization_instance,
    build_dqi_circuit_algebraic,
    dqi_satisfaction_stats,
)

warnings.filterwarnings("ignore")

HERE = "/Users/joni/personal/hackathons/quantumhack/artifacts/iqm/circuit"
SIM_BASIS = ["cx", "rz", "rx", "ry", "h", "x"]
# device-agnostic standard basis for portable QASM3 (NOT iqm cz/r)
STD_BASIS = ["cx", "rz", "sx", "x", "h", "rx", "ry"]


def readout_counts(counts, B, v, m, n):
    """EXACT prototype readout (immunization_prototype.simulate, verbatim logic)."""
    tot = 0
    sat_hist = {}
    wt = 0
    for bs, c in counts.items():
        bits = bs.replace(" ", "")[::-1]
        x = np.array([int(bits[m + i]) for i in range(n)])
        s = int(np.sum((B @ x) % 2 == v))
        sat_hist[s] = sat_hist.get(s, 0) + c
        tot += c
        wt += s * c
    mean = wt / tot
    p_opt = sat_hist.get(7, 0) / tot  # filled by caller-known opt
    return mean, sat_hist, tot


def main():
    # --- build instance ---
    chk = None
    B, v, code, ladder = build_immunization_instance(m_field=3)
    m, n = B.shape
    print(f"instance: B shape {B.shape} (m={m} constraints, n={n} vars)")
    print(f"v = {v.tolist()}")

    # --- build measured circuit (ell=1, ancilla v-chain) ---
    qc = build_dqi_circuit_algebraic(
        code, B, v, ell=1, use_ancilla=True, mcx_mode="v-chain",
        with_measurements=True)
    print(f"circuit qubits = {qc.num_qubits}")
    # IQM CZ check
    iqm = transpile(qc, basis_gates=["cz", "rz", "rx", "ry"], optimization_level=3)
    n_cz = iqm.count_ops().get("cz", 0)
    print(f"IQM-CZ count = {n_cz}  (expect ~86)")

    # --- classical ground truth + brute force ---
    st = dqi_satisfaction_stats(B, v, code, ell=1)
    opt = st["max_satisfied"]
    print(f"brute-force optimum = {opt}/{m}")

    # brute force over all 2^n assignments
    n_optimal = 0
    rand_sat_sum = 0
    for k in range(1 << n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        s = int(np.sum((B @ x) % 2 == v))
        rand_sat_sum += s
        if s == opt:
            n_optimal += 1
    random_mean = rand_sat_sum / (1 << n)
    random_p_opt = n_optimal / (1 << n)
    print(f"#optimal assignments = {n_optimal}/{1<<n}")
    print(f"random mean satisfied = {random_mean:.4f}/{m}")
    print(f"random P(opt) = {random_p_opt:.4f}")

    # --- VALIDATION SELF-CHECK on AerSimulator ---
    sim = AerSimulator()
    qct = transpile(qc, sim, basis_gates=SIM_BASIS)
    counts = sim.run(qct, shots=16384).result().get_counts()
    mean_sim, sat_hist, tot = readout_counts(counts, B, v, m, n)
    p_opt_sim = sat_hist.get(opt, 0) / tot
    print("\n=== SELF-CHECK (AerSimulator) ===")
    print(f"  mean satisfied = {mean_sim:.4f}/{m}  (target ~6.34)")
    print(f"  P(opt={opt}/{m}) = {p_opt_sim:.4f}  (target ~0.836)")
    print(f"  sat histogram = { {k: round(sat_hist[k]/tot,4) for k in sorted(sat_hist)} }")
    assert abs(mean_sim - 6.34) < 0.05, f"mean off: {mean_sim}"
    assert abs(p_opt_sim - 0.836) < 0.03, f"P(opt) off: {p_opt_sim}"
    print("  SELF-CHECK PASSED")

    # --- export portable QASM3 (kept for reference) AND QASM2 (shipped) ---
    qc_std = transpile(qc, basis_gates=STD_BASIS, optimization_level=1)
    # Relabel registers to names that DON'T collide with qelib1.inc gate names
    # (orig "y" register clashes with the Pauli-Y gate in QASM2 parsing; "x" too).
    # Build a fresh circuit preserving qubit ORDER (er=y[0..6], sol=x[7..9],
    # an=anc[10]) so the readout offsets [m:m+n] are unchanged.
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    er = QuantumRegister(7, "er")
    sol = QuantumRegister(3, "sol")
    an = QuantumRegister(1, "an")
    cl = ClassicalRegister(11, "c")
    relabeled = QuantumCircuit(er, sol, an, cl)
    relabeled.compose(qc_std, qubits=list(range(11)),
                      clbits=list(range(11)), inplace=True)
    qc_std = relabeled
    qasm3_path = f"{HERE}/q50_immunization_7bond.qasm3"
    with open(qasm3_path, "w") as f:
        q3.dump(qc_std, f)
    print(f"\nwrote {qasm3_path}")

    # QASM2 is what ships to LUMI: its qiskit lacks qiskit_qasm3_import (q3.load
    # fails) but QuantumCircuit.from_qasm_file / qasm2.load are always available.
    # Circuit is pure standard gates (cx,h,x,ry,rz,barrier) so QASM2 is exact.
    import qiskit.qasm2 as q2
    qasm_path = f"{HERE}/q50_immunization_7bond.qasm2"
    q2.dump(qc_std, qasm_path)
    print(f"wrote {qasm_path}")

    # --- verify QASM2 reload + re-sim to same P(opt) ---
    qc_re = q2.load(qasm_path)
    qct2 = transpile(qc_re, sim, basis_gates=SIM_BASIS)
    counts2 = sim.run(qct2, shots=16384).result().get_counts()
    mean_re, hist_re, tot2 = readout_counts(counts2, B, v, m, n)
    p_opt_re = hist_re.get(opt, 0) / tot2
    print(f"RELOADED QASM3 re-sim: mean={mean_re:.4f}, P(opt)={p_opt_re:.4f}")
    assert abs(p_opt_re - 0.836) < 0.03, f"reloaded P(opt) off: {p_opt_re}"
    print("RELOAD CHECK PASSED — shipped file is faithful")

    # --- meta sidecar ---
    meta = {
        "instance": "7-bond fixed-income immunization (m_field=3, t=1, ell=1)",
        "B": B.tolist(),
        "v": v.tolist(),
        "n": int(n),
        "m": int(m),
        "num_qubits": int(qc.num_qubits),
        "iqm_cz_local_estimate": int(n_cz),
        "optimum": int(opt),
        "n_optimal_assignments": int(n_optimal),
        "n_assignments": int(1 << n),
        "random_mean_satisfied": float(random_mean),
        "random_p_opt": float(random_p_opt),
        "simulated_mean_satisfied": float(mean_sim),
        "simulated_p_opt": float(p_opt_sim),
        "simulated_sat_histogram": {str(k): int(sat_hist[k]) for k in sat_hist},
        "sim_shots": int(tot),
        "bond_ladder": {
            "maturities": [float(x) for x in ladder.maturities],
            "locators": [int(x) for x in ladder.locators],
        },
        "readout": {
            "description": (
                "Measured bitstring bs from measure_all(). Strip spaces, REVERSE "
                "(bs[::-1]) to get little-endian bit array 'bits'. The solution "
                "register x (n=3 vars) is bits[m : m+n] where m=7. Register layout "
                "is y(7) + x(3) + anc(1) = 11 qubits; measure_all appends classical "
                "bits in qubit order, and after reversal x occupies indices 7,8,9. "
                "satisfied(x) = sum((B @ x) % 2 == v)."),
            "m_offset": int(m),
            "n_vars": int(n),
            "bit_order": "reverse measured string, then index [m : m+n]",
        },
    }
    meta_path = f"{HERE}/q50_immunization_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
