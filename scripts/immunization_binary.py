"""Hardware-optimized BINARY RE-ENCODING of the 7-bond immunization DQI circuit.

THE COLLAPSE (why 11 qubits become 3)
-------------------------------------
The existing circuit (build_dqi_circuit_algebraic, driven by
scripts/immunization_prototype.py) solves the 7-bond fixed-income immunization
instance, which is exactly the BCH(7,4) / Hamming GF(2^3), t=1 single-error
code. It carries a 7-qubit ONE-HOT error register ``y`` + a 3-qubit syndrome
register ``x`` + 1 ancilla = 11 qubits, and a reversible Hamming decode/uncompute
(MCX-heavy). On the IQM Q50 it routed to 154 CZ at depth 226 and DECOHERED to
random (P(opt) 0.122 vs random 0.125), because the coherence wall is ~40 routed
CZ.

But for t=1 over GF(2) the single-error location -> syndrome map is a BIJECTION
on {0..7}: error at position p gives syndrome s_p = B[p] (the p-th row of B, as an
integer), and for this instance {s_0..s_6} = {1,2,4,3,6,7,5} = a permutation of
{1..7}. The no-error term is syndrome 0. So the one-hot error register, the
syndrome computation, AND the reversible decode/uncompute all FOLD into the
3-qubit syndrome register: there is nothing left to decode because preparing the
state directly in the syndrome basis already applies the location->syndrome
relabel. The whole DQI computation reduces to:

    prepare a specific 3-qubit amplitude vector  ->  H^{(x)3}  ->  measure.

No y register, no ancilla, no MCX. The DQI prescription (optimal weights, the
syndrome map, and the v-phase signs) is computed PROGRAMMATICALLY below — it is
not the planted answer hardcoded. After H^{(x)3} the marginal over the 3 qubits
is the SAME distribution the 11-qubit circuit produces (P(x=3)=0.8307, optimum
7/7), now in 3 qubits and well under the 40-CZ coherence wall.

Run:  uv run python scripts/immunization_binary.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

from dqi_portfolio import (
    immunization_is_bch_instance,
    dqi_satisfaction_stats,
    decode_portfolio,
    gate_stats,
)
from dqi_portfolio.dqi_algebraic import _optimal_poly_weights
from dqi_portfolio.bases import IQM_BASIS
from dqi_portfolio.report import section
from scripts._immun_circuit import (
    build_amplitude_vector,
    v_phase_positions,
    build_structured,
    build_minimal,
    statevector_marginal,
    shots_score,
    routed_cz,
)

warnings.filterwarnings("ignore")

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "artifacts" / "iqm" / "binary"
)

# Old 11-qubit baseline (from artifacts/iqm meta + q50 results), for the table.
BASELINE = {
    "qubits": 11,
    "logical_cz": 86,
    "routed_cz": 154,
    "depth": 226,
}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    section("BINARY RE-ENCODING of 7-bond immunization DQI  (t=1 bijection collapse)")

    # --- 1. instance + DQI prescription (programmatic, not hardcoded) ---
    chk = immunization_is_bch_instance(3, 1, 0)
    B, v, code, ladder = chk["B"], chk["v"], chk["code"], chk["ladder"]
    m, n_syn = B.shape  # 7 x 3
    assert chk["B_equals_bch"], "B is not the BCH instance!"
    w = _optimal_poly_weights(m, 1)  # ell=1 optimal weights (uniform 1/sqrt2)
    a, syndromes = build_amplitude_vector(B, v, w)
    flips = v_phase_positions(B, v)

    print(f"  instance: B {B.shape}  v={v.tolist()}  (B==BCH: {chk['B_equals_bch']})")
    print(f"  optimal ell=1 weights w = [{w[0]:.6f}, {w[1]:.6f}] (uniform 1/sqrt2)")
    print(f"  single-error syndrome map  p -> s_p = {syndromes}  "
          f"(permutation of 1..7: {sorted(syndromes) == list(range(1, 8))})")
    print(f"  v-phase oracle flips |s> for s in {flips}  (= {{s_p : v_p=1}})")
    print(f"  pre-H amplitude vector a (index = syndrome int):")
    print(f"    a = (1/sqrt14) * "
          f"{np.round(a * np.sqrt(14), 4).tolist()}")
    print(f"    |a|^2 sum = {np.sum(a**2):.6f} (normalized)")

    # --- 2. build both variants ---
    qc_struct = build_structured(a, flips, n_syn)
    qc_min = build_minimal(a, n_syn)

    # --- 3. verify statevector marginals match ground truth ---
    section("VERIFY: statevector marginal P(x) over 3 qubits (both variants)")
    p_struct = statevector_marginal(qc_struct)
    p_min = statevector_marginal(qc_min)
    # ground truth: the minimal variant is the exact DQI prescription; assert the
    # known optimum probabilities hold and that the two variants agree.
    for label, p in [("minimal", p_min), ("structured", p_struct)]:
        opt_x = int(np.argmax(p))
        print(f"  {label:11s}: P(x=3) = {p[3]:.4f}   argmax x = {opt_x}   "
              f"min other = {min(p[i] for i in range(8) if i != 3):.4f}   "
              f"max other = {max(p[i] for i in range(8) if i != 3):.4f}")
        assert abs(p[3] - 0.8307) < 1e-4, f"{label}: P(x=3) {p[3]} != 0.8307"
        for i in range(8):
            if i != 3:
                assert abs(p[i] - 0.0242) < 1e-3, f"{label}: P(x={i}) {p[i]} != 0.0242"
    assert np.allclose(p_struct, p_min, atol=1e-9), "variants disagree!"
    print("  PASS: both variants reproduce P(x=3)=0.8307, all others ~0.0242.")

    # --- 3b. shots sim + satisfied-count scoring ---
    section("VERIFY: AerSimulator shots — mean satisfied + P(opt)")
    st1 = dqi_satisfaction_stats(B, v, code, 1)
    opt = st1["max_satisfied"]
    qc_min_meas = build_minimal(a, n_syn, with_measurements=True)
    mean_sim, hist, sol_hist, shots = shots_score(qc_min_meas, B, v)
    p_opt = hist.get(opt, 0.0)
    print(f"  minimal variant ({shots} shots):")
    print(f"    mean satisfied = {mean_sim:.3f}/{m}  (random {m/2})  "
          f"(classical predicts {st1['mean_satisfied']:.3f})")
    print(f"    P(optimum={opt}/{m}) = {p_opt:.4f}")
    print(f"    satisfied-count distribution: "
          f"{ {s: round(hist[s], 4) for s in sorted(hist, reverse=True)} }")
    assert abs(mean_sim - 6.32) < 0.05, f"mean satisfied off: {mean_sim}"
    assert abs(p_opt - 0.83) < 0.03, f"P(opt) off: {p_opt}"
    print("  PASS: mean ~6.32/7, P(opt) ~0.83.")

    # --- financial readout (demo narrative intact) ---
    best_x = max(
        sol_hist,
        key=lambda x: (int(np.sum((B @ np.array(x)) % 2 == v)), sol_hist[x]),
    )
    best_score = int(np.sum((B @ np.array(best_x)) % 2 == v))
    readout = decode_portfolio(code, ladder, np.array(best_x))
    print(f"\n  FINANCIAL READOUT (most-probable best solution):")
    print(f"    best bitstring (syndrome vars) = {list(best_x)}  "
          f"(satisfies {best_score}/{m})")
    if readout["included_bonds"]:
        print(f"    decoded 'odd-one-out' bond(s): {readout['included_bonds']}")
        for d in readout["bond_details"]:
            print(f"       bond {d['bond']}: maturity {d['maturity']:.3f} yr, "
                  f"locator alpha^{d['bond']} = {d['locator']}")
    else:
        print(f"    decoded set: none (syndrome consistent with no odd bond)")

    # --- 4. transpile to IQM + route under restricted connectivity ---
    section("TRANSPILE: IQM basis (all-to-all) + routed line/triangle")
    results = {}
    for label, qc in [("structured", qc_struct), ("minimal", qc_min)]:
        iqm = gate_stats(
            transpile(qc, basis_gates=IQM_BASIS, optimization_level=3)
        )
        line_cz, line_d = routed_cz(qc, [[0, 1], [1, 2]])
        tri_cz, tri_d = routed_cz(qc, [[0, 1], [1, 2], [0, 2]])
        results[label] = {
            "qubits": iqm["qubits"], "logical_cz": iqm["2q_gates"],
            "depth": iqm["depth"],
            "line_cz": line_cz, "line_depth": line_d,
            "tri_cz": tri_cz, "tri_depth": tri_d,
        }

    b = BASELINE
    print(f"  {'variant':<22}{'qubits':>7}{'logCZ':>7}{'depth':>7}"
          f"{'lineCZ':>8}{'lineD':>7}{'triCZ':>7}{'triD':>7}")
    print("  " + "-" * 72)
    print(f"  {'OLD baseline (11q)':<22}{b['qubits']:>7}{b['logical_cz']:>7}"
          f"{b['depth']:>7}{b['routed_cz']:>8}{'-':>7}{b['routed_cz']:>7}{'-':>7}"
          f"   <- 154 routed CZ, decohered")
    for label in ("structured", "minimal"):
        r = results[label]
        print(f"  {'binary ' + label:<22}{r['qubits']:>7}{r['logical_cz']:>7}"
              f"{r['depth']:>7}{r['line_cz']:>8}{r['line_depth']:>7}"
              f"{r['tri_cz']:>7}{r['tri_depth']:>7}")
    wall = 40
    print(f"\n  coherence wall ~{wall} routed CZ. "
          f"minimal routed CZ (line) = {results['minimal']['line_cz']} "
          f"=> {'UNDER' if results['minimal']['line_cz'] < wall else 'OVER'} the wall.")
    assert results["minimal"]["logical_cz"] < 20, "minimal logical CZ too high"
    assert results["minimal"]["line_cz"] < wall, "minimal routed CZ over the wall"

    # --- 5. package hardware-submission artifact (NO submission) ---
    section("PACKAGE: artifacts/iqm/binary/  (no hardware submission)")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Export the MINIMAL measured circuit as portable QASM (match existing path:
    # QASM2 ships to LUMI; QASM3 kept for reference). 3 qubits, plain gates, so no
    # register-name collision with qelib1, but we still relabel to "sol" for clarity.
    from qiskit import QuantumRegister, ClassicalRegister
    import qiskit.qasm2 as q2
    import qiskit.qasm3 as q3

    # Transpile minimal to a portable standard basis (device-agnostic), with
    # measurements, mirroring artifacts/iqm/circuit/gen_circuit.py.
    STD_BASIS = ["cx", "rz", "sx", "x", "h", "rx", "ry"]
    qc_ship = transpile(
        build_minimal(a, n_syn, with_measurements=True),
        basis_gates=STD_BASIS, optimization_level=1,
    )
    sol = QuantumRegister(3, "sol")
    cl = ClassicalRegister(3, "c")
    relabeled = QuantumCircuit(sol, cl)
    relabeled.compose(qc_ship, qubits=[0, 1, 2], clbits=[0, 1, 2], inplace=True)
    qc_ship = relabeled

    qasm2_path = ARTIFACT_DIR / "q50_immunization_binary.qasm2"
    qasm3_path = ARTIFACT_DIR / "q50_immunization_binary.qasm3"
    q2.dump(qc_ship, str(qasm2_path))
    with open(qasm3_path, "w") as f:
        q3.dump(qc_ship, f)

    # Verify the shipped QASM2 reloads + re-simulates to the same P(opt).
    qc_re = q2.load(str(qasm2_path))
    mean_re, hist_re, _, _ = shots_score(qc_re, B, v)
    p_opt_re = hist_re.get(opt, 0.0)
    assert abs(p_opt_re - 0.83) < 0.03, f"reloaded P(opt) off: {p_opt_re}"
    print(f"  reload check: shipped QASM2 re-sims to P(opt)={p_opt_re:.4f} (faithful)")

    # Meta sidecar mirroring artifacts/iqm/circuit/q50_immunization_meta.json.
    iqm_min = results["minimal"]
    meta = {
        "instance": "7-bond fixed-income immunization (m_field=3, t=1, ell=1) "
                    "— BINARY RE-ENCODING (t=1 syndrome-basis collapse, 3 qubits)",
        "encoding": {
            "description": (
                "For t=1 over GF(2) the single-error location->syndrome map is a "
                "bijection on {0..7}. The 7-qubit one-hot error register + ancilla "
                "+ reversible Hamming decode of the 11-qubit circuit COLLAPSE into "
                "the 3-qubit syndrome register. The DQI computation reduces to: "
                "prepare pre-H amplitude vector a, apply H^3, measure. Same output "
                "distribution as the 11-qubit circuit, in 3 qubits."
            ),
            "syndrome_map_p_to_s": syndromes,
            "syndrome_map_is_bijection_1_to_7": sorted(syndromes) == list(range(1, 8)),
            "optimal_weights_w": [float(w[0]), float(w[1])],
            "v_phase_flip_syndromes": flips,
            "pre_H_amplitude_vector_a": [float(x) for x in a],
        },
        "B": B.tolist(),
        "v": v.tolist(),
        "n": int(n_syn),
        "m": int(m),
        "num_qubits": int(iqm_min["qubits"]),
        "iqm_logical_cz": int(iqm_min["logical_cz"]),
        "iqm_depth": int(iqm_min["depth"]),
        "routed_cz_line": int(iqm_min["line_cz"]),
        "routed_depth_line": int(iqm_min["line_depth"]),
        "routed_cz_triangle": int(iqm_min["tri_cz"]),
        "routed_depth_triangle": int(iqm_min["tri_depth"]),
        "optimum": int(opt),
        "random_mean_satisfied": float(m / 2),
        "random_p_opt": 1.0 / (1 << n_syn),
        "simulated_mean_satisfied": float(mean_sim),
        "simulated_p_opt": float(p_opt),
        "statevector_p_opt": float(p_min[3]),
        "sim_shots": int(shots),
        "baseline_11q": {
            "qubits": b["qubits"], "logical_cz": b["logical_cz"],
            "routed_cz": b["routed_cz"], "depth": b["depth"],
            "hardware_result": "decohered to random (P(opt) 0.122 vs random 0.125)",
        },
        "bond_ladder": {
            "maturities": [float(x) for x in ladder.maturities],
            "locators": [int(x) for x in ladder.locators],
        },
        "financial_readout": {
            "best_bitstring": list(int(b_) for b_ in best_x),
            "satisfies": int(best_score),
            "included_bonds": readout["included_bonds"],
            "bond_details": readout["bond_details"],
        },
        "readout": {
            "description": (
                "Measured bitstring from measure_all() on 3 qubits = the syndrome "
                "register. Strip spaces, REVERSE to little-endian, the solution "
                "vars x are bits[0:3] (no offset — no y register). "
                "satisfied(x) = sum((B @ x) % 2 == v)."),
            "m_offset": 0,
            "n_vars": int(n_syn),
            "bit_order": "reverse measured string, then index [0:3]",
        },
        "provenance": {
            "generator": "scripts/immunization_binary.py",
            "amplitude_vector": "computed from (optimal weights, syndrome map, "
                                "v-signs) — NOT hardcoded",
            "no_hardware_submission": True,
        },
    }
    meta_path = ARTIFACT_DIR / "q50_immunization_binary_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for p in (qasm2_path, qasm3_path, meta_path):
        print(f"  wrote {p}")

    # --- summary ---
    section("SUMMARY")
    print(f"  11-qubit baseline : 11q  {b['logical_cz']} logical CZ  "
          f"{b['routed_cz']} routed CZ  depth {b['depth']}  -> DECOHERED")
    print(f"  binary minimal    : {iqm_min['qubits']}q  "
          f"{iqm_min['logical_cz']} logical CZ  "
          f"{iqm_min['line_cz']} routed CZ (line)  depth {iqm_min['depth']}  "
          f"-> P(opt)={p_opt:.3f}")
    print(f"  same DQI output, {b['routed_cz']} -> {iqm_min['line_cz']} routed CZ "
          f"(under the {wall}-CZ coherence wall).")
    print(f"\n  To run on Q50 (user does this; no submission performed here):")
    print(f"    copy {qasm2_path.name} to the LUMI scratch dir alongside a runner")
    print(f"    adapted from artifacts/iqm/q50_run_immunization.py (load via "
          f"qiskit.qasm2, transpile to backend, backend.run), then sbatch "
          f"artifacts/iqm/submit_q50_immunization.sh.")


if __name__ == "__main__":
    main()
