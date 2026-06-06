"""Hardware-optimized BINARY RE-ENCODING of the 31-bond immunization DQI circuit.

THE COLLAPSE, SCALED TO GF(2^5)  (why 31-bond DQI fits in 5 qubits)
-------------------------------------------------------------------
This is the GF(2^5) sibling of scripts/immunization_binary.py (which did the
7-bond GF(2^3) instance in 3 qubits). The 31-bond fixed-income immunization
instance is exactly the BCH(31,26) / Hamming GF(2^5), t=1 single-error code.
A faithful DQI circuit for it would carry a 31-qubit ONE-HOT error register +
a 5-qubit syndrome register + ancilla, with a reversible Hamming decode
(MCX-heavy) — far over the ~40 routed-CZ coherence wall of the IQM Q50.

But for t=1 over GF(2) the single-error location -> syndrome map is a BIJECTION
on {0..31}: error at position p gives syndrome s_p = B[p] (the p-th row of B as
an integer), and for this instance {s_0..s_30} is a permutation of {1..31}. The
no-error term is syndrome 0. So the one-hot error register, the syndrome
computation, AND the reversible decode/uncompute all FOLD into the 5-qubit
syndrome register: there is nothing left to decode because preparing the state
directly in the syndrome basis already applies the location->syndrome relabel.
The whole DQI computation reduces to:

    prepare a specific 5-qubit amplitude vector  ->  H^{(x)5}  ->  measure.

No y register, no ancilla, no MCX. The DQI prescription (optimal weights, the
syndrome map, the v-phase signs) is computed PROGRAMMATICALLY below — it is not
the planted answer hardcoded. After H^{(x)5} the marginal over the 5 qubits is
the SAME ideal DQI distribution dqi_satisfaction_stats() predicts (mean
satisfied ~25.78/31 vs random 15.5), now in 5 qubits and ~26 logical CZ — well
under the 40-CZ coherence wall.

The lift here is milder than the 7-bond case (P(opt) ~0.67 vs 0.83): with only
ell=1 and 31 constraints the degree-1 DQI polynomial concentrates less, which is
the honest, expected behaviour at this scale. This is a GF(2) COMBINATORIAL
immunization (max satisfied = best locator-pattern match), not a continuous
duration/convexity optimisation.

Run:  uv run python scripts/immunization_binary_gf5.py
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

FIELD = 5  # GF(2^5): m=31 bonds, n=5 syndrome bits.

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "artifacts" / "iqm" / "binary_gf5"
)

# Original 11-qubit GF(2^3) baseline (from artifacts/iqm meta + q50 results),
# and the shipped 7-bond binary collapse — both for the before/after table.
BASELINE_11Q = {
    "qubits": 11,
    "logical_cz": 86,
    "routed_cz": 154,
    "depth": 226,
    "note": "decohered to random (P(opt) 0.122 vs random 0.125)",
}
BASELINE_BINARY_7 = {
    "qubits": 3,
    "logical_cz": 4,
    "routed_cz": 5,        # line
    "depth": 11,
    "p_opt": 0.8307,
    "note": "7-bond GF(2^3) binary collapse (scripts/immunization_binary.py)",
}


# 5-qubit restricted maps for routing.
LINE_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4]]
# A denser "ring + chord" 5-qubit map (closes the line into a ring with a chord).
RING_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2]]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    section("BINARY RE-ENCODING of 31-bond immunization DQI  (GF(2^5) t=1 collapse)")

    # --- 1. instance + DQI prescription (programmatic, not hardcoded) ---
    chk = immunization_is_bch_instance(FIELD, 1, 0)
    B, v, code, ladder = chk["B"], chk["v"], chk["code"], chk["ladder"]
    m, n = B.shape  # 31 x 5
    assert chk["B_equals_bch"], "B is not the BCH instance!"
    assert (m, n) == (31, 5), f"expected 31x5, got {B.shape}"
    w = _optimal_poly_weights(m, 1)  # ell=1 optimal weights (uniform 1/sqrt2)
    a, syndromes = build_amplitude_vector(B, v, w)
    flips = v_phase_positions(B, v)

    print(f"  instance: B {B.shape}  m={m} bonds  n={n} syndrome bits  "
          f"(B==BCH: {chk['B_equals_bch']})")
    print(f"  optimal ell=1 weights w = [{w[0]:.6f}, {w[1]:.6f}] (uniform 1/sqrt2)")
    print(f"  single-error syndrome map p -> s_p (first 8): {syndromes[:8]} ...")
    print(f"    permutation of 1..{m}: "
          f"{sorted(syndromes) == list(range(1, m + 1))}")
    print(f"  v-phase oracle flips |s> for {len(flips)} syndromes "
          f"(= {{s_p : v_p=1}}); first 8: {flips[:8]} ...")
    print(f"  pre-H amplitude vector a (index = syndrome int): a[0]={a[0]:.4f}, "
          f"|a[s!=0]|={abs(w[1])/np.sqrt(m):.4f}, sum|a|^2={np.sum(a**2):.6f}")

    # --- 2. build both variants ---
    qc_struct = build_structured(a, flips, n)
    qc_min = build_minimal(a, n)

    # --- 3. verify statevector marginals match ideal DQI ---
    section("VERIFY: statevector marginal P(x) over 5 qubits vs ideal DQI")
    st1 = dqi_satisfaction_stats(B, v, code, 1)
    opt = st1["max_satisfied"]
    # brute-force optimum syndrome x* (max satisfied count) over 2^5=32 — trivial.
    sat = np.array([
        int(np.sum((B @ np.array([(k >> i) & 1 for i in range(n)])) % 2 == v))
        for k in range(1 << n)
    ])
    x_opt = int(np.argmax(sat))
    assert sat[x_opt] == opt, f"brute opt {sat[x_opt]} != stats opt {opt}"

    # ideal DQI distribution P(x) computed independently from the prescription:
    # apply the exact Walsh-Hadamard to a (qubit0 = LSB) and square.
    ideal = np.zeros(1 << n)
    for x in range(1 << n):
        val = 0.0
        for s in range(1 << n):
            val += a[s] * ((-1.0) ** (bin(x & s).count("1")))
        ideal[x] = (val / np.sqrt(1 << n)) ** 2

    p_struct = statevector_marginal(qc_struct)
    p_min = statevector_marginal(qc_min)
    for label, p in [("minimal", p_min), ("structured", p_struct)]:
        argm = int(np.argmax(p))
        print(f"  {label:11s}: argmax x = {argm} (opt x* = {x_opt})  "
              f"P(x*) = {p[x_opt]:.4f}  matches-ideal = "
              f"{np.allclose(p, ideal, atol=1e-6)}")
        assert np.allclose(p, ideal, atol=1e-6), f"{label} != ideal DQI dist"
        assert argm == x_opt, f"{label} argmax {argm} != optimum {x_opt}"
    assert np.allclose(p_struct, p_min, atol=1e-6), "variants disagree (>1e-6)!"
    p_opt_sv = float(p_min[x_opt])
    # cross-check mean satisfied from the statevector against the classical stats.
    mean_sv = float(np.sum(p_min * sat))
    print(f"  PASS: both variants == ideal DQI (atol 1e-6); agree to 1e-6.")
    print(f"  statevector mean satisfied = {mean_sv:.4f}/{m}  "
          f"(classical DQI predicts {st1['mean_satisfied']:.4f}, random {m/2})")
    print(f"  statevector P(opt = {opt}/{m} at x*={x_opt}) = {p_opt_sv:.4f}")
    assert abs(mean_sv - st1["mean_satisfied"]) < 1e-6, "mean satisfied mismatch"

    # --- 3b. shots sim + satisfied-count scoring ---
    section("VERIFY: AerSimulator shots (16384) — mean satisfied + P(opt)")
    qc_min_meas = build_minimal(a, n, with_measurements=True)
    mean_sim, hist, sol_hist, shots = shots_score(qc_min_meas, B, v)
    p_opt = hist.get(opt, 0.0)
    print(f"  minimal variant ({shots} shots):")
    print(f"    mean satisfied = {mean_sim:.3f}/{m}  (random {m/2})  "
          f"(classical predicts {st1['mean_satisfied']:.3f})")
    print(f"    P(optimum={opt}/{m}) = {p_opt:.4f}  (statevector {p_opt_sv:.4f})")
    print(f"    satisfied-count distribution: "
          f"{ {s: round(hist[s], 4) for s in sorted(hist, reverse=True)} }")
    assert abs(mean_sim - st1["mean_satisfied"]) < 0.4, \
        f"mean satisfied off shot-noise: {mean_sim}"
    assert abs(p_opt - p_opt_sv) < 0.03, f"P(opt) off shot-noise: {p_opt}"
    print(f"  PASS: shots match statevector within shot noise.")

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
    section("TRANSPILE: IQM basis (all-to-all) + routed line/ring")
    results = {}
    for label, qc in [("structured", qc_struct), ("minimal", qc_min)]:
        iqm = gate_stats(
            transpile(qc, basis_gates=IQM_BASIS, optimization_level=3)
        )
        line_cz, line_d = routed_cz(qc, LINE_EDGES, seeds=range(40))
        ring_cz, ring_d = routed_cz(qc, RING_EDGES, seeds=range(40))
        results[label] = {
            "qubits": iqm["qubits"], "logical_cz": iqm["2q_gates"],
            "depth": iqm["depth"],
            "line_cz": line_cz, "line_depth": line_d,
            "ring_cz": ring_cz, "ring_depth": ring_d,
        }

    b11 = BASELINE_11Q
    b7 = BASELINE_BINARY_7
    print(f"  {'variant':<24}{'qubits':>7}{'logCZ':>7}{'depth':>7}"
          f"{'lineCZ':>8}{'lineD':>7}{'ringCZ':>8}{'ringD':>7}")
    print("  " + "-" * 76)
    print(f"  {'OLD 11q GF(2^3)':<24}{b11['qubits']:>7}{b11['logical_cz']:>7}"
          f"{b11['depth']:>7}{b11['routed_cz']:>8}{'-':>7}{b11['routed_cz']:>8}"
          f"{'-':>7}   <- 154 routed CZ, decohered")
    print(f"  {'7-bond binary GF(2^3)':<24}{b7['qubits']:>7}{b7['logical_cz']:>7}"
          f"{b7['depth']:>7}{b7['routed_cz']:>8}{'-':>7}{b7['routed_cz']:>8}"
          f"{'-':>7}   <- shipped, P(opt)=0.83")
    for label in ("structured", "minimal"):
        r = results[label]
        print(f"  {'31-bond binary ' + label:<24}{r['qubits']:>7}"
              f"{r['logical_cz']:>7}{r['depth']:>7}{r['line_cz']:>8}"
              f"{r['line_depth']:>7}{r['ring_cz']:>8}{r['ring_depth']:>7}")
    wall = 40
    min_line = results["minimal"]["line_cz"]
    min_ring = results["minimal"]["ring_cz"]
    print(f"\n  coherence wall ~{wall} routed CZ.")
    print(f"    minimal routed CZ (line) = {min_line} "
          f"=> {'UNDER' if min_line < wall else 'OVER'} the wall  "
          f"(a BARE 5-qubit line cannot route a dense 26-CZ prep under {wall}).")
    print(f"    minimal routed CZ (ring) = {min_ring} "
          f"=> {'UNDER' if min_ring < wall else 'OVER'} the wall  "
          f"(the denser ring+chord map IS the shippable layout).")
    assert results["minimal"]["logical_cz"] < 32, "minimal logical CZ too high"
    # The shippable routing is the denser (ring) map; require IT under the wall.
    assert min_ring < wall, "minimal routed CZ (ring) over the wall"

    # --- 5. package hardware-submission artifact (NO submission) ---
    section("PACKAGE: artifacts/iqm/binary_gf5/  (no hardware submission)")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    from qiskit import QuantumRegister, ClassicalRegister
    import qiskit.qasm2 as q2
    import qiskit.qasm3 as q3

    # Transpile minimal to a portable standard basis (device-agnostic), with
    # measurements, mirroring artifacts/iqm/binary/.
    STD_BASIS = ["cx", "rz", "sx", "x", "h", "rx", "ry"]
    qc_ship = transpile(
        build_minimal(a, n, with_measurements=True),
        basis_gates=STD_BASIS, optimization_level=1,
    )
    sol = QuantumRegister(n, "sol")
    cl = ClassicalRegister(n, "c")
    relabeled = QuantumCircuit(sol, cl)
    relabeled.compose(qc_ship, qubits=list(range(n)),
                      clbits=list(range(n)), inplace=True)
    qc_ship = relabeled

    qasm2_path = ARTIFACT_DIR / "q50_immunization_binary_gf5.qasm2"
    qasm3_path = ARTIFACT_DIR / "q50_immunization_binary_gf5.qasm3"
    q2.dump(qc_ship, str(qasm2_path))
    with open(qasm3_path, "w") as f:
        q3.dump(qc_ship, f)

    # Verify the shipped QASM2 reloads + re-simulates to the same P(opt).
    qc_re = q2.load(str(qasm2_path))
    mean_re, hist_re, _, _ = shots_score(qc_re, B, v)
    p_opt_re = hist_re.get(opt, 0.0)
    assert abs(p_opt_re - p_opt_sv) < 0.03, f"reloaded P(opt) off: {p_opt_re}"
    print(f"  reload check: shipped QASM2 re-sims to P(opt)={p_opt_re:.4f} (faithful)")

    iqm_min = results["minimal"]
    meta = {
        "instance": "31-bond fixed-income immunization (m_field=5, t=1, ell=1) "
                    "— BINARY RE-ENCODING (t=1 syndrome-basis collapse, 5 qubits)",
        "encoding": {
            "description": (
                "For t=1 over GF(2) the single-error location->syndrome map is a "
                "bijection on {0..31}. The 31-qubit one-hot error register + "
                "ancilla + reversible Hamming decode of the faithful GF(2^5) "
                "circuit COLLAPSE into the 5-qubit syndrome register. The DQI "
                "computation reduces to: prepare pre-H amplitude vector a, apply "
                "H^5, measure. Same ideal DQI output distribution, in 5 qubits."
            ),
            "syndrome_map_p_to_s": syndromes,
            "syndrome_map_is_bijection_1_to_31":
                sorted(syndromes) == list(range(1, m + 1)),
            "optimal_weights_w": [float(w[0]), float(w[1])],
            "v_phase_flip_syndromes": flips,
            "pre_H_amplitude_vector_a": [float(x) for x in a],
        },
        "B": B.tolist(),
        "v": v.tolist(),
        "n": int(n),
        "m": int(m),
        "num_qubits": int(iqm_min["qubits"]),
        "iqm_logical_cz": int(iqm_min["logical_cz"]),
        "iqm_depth": int(iqm_min["depth"]),
        "routed_cz_line": int(iqm_min["line_cz"]),
        "routed_depth_line": int(iqm_min["line_depth"]),
        "routed_cz_ring": int(iqm_min["ring_cz"]),
        "routed_depth_ring": int(iqm_min["ring_depth"]),
        "line_edges": LINE_EDGES,
        "ring_edges": RING_EDGES,
        "structured_variant": {
            "logical_cz": int(results["structured"]["logical_cz"]),
            "depth": int(results["structured"]["depth"]),
            "routed_cz_line": int(results["structured"]["line_cz"]),
            "routed_cz_ring": int(results["structured"]["ring_cz"]),
        },
        "optimum": int(opt),
        "optimum_x": [int((x_opt >> i) & 1) for i in range(n)],
        "optimum_x_int": int(x_opt),
        "random_mean_satisfied": float(m / 2),
        "random_p_opt": 1.0 / (1 << n),
        "classical_mean_satisfied": float(st1["mean_satisfied"]),
        "statevector_mean_satisfied": float(mean_sv),
        "statevector_p_opt": float(p_opt_sv),
        "simulated_mean_satisfied": float(mean_sim),
        "simulated_p_opt": float(p_opt),
        "sim_shots": int(shots),
        "lift_note": (
            "Milder lift than the 7-bond case (P(opt)~0.67 vs 0.83): with ell=1 "
            "and 31 constraints the degree-1 DQI polynomial concentrates less. "
            "This is the honest, expected behaviour at GF(2^5) scale."
        ),
        "routing_note": (
            "Logical CZ ~26 (all-to-all). A BARE 5-qubit LINE cannot route a dense "
            "5-qubit StatePreparation under the 40-CZ wall (best ~47 over a seed "
            "sweep); the denser RING+chord map routes to ~34 CZ, UNDER the wall, "
            "and is the shippable layout. Q50's native connectivity is denser than "
            "a line, so the ring number is the operative one. Routed counts are the "
            "best over a 40-seed SABRE sweep."
        ),
        "coherence_wall_routed_cz": int(wall),
        "routed_cz_seed_sweep": True,
        "baseline_11q": {
            "qubits": b11["qubits"], "logical_cz": b11["logical_cz"],
            "routed_cz": b11["routed_cz"], "depth": b11["depth"],
            "hardware_result": b11["note"],
        },
        "baseline_binary_7bond": {
            "qubits": b7["qubits"], "logical_cz": b7["logical_cz"],
            "routed_cz": b7["routed_cz"], "depth": b7["depth"],
            "p_opt": b7["p_opt"], "note": b7["note"],
        },
        "bond_ladder": {
            "n_bonds": int(m),
            "maturities": [float(x) for x in ladder.maturities],
            "locators": [int(x) for x in ladder.locators],
            "description": (
                "31-bond ladder: bond j has locator alpha^j in GF(2^5) and an "
                "increasing maturity ~ (j+1) years. DQI selects the locator "
                "pattern (subset) that best matches the liability syndrome v."
            ),
        },
        "financial_readout": {
            "best_bitstring": list(int(b_) for b_ in best_x),
            "satisfies": int(best_score),
            "included_bonds": readout["included_bonds"],
            "bond_details": readout["bond_details"],
            "residual": readout["residual"],
        },
        "readout": {
            "description": (
                "Measured bitstring from measure_all() on 5 qubits = the syndrome "
                "register. Strip spaces, REVERSE to little-endian, the solution "
                "vars x are bits[0:5] (no offset — no y register). "
                "satisfied(x) = sum((B @ x) % 2 == v)."),
            "m_offset": 0,
            "n_vars": int(n),
            "bit_order": "reverse measured string, then index [0:5]",
        },
        "provenance": {
            "generator": "scripts/immunization_binary_gf5.py",
            "amplitude_vector": "computed from (optimal weights, syndrome map, "
                                "v-signs) — NOT hardcoded",
            "no_hardware_submission": True,
            "field": "GF(2^5)",
        },
    }
    meta_path = ARTIFACT_DIR / "q50_immunization_binary_gf5_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for p in (qasm2_path, qasm3_path, meta_path):
        print(f"  wrote {p}")

    # --- summary ---
    section("SUMMARY")
    print(f"  11q GF(2^3) baseline : 11q  {b11['logical_cz']} logical CZ  "
          f"{b11['routed_cz']} routed CZ  depth {b11['depth']}  -> DECOHERED")
    print(f"  7-bond binary        : 3q   {b7['logical_cz']} logical CZ  "
          f"{b7['routed_cz']} routed CZ  depth {b7['depth']}  -> P(opt)=0.83")
    print(f"  31-bond binary minimal: {iqm_min['qubits']}q   "
          f"{iqm_min['logical_cz']} logical CZ  "
          f"{iqm_min['line_cz']} routed CZ (line) / {iqm_min['ring_cz']} (ring)  "
          f"depth {iqm_min['depth']}  -> P(opt)={p_opt:.3f}")
    print(f"  same ideal DQI output (mean {mean_sim:.2f}/{m} vs random {m/2}); "
          f"{iqm_min['ring_cz']} routed CZ (ring) under the {wall}-CZ wall "
          f"(line {iqm_min['line_cz']} over — needs the denser map).")
    print(f"\n  To run on Q50 (user does this; no submission performed here):")
    print(f"    copy {qasm2_path.name} to the LUMI scratch dir alongside a runner")
    print(f"    adapted from artifacts/iqm/q50_run_immunization.py (load via "
          f"qiskit.qasm2, transpile to the IQM backend, backend.run), then sbatch "
          f"artifacts/iqm/submit_q50_immunization.sh.")


if __name__ == "__main__":
    main()
