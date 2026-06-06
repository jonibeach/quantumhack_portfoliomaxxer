"""Fixed-income IMMUNIZATION as a natively-algebraic DQI instance — driver.

Shows that a bond-immunization instance (match a liability's duration /
convexity / higher key-rate moments by choosing a bond subset) is the SAME cheap
algebraic-decoder DQI instance the project already runs on hardware — only the
financial wrapper is new.

For each field GF(2^m), m in {3} (and {4} if it simulates):
  1. build the SYNTHETIC bond ladder + immunization instance B x = v;
  2. ASSERT B is byte-for-byte the binary BCH parity check (build_bch_instance);
  3. classical DQI amplification (dqi_satisfaction_stats);
  4. simulate build_dqi_circuit_algebraic -> mean satisfied,
     IQM-CZ, depth, qubits, P(opt);
  5. financial readout of the best decoded portfolio;
  6. compare to the known baseline (len 7, ell=1: 86 CZ, 5.04/7, P(opt) 0.52).

Also runs the RS-vs-binary-BCH impedance investigation.

Run:  uv run python scripts/immunization_prototype.py
"""

import warnings
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from dqi_portfolio import (
    build_immunization_instance,
    immunization_is_bch_instance,
    build_bch_instance,
    dqi_satisfaction_stats,
    build_dqi_circuit_algebraic,
    decode_portfolio,
    rs_immunization_check,
    gate_stats,
)
from dqi_portfolio.bases import IQM_BASIS, SIM_BASIS
from dqi_portfolio.readout import score_counts
from dqi_portfolio.report import section

warnings.filterwarnings("ignore")


def simulate(qc, B, v, m, n):
    """Return (mean_satisfied, histogram of satisfied-count -> prob)."""
    sim = AerSimulator()
    qct = transpile(qc, sim, basis_gates=SIM_BASIS)
    counts = sim.run(qct, shots=16384).result().get_counts()
    # Full DQI circuit: solution register x follows the m-qubit error register y.
    r = score_counts(counts, B, v, x_offset=m)
    tot = r["total"]
    return r["mean"], {k: r["hist"][k] / tot for k in r["hist"]}, r["sol_hist"], tot


def run_field(m_field, t=1, seed=0):
    section(f"IMMUNIZATION INSTANCE  GF(2^{m_field})  t={t}  (seed={seed})")

    # --- build + faithfulness check ---
    chk = immunization_is_bch_instance(m_field, t, seed)
    B, v, code, ladder = (chk["B"], chk["v"], chk["code"], chk["ladder"])
    m, n = B.shape
    print(ladder.describe())
    print(f"\n  instance: {m} constraints (bond positions) x {n} vars (syndrome bits)")
    print(f"  FAITHFULNESS:  B == binary BCH parity check ?  {chk['B_equals_bch']}")
    print(f"                 H columns are bit-expansion of locators alpha^j ?  "
          f"{chk['locator_columns_ok']}")
    assert chk["B_equals_bch"], "immunization B is NOT the BCH instance — wrapper altered matrix!"
    print(f"  => the circuit is UNCHANGED from the existing BCH demo; only the "
          f"financial reading is new.")
    print(f"  liability target syndrome v = {v.tolist()}")

    # --- classical amplification ---
    print(f"\n  classical DQI amplification (random baseline {m/2:.2f}/{m}):")
    for ell in (1, 2, 3):
        st = dqi_satisfaction_stats(B, v, code, ell)
        print(f"    ell={ell}: mean satisfied = {st['mean_satisfied']:.3f}/{m}"
              f"   (brute-force max = {st['max_satisfied']})")

    # --- simulate the reversible circuit ---
    ell = 1
    qc = build_dqi_circuit_algebraic(code, B, v, ell=ell)
    iqm = gate_stats(transpile(qc, basis_gates=IQM_BASIS, optimization_level=3))
    raw = gate_stats(qc)

    qc_meas = build_dqi_circuit_algebraic(code, B, v, ell=ell, with_measurements=True)
    mean_sim, hist, sol_hist, shots = simulate(qc_meas, B, v, m, n)

    st1 = dqi_satisfaction_stats(B, v, code, ell)
    opt = st1["max_satisfied"]
    p_opt = hist.get(opt, 0.0)

    print(f"\n  reversible circuit (ell={ell}):")
    print(f"    raw : {raw['qubits']}q  {raw['2q_gates']} 2q  depth {raw['depth']}")
    print(f"    IQM : {iqm['qubits']}q  {iqm['2q_gates']} CZ  depth {iqm['depth']}")
    print(f"    SIMULATED mean satisfied = {mean_sim:.3f}/{m}  (random {m/2})  "
          f"(classical predicts {st1['mean_satisfied']:.3f})")
    print(f"    P(optimum={opt}/{m}) = {p_opt:.3f}")
    print(f"    satisfied-count distribution:")
    for s in sorted(hist, reverse=True):
        print(f"      {s}/{m}: {100*hist[s]:5.1f}%")

    # --- financial readout of the best decoded portfolio ---
    print(f"\n  FINANCIAL READOUT of the most-probable best-scoring solution:")
    best_x = None
    best_score = -1
    best_count = -1
    for x, c in sol_hist.items():
        s = int(np.sum((B @ np.array(x)) % 2 == v))
        if s > best_score or (s == best_score and c > best_count):
            best_score, best_count, best_x = s, c, x
    readout = decode_portfolio(code, ladder, np.array(best_x))
    print(f"    best solution bitstring (syndrome vars) = {list(best_x)}  "
          f"(satisfies {best_score}/{m})")
    if readout["included_bonds"]:
        print(f"    decoded 'odd-one-out' bond(s): {readout['included_bonds']}")
        for d in readout["bond_details"]:
            print(f"       bond {d['bond']}: maturity {d['maturity']:.3f} yr, "
                  f"locator alpha^{d['bond']} = {d['locator']}")
    else:
        print(f"    decoded set: none (syndrome consistent with no odd bond)")
    res = readout["residual"]
    print(f"    immunization residual: {res['field_nonzero_syndromes']} nonzero "
          f"field syndromes (0 = perfect combinatorial match)")
    print(f"    R-side moment readout (LABELLED companion): {res['real_moment_readout_LABELLED']}")

    # --- baseline comparison ---
    print(f"\n  BASELINE COMPARISON (known algebraic baseline: len 7, ell=1, "
          f"86 CZ, 5.04/7, P(opt) 0.52):")
    print(f"    this instance: {iqm['qubits']}q, {iqm['2q_gates']} CZ, "
          f"mean {mean_sim:.2f}/{m}, P(opt) {p_opt:.2f}")
    print(f"    same cheap ALGEBRAIC regime?  "
          f"{'YES' if iqm['2q_gates'] < 400 and mean_sim > m/2 + 0.5 else 'CHECK'}")

    return {
        "m_field": m_field, "qubits": iqm["qubits"], "cz": iqm["2q_gates"],
        "depth": iqm["depth"], "mean": mean_sim, "p_opt": p_opt, "opt": opt, "m": m,
    }


def rs_investigation():
    section("RS-vs-binary-BCH IMPEDANCE INVESTIGATION")
    print("  Q: does immunization map onto the existing BINARY-BCH circuit, or "
          "need a")
    print("     generalized Reed-Solomon (symbol-level) decoder the builder "
          "lacks?\n")
    for (mf, t) in [(3, 1), (4, 2), (4, 3)]:
        r = rs_immunization_check(mf, t)
        print(f"  GF(2^{mf}) t={t}: planted symbol errors at {r['planted_error_positions']} "
              f"(vals {r['planted_error_symbols']})")
        print(f"            classical RS Berlekamp-Massey recovered "
              f"{r['rs_classical_BM_recovered_positions']}  "
              f"-> decode_ok = {r['rs_classical_BM_decode_ok']}")
    print()
    print("  " + r["note"].replace("\n", "\n  "))


def main():
    results = []
    results.append(run_field(3, t=1))
    # m_field=4 -> length-15 ladder; circuit ~ 19 qubits + ancillas, simulate if it fits.
    try:
        results.append(run_field(4, t=1))
    except Exception as e:
        section("GF(2^4) length-15 ladder")
        print(f"  skipped simulation (too large / error): {e}")

    rs_investigation()

    section("SUMMARY TABLE")
    print("  field      qubits  CZ    depth   mean/ m    P(opt)   regime")
    print("  " + "-" * 60)
    print("  baseline(7)  11     86     -     5.04/7     0.52    ALGEBRAIC (prior)")
    for r in results:
        print(f"  GF(2^{r['m_field']})({r['m']:>2})  {r['qubits']:>3}   {r['cz']:>4}  "
              f"{r['depth']:>4}    {r['mean']:.2f}/{r['m']:>2}   {r['p_opt']:.2f}    "
              f"ALGEBRAIC (measured)")


if __name__ == "__main__":
    main()
