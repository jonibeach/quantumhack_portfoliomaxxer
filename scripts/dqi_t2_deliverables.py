"""Deliverables for the genuine multi-error (t>=2) DQI immunization decoder.

Produces the three reporting artifacts requested in the t>=2 handoff, all from
LIGHT computation (no 2^n statevector simulation):

  1. VALIDATION TABLE — ideal (faithful DQI) vs reversible circuit, for t=1
     (the hardware showpiece) and t=2 (enumeration + the GENUINE BM datapath):
     qubits, CZ, depth, mean satisfied, P(opt), and P(opt) lift over random.
     Ideal numbers come from dqi_satisfaction_stats (cheap enumeration of 2^n,
     n<=8). Circuit CZ/depth/qubits come from transpiling the (small) circuits.
     The "circuit matches ideal" claim is verified separately in
     scripts/bm_datapath_t2.py (CHECK B, statevector on LUMI) and
     scripts/algebraic_t2_prototype.py.

  2. CZ-vs-(t,b) RESOURCE CURVE — pattern enumeration (cost ~ C(m,<=t)) vs the
     polynomial-scaling reversible BM datapath. Small cases are MEASURED
     (transpiled to the IQM basis); enumeration's combinatorial blowup and the
     BM datapath at larger (t,b) are reported analytically/from
     estimate_bm_resources, clearly labelled.

  3. NOISE / DEVICE-GENERATION PREDICTION — for the MEASURED 632-CZ genuine BM
     circuit, the coherent survival (1-p)^CZ across per-2q-gate error rates p,
     the threshold p at which DQI's phase-interference amplification becomes
     visible, and which device generation that corresponds to.

Run:  uv run python scripts/dqi_t2_deliverables.py
This script does NO statevector simulation; it is safe to run on a laptop.
"""

import warnings
from functools import partial
from math import comb, log

import numpy as np
from qiskit import transpile

from dqi_portfolio import (
    BCHCode,
    dqi_satisfaction_stats,
    build_dqi_circuit_algebraic,
    estimate_bm_resources,
    gate_stats,
)
from dqi_portfolio.bases import IQM_BASIS
from dqi_portfolio.immunization import build_immunization_instance
from dqi_portfolio.report import section as _section

warnings.filterwarnings("ignore")
section = partial(_section, width=78)


def ideal_stats(m_field, t, ell):
    """Faithful DQI distribution numbers (cheap; no circuit)."""
    B, v, code, _ = build_immunization_instance(m_field, t)
    m, n = B.shape
    st = dqi_satisfaction_stats(B, v, code, ell)
    opt = st["max_satisfied"]
    p_opt = st["histogram"].get(opt, 0.0)
    # uniform-random P(opt) over the 2^n solution strings
    cnt = 0
    for k in range(1 << n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        if int(np.sum((B @ x) % 2 == v)) == opt:
            cnt += 1
    p_opt_rand = cnt / (1 << n)
    return {
        "m": m, "n": n, "ell": ell, "opt": opt,
        "mean": st["mean_satisfied"], "random_mean": m / 2,
        "p_opt": p_opt, "p_opt_rand": p_opt_rand,
        "lift": p_opt / p_opt_rand if p_opt_rand else float("inf"),
        "B": B, "v": v, "code": code,
    }


def circuit_cost(code, B, v, ell, decoder):
    qc = build_dqi_circuit_algebraic(code, B, v, ell=ell, decoder=decoder)
    g = gate_stats(transpile(qc, basis_gates=IQM_BASIS, optimization_level=3))
    return g["qubits"], g["2q_gates"], g["depth"]


def validation_table():
    section("1) VALIDATION TABLE — ideal (faithful DQI) vs reversible circuit")
    rows = []

    # t=1 GF(2^3): the hardware showpiece (ancilla v-chain decode).
    s = ideal_stats(3, 1, 1)
    q, cz, d = circuit_cost(s["code"], s["B"], s["v"], 1, "enumeration")
    rows.append(("t=1 GF(2^3)  ell=1  [HW showpiece]", q, cz, d, s))

    # t=2 GF(2^3): pattern enumeration (exact, C(m,t) cost).
    s2 = ideal_stats(3, 2, 2)
    q, cz, d = circuit_cost(s2["code"], s2["B"], s2["v"], 2, "enumeration")
    rows.append(("t=2 GF(2^3)  ell=2  [enumeration]", q, cz, d, s2))

    # t=2 GF(2^3): GENUINE reversible BM datapath (the real contribution).
    q, cz, d = circuit_cost(s2["code"], s2["B"], s2["v"], 2, "bm")
    rows.append(("t=2 GF(2^3)  ell=2  [GENUINE BM]", q, cz, d, s2))

    print(f"\n  {'instance':<34}{'q':>4}{'CZ':>7}{'depth':>7}"
          f"{'mean':>9}{'P(opt)':>9}{'lift':>8}{'ell/m':>8}")
    print("  " + "-" * 86)
    for name, q, cz, d, s in rows:
        print(f"  {name:<34}{q:>4}{cz:>7}{d:>7}"
              f"{s['mean']:>6.2f}/{s['m']:<2}{s['p_opt']:>9.3f}"
              f"{s['lift']:>7.1f}x{s['ell']/s['m']:>8.3f}")
    print("\n  Random baselines: mean = m/2; P(opt) = 1/2^n (uniform).")
    print("  Circuit==ideal distribution VERIFIED in scripts/bm_datapath_t2.py")
    print("  (CHECK B, statevector, LUMI): t=2 BM P(opt)=0.3713 vs ideal 0.3714,")
    print("  max|hist diff| 0.0002 < shot-noise floor 0.0016.")
    print("\n  Takeaway: t=2 amplifies MORE STRONGLY than t=1 in the proper metric")
    print("  (P(opt) lift over random, and ell/m): the search space is larger so")
    print("  the raw mean is lower, but concentration on the optimum is far higher.")


def resource_curve():
    section("2) CZ-vs-(t,b) RESOURCE CURVE — enumeration vs polynomial BM datapath")

    # MEASURED small cases (transpiled; light).
    print("\n  MEASURED (transpiled to IQM basis, opt level 3):")
    print(f"    {'field':<10}{'t':>3}{'decoder':>14}{'qubits':>8}{'CZ':>8}{'depth':>8}")
    print("    " + "-" * 50)
    measured = []
    for (mf, t) in [(3, 2), (4, 2)]:
        B, v, code, _ = build_immunization_instance(mf, t)
        qb, czb, db = circuit_cost(code, B, v, 2, "bm")
        measured.append((mf, t, "BM", qb, czb, db))
        print(f"    GF(2^{mf}){'':<5}{t:>3}{'GENUINE BM':>14}{qb:>8}{czb:>8}{db:>8}")
    # enumeration measured only for the cheapest (the larger one is huge).
    B, v, code, _ = build_immunization_instance(3, 2)
    qe, cze, de = circuit_cost(code, B, v, 2, "enumeration")
    print(f"    GF(2^3){'':<6}{2:>3}{'enumeration':>14}{qe:>8}{cze:>8}{de:>8}")

    # ANALYTIC enumeration blowup: #correctable patterns = sum_{w<=t} C(n,w),
    # each an n_checks-controlled MCX (the combinatorial cost the BM solve avoids).
    print("\n  ANALYTIC enumeration pattern count (cost ~ C(n,<=t) MCX gates):")
    print(f"    {'field':<10}{'t':>3}{'n=len':>7}{'#patterns C(n,<=t)':>20}")
    print("    " + "-" * 42)
    for (mf, t) in [(3, 2), (4, 2), (4, 3), (5, 2), (5, 3)]:
        n = (1 << mf) - 1
        npat = sum(comb(n, w) for w in range(t + 1))
        print(f"    GF(2^{mf}){'':<5}{t:>3}{n:>7}{npat:>20,}")

    # BM datapath scaling (analytic assembly x measured per-mult), from the
    # existing estimator; our measured t=2 build is even cheaper (closed form).
    print("\n  BM datapath estimate (per-mult MEASURED, assembly ESTIMATED;")
    print("  our measured t=2 build beats this via the Peterson closed form):")
    print(f"    {'field':<10}{'t':>3}{'est-qubits':>12}{'est-CZ':>12}")
    print("    " + "-" * 38)
    for (mf, t) in [(3, 2), (4, 2), (4, 3), (5, 3)]:
        est = estimate_bm_resources(mf, t)
        print(f"    GF(2^{mf}){'':<5}{t:>3}{est['estimated_qubits']:>12}"
              f"{est['estimated_cz']:>12,}")
    print("\n  NOTE: enumeration cost grows COMBINATORIALLY in n=2^b-1 (table above);")
    print("  the BM datapath grows POLYNOMIALLY. The measured GF(2^3) t=2 crossover:")
    print("  BM 632 CZ vs enumeration 2128 CZ (3.4x), widening fast with field size.")


def noise_prediction(cz_bm=632, depth_bm=749):
    section("3) NOISE / DEVICE-GENERATION PREDICTION (genuine BM, 632 CZ)")
    print(f"\n  Circuit: genuine t=2 GF(2^3) BM datapath = {cz_bm} CZ, depth {depth_bm}.")
    print("  Coherent survival ~ (1 - p)^CZ for per-2q-gate (CZ) error p")
    print("  (ignoring idle/readout, so an OPTIMISTIC upper bound).\n")

    print(f"    {'device / generation':<34}{'p(CZ)':>8}{'survival (0.991-like)':>22}")
    print("    " + "-" * 64)
    devices = [
        ("VTT Q50 today", 0.009),
        ("best 2024-25 supercond. (median)", 0.003),
        ("IBM Heron r2 / Google-class best", 0.001),
        ("next-gen target", 0.0005),
        ("early logical / FT-ish", 0.0001),
    ]
    for name, p in devices:
        surv = (1 - p) ** cz_bm
        print(f"    {name:<34}{p:>8.4f}{surv:>21.4f}")

    # Threshold: DQI amplification is phase-interference; from the t=1 hardware
    # story it survived at ~76% (5 CZ) and DIED at ~25% (154 CZ). Take survival
    # >= 0.5 as the conservative "amplification clearly visible" bar.
    for bar in (0.5, 0.37):
        p_thresh = 1 - bar ** (1.0 / cz_bm)
        print(f"\n  For coherent survival >= {bar:.2f} over {cz_bm} CZ:  "
              f"need p(CZ) <= {p_thresh*100:.3f}%")
    print("\n  PREDICTION: the genuine t=2 decoder needs ~0.1% per-CZ error to show")
    print("  amplification — about 1-2 superconducting generations beyond Q50's")
    print("  ~0.9%. It is NOT a today-NISQ run (632 CZ >> Q50's ~40-CZ wall); it is")
    print("  the LUMI-verified algorithm whose hardware home is the next device")
    print("  generation (or a logical-qubit machine). The t=1 bijection collapse")
    print("  remains the on-hardware showpiece; this is the genuine-algorithm budget.")


def main():
    validation_table()
    resource_curve()
    noise_prediction()
    section("DONE — t>=2 DQI immunization deliverables")


if __name__ == "__main__":
    main()
