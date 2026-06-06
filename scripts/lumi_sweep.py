"""LUMI decoder-regime sweep (paper Table 2 / tab:regimes) — refactored API.

Reconstructed validation driver. Regenerates the decoder x size x ell scaling
table directly from the *refactored* package (no v-chain ancilla kwargs, which
were removed). Part B (genuine LDPC portfolio control) runs only if the optional
``dqi_portfolio.portfolio`` module is present; the immunization cleanup dropped
it, so it is skipped gracefully otherwise.

Run on LUMI inside the container:
  export PATH=<container>/bin:$PATH ; export PYTHONPATH=<repo>
  python scripts/lumi_sweep.py --out artifacts/lumi/sweep_results_refactored
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import Counter

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from dqi_portfolio import (
    build_bch_instance,
    build_dqi_circuit,
    build_dqi_circuit_algebraic,
    build_dqi_circuit_gje,
    gate_stats,
)
from dqi_portfolio.bases import IQM_BASIS

warnings.filterwarnings("ignore")

SIM_QUBIT_CAP = 28
SHOTS = 16384


def iqm_stats(qc):
    return gate_stats(transpile(qc, basis_gates=IQM_BASIS, optimization_level=3))


def iqm_routed_stats(qc):
    nq = qc.num_qubits
    side = math.ceil(math.sqrt(nq))
    cm = CouplingMap.from_grid(side, side)
    return gate_stats(transpile(qc, basis_gates=IQM_BASIS, coupling_map=cm,
                                optimization_level=3))


def brute_force_opt(B, v):
    m, n = B.shape
    best = 0
    for k in range(2 ** n):
        x = np.array([(k >> i) & 1 for i in range(n)])
        best = max(best, int(np.sum((B @ x) % 2 == v)))
    return best


def simulate_mean(qc_meas, m, n, B, v, opt):
    sim = AerSimulator()
    qct = transpile(qc_meas, sim, basis_gates=["cx", "rz", "rx", "ry", "h", "x"])
    counts = sim.run(qct, shots=SHOTS).result().get_counts()
    tot = wt = p_opt = 0
    hist = Counter()
    for bs, c in counts.items():
        bits = bs.replace(" ", "")[::-1]
        x = np.array([int(bits[m + i]) for i in range(n)])
        s = int(np.sum((B @ x) % 2 == v))
        tot += c
        wt += s * c
        hist[s] += c
        if s == opt:
            p_opt += c
    return wt / tot, p_opt / tot


def part_a():
    rows = []
    for m_field in (3, 4, 5):
        B, v, code = build_bch_instance(m_field, 1)
        m, n = B.shape
        opt = brute_force_opt(B, v)
        for ell in (1, 2, 3):
            for decoder in ("BP", "algebraic-t1", "GJE"):
                rec = {"part": "A", "decoder": decoder, "length": m, "n": n,
                       "ell": ell, "random_baseline": m / 2.0,
                       "brute_force_opt": opt}
                try:
                    if decoder == "BP":
                        qc = build_dqi_circuit(B, v, ell=ell, bp_iters=1)
                        qcm = build_dqi_circuit(B, v, ell=ell, bp_iters=1,
                                                with_measurements=True)
                    elif decoder == "algebraic-t1":
                        qc = build_dqi_circuit_algebraic(code, B, v, ell=ell)
                        qcm = build_dqi_circuit_algebraic(code, B, v, ell=ell,
                                                          with_measurements=True)
                    else:
                        qc = build_dqi_circuit_gje(B, v, ell=ell)
                        qcm = build_dqi_circuit_gje(B, v, ell=ell,
                                                    with_measurements=True)
                    nq = qc.num_qubits
                    iqm = iqm_stats(qc)
                    routed = iqm_routed_stats(qc)
                    rec.update({"qubits": nq, "iqm_cz": iqm["2q_gates"],
                                "iqm_depth": iqm["depth"],
                                "routed_cz": routed["2q_gates"],
                                "routed_depth": routed["depth"]})
                    if nq <= SIM_QUBIT_CAP:
                        mean, p = simulate_mean(qcm, m, n, B, v, opt)
                        rec.update({"simulated": True, "mean_satisfied": mean,
                                    "p_optimum": p})
                    else:
                        rec.update({"simulated": False, "mean_satisfied": None,
                                    "p_optimum": None,
                                    "note": "not simulated (qubits>%d)" % SIM_QUBIT_CAP})
                except Exception as e:
                    rec.update({"error": f"{type(e).__name__}: {e}"})
                rows.append(rec)
                tag = (f"mean={rec.get('mean_satisfied')} P(opt)={rec.get('p_optimum')}"
                       if rec.get("simulated") else rec.get("note", rec.get("error", "")))
                print(f"  [A {decoder:>12}] n={n} len={m} ell={ell} "
                      f"q={rec.get('qubits','?')} logicalCZ={rec.get('iqm_cz','?')} "
                      f"routedCZ={rec.get('routed_cz','?')}  {tag}")
    return rows


def part_b():
    try:
        from dqi_portfolio.portfolio import (
            build_universe, encode_portfolio_xorsat, dual_code_distance)
    except Exception as e:
        print(f"  PART B skipped: portfolio module unavailable ({e})")
        return []
    rows = []
    uni = build_universe()
    for k, red in [(6, 0.64), (7, 0.64)]:
        idx = list(range(k))
        su = {"tickers": [uni["tickers"][i] for i in idx],
              "sectors": [uni["sectors"][i] for i in idx],
              "sector_names": sorted(set(uni["sectors"][i] for i in idx)),
              "sec_idx": {}, "mu": uni["mu"][idx],
              "corr": uni["corr"][np.ix_(idx, idx)], "n": len(idx),
              "seed": uni["seed"], "label": "SUB"}
        enc = encode_portfolio_xorsat(su, div_thresh=-0.05, red_thresh=red,
                                      add_sector_parity=False, add_return_soft=False)
        B, v = enc["B"], enc["v"]
        m, n = B.shape
        opt = brute_force_opt(B, v)
        for ell in (1, 2, 3):
            for decoder in ("BP", "GJE"):
                rec = {"part": "B", "k": k, "decoder": decoder, "length": m,
                       "n": n, "ell": ell, "random_baseline": m / 2.0,
                       "brute_force_opt": opt}
                try:
                    if decoder == "BP":
                        qc = build_dqi_circuit(B, v, ell=ell, bp_iters=1)
                        qcm = build_dqi_circuit(B, v, ell=ell, bp_iters=1,
                                                with_measurements=True)
                    else:
                        qc = build_dqi_circuit_gje(B, v, ell=ell)
                        qcm = build_dqi_circuit_gje(B, v, ell=ell,
                                                    with_measurements=True)
                    nq = qc.num_qubits
                    iqm = iqm_stats(qc)
                    rec.update({"qubits": nq, "iqm_cz": iqm["2q_gates"],
                                "iqm_depth": iqm["depth"]})
                    if nq <= SIM_QUBIT_CAP:
                        mean, p = simulate_mean(qcm, m, n, B, v, opt)
                        rec.update({"simulated": True, "mean_satisfied": mean,
                                    "p_optimum": p})
                    else:
                        rec.update({"simulated": False, "mean_satisfied": None,
                                    "p_optimum": None, "note": "not simulated"})
                except Exception as e:
                    rec.update({"error": f"{type(e).__name__}: {e}"})
                rows.append(rec)
                print(f"  [B {decoder:>3} k={k}] ell={ell} q={rec.get('qubits','?')} "
                      f"CZ={rec.get('iqm_cz','?')} mean={rec.get('mean_satisfied')} "
                      f"P(opt)={rec.get('p_optimum')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/lumi/sweep_results_refactored")
    args = ap.parse_args()
    t0 = time.time()
    print("#" * 60 + "\n# PART A — decoder x size x ell (Table 2)\n" + "#" * 60)
    rows = part_a()
    print("#" * 60 + "\n# PART B — LDPC portfolio control\n" + "#" * 60)
    rows += part_b()
    keys = sorted({k for r in rows for k in r})
    with open(args.out + ".json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}.json/.csv  ({len(rows)} rows, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
