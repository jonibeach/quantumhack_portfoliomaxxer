"""DQI immunization web-app bridge CLI.

Subcommands (each prints ONE JSON object to stdout; errors -> {"error": ...} + exit 1):

  instance  --size {7,15,31} --liability I [--maturities csv]
            Build the immunization max-XORSAT instance (B, v) for the chosen bond
            ladder + liability and return the bond<->maturity<->locator table,
            optimum, and the random baseline. No circuit, no hardware.

  simulate  --size --liability [--maturities] [--shots N]
            Build the collapsed t=1 DQI circuit, run it on AerSimulator, and return
            mean satisfied / P(opt) / amplification + the decoded immunizing bond
            subset + the per-solution distribution. Instant, free, unlimited.

  submit    --size --liability [--maturities] [--shots 4096] [--backend ...]
            Transpile the SAME circuit for an IBM backend and submit it. Returns
            {job_id, backend, routed_2q, depth, shots} immediately (non-blocking).

  result    --job-id ID --size --liability [--maturities]
            Poll an IBM job. If not done -> {"status": ...}. If done -> the full
            interpreted hardware result (scored against the instance re-derived
            from --size/--liability), with lift vs random.

The hardware circuit is the binary t=1 "collapse" (build_minimal): for a single
error over GF(2) the location->syndrome map is a bijection, so the whole DQI
computation folds into the m_field-qubit syndrome register (3/4/5 qubits for
size 7/15/31) and stays under the hardware coherence wall. See
scripts/immunization_binary.py and dqi_portfolio/immunization.py for the theory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Import the existing quantum code. cwd is the repo root (set by the web server),
# so dqi_portfolio + scripts resolve. The package __init__ guards the BP/GJE
# baselines, so this works without the external/ submodules.
from dqi_portfolio.dqi_algebraic import (
    build_bch_instance,
    dqi_satisfaction_stats,
    _optimal_poly_weights,
)
from dqi_portfolio.immunization import BondLadder, decode_portfolio
from dqi_portfolio.readout import score_counts
from scripts._immun_circuit import (
    build_amplitude_vector,
    v_phase_positions,
    build_minimal,
)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation


# size (number of bonds) -> GF(2^m_field) field order. n = 2^m - 1.
SIZE_TO_MFIELD = {7: 3, 15: 4, 31: 5}
DEFAULT_BACKEND = os.environ.get("IBM_BACKEND", "ibm_marrakesh")


# ---------------------------------------------------------------------------
# JSON helpers.
# ---------------------------------------------------------------------------
def _jsonable(obj):
    """Recursively coerce numpy scalars/arrays into plain JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _emit(obj):
    sys.stdout.write(json.dumps(_jsonable(obj)))
    sys.stdout.flush()


def _fail(msg, code=1):
    sys.stdout.write(json.dumps({"error": str(msg)}))
    sys.stdout.flush()
    sys.exit(code)


# ---------------------------------------------------------------------------
# Instance construction (B fixed by the code; v derived from the liability).
# ---------------------------------------------------------------------------
def _mfield(size):
    if size not in SIZE_TO_MFIELD:
        _fail(f"size must be one of {sorted(SIZE_TO_MFIELD)} (got {size})")
    return SIZE_TO_MFIELD[size]


def _liability_v(code, liability_index):
    """Target syndrome v for a weight-1 'ideal' liability matched by one bond.

    Mirrors dqi_portfolio.immunization._liability_syndrome for t=1, but with a
    USER-chosen ideal bond instead of a random one: y_L = e_{liability_index},
    s_L = (H y_L) % 2, v = (H^T s_L) % 2. B is unchanged; only v (the liability)
    varies. The optimum solution is then x = s_L (satisfies all m constraints).
    """
    n = code.n
    if not (0 <= liability_index < n):
        _fail(f"liability index must be in [0,{n}) for size {n} (got {liability_index})")
    H = np.asarray(code.H, dtype=int) % 2
    y_L = np.zeros(n, dtype=int)
    y_L[liability_index] = 1
    s_L = (H @ y_L) % 2
    v = (H.T @ s_L) % 2
    return v.astype(int)


def _make_ladder(m_field, maturities):
    """BondLadder whose maturities are the user's real selection (sorted), if given.

    Locators stay alpha^i (the standard RS/BCH evaluation points); only the
    financial labelling (maturities) is replaced. Falls back to the synthetic
    ladder when --maturities is omitted.
    """
    ladder = BondLadder(m_field=m_field)
    if maturities:
        mats = sorted(float(x) for x in maturities)
        if len(mats) != ladder.n:
            _fail(f"need exactly {ladder.n} maturities for size {ladder.n} "
                  f"(got {len(mats)})")
        ladder.maturities = np.asarray(mats, dtype=float)
    return ladder


def _build_instance(size, liability_index, maturities):
    """Return (B, v, code, ladder, m, n_syn) for the chosen size + liability."""
    m_field = _mfield(size)
    B, _v_rand, code = build_bch_instance(m_field, 1)
    v = _liability_v(code, liability_index)
    ladder = _make_ladder(m_field, maturities)
    m, n_syn = B.shape  # m positions (= size), n_syn syndrome bits (= m_field)
    return B, v, code, ladder, m, n_syn


def _measured_circuit(B, v):
    """Build the collapsed t=1 DQI circuit with a classical register named 'c'.

    Gate content (StatePreparation(a) then H^n) is identical to
    scripts._immun_circuit.build_minimal — the validated hardware circuit; we only
    fix the classical-register name to 'c' so counts read back uniformly from both
    AerSimulator and IBM SamplerV2 (data.c.get_counts()).
    """
    m, n_syn = B.shape
    w = _optimal_poly_weights(m, 1)            # ell=1 optimal weights (uniform)
    a, _syndromes = build_amplitude_vector(B, v, w)
    qr = QuantumRegister(n_syn, "q")
    cr = ClassicalRegister(n_syn, "c")
    qc = QuantumCircuit(qr, cr)
    qc.append(StatePreparation(a), list(qr))
    qc.h(list(qr))
    qc.measure(qr, cr)
    return qc, a


def _bond_table(ladder, n):
    return [
        {
            "bond": int(i),
            "maturity": float(ladder.maturities[i]),
            "locator": int(ladder.locators[i]),
        }
        for i in range(n)
    ]


def _score(counts, B, v, code, ladder):
    """Interpret a counts dict into the satisfied-count + financial readout."""
    m, n_syn = B.shape
    opt = int(dqi_satisfaction_stats(B, v, code, 1)["max_satisfied"])
    r = score_counts(counts, B, v, x_offset=0)
    total = r["total"]
    hist = {int(k): r["hist"][k] / total for k in r["hist"]} if total else {}
    p_opt = hist.get(opt, 0.0)
    random_p = 1.0 / (1 << n_syn)
    # per-solution distribution keyed by solution integer (LSB-first, matches a/syndrome).
    sol_dist = {}
    for x_tuple, c in r["sol_hist"].items():
        xi = int(sum(int(b) << i for i, b in enumerate(x_tuple)))
        sol_dist[xi] = sol_dist.get(xi, 0.0) + (c / total if total else 0.0)
    # most-probable best solution -> decode the immunizing bond subset.
    best_x = max(
        r["sol_hist"],
        key=lambda x: (int(np.sum((np.asarray(B) % 2 @ np.array(x)) % 2 == (np.asarray(v) % 2))),
                       r["sol_hist"][x]),
    )
    best_score = int(np.sum((np.asarray(B) % 2 @ np.array(best_x)) % 2 == (np.asarray(v) % 2)))
    readout = decode_portfolio(code, ladder, np.array(best_x))
    return {
        "mean": float(r["mean"]),
        "m": int(m),
        "n_syn": int(n_syn),
        "optimum": opt,
        "p_opt": float(p_opt),
        "random_mean": float(m / 2),
        "random_p": float(random_p),
        "lift": float(p_opt / random_p) if random_p else None,
        "satisfied_hist": {str(k): float(val) for k, val in sorted(hist.items())},
        "solution_dist": {str(k): float(val) for k, val in sorted(sol_dist.items())},
        "shots": int(total),
        "best_solution": [int(b) for b in best_x],
        "best_satisfies": best_score,
        "decoded": _jsonable(readout),
    }


# ---------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------
def cmd_instance(args):
    B, v, code, ladder, m, n_syn = _build_instance(
        args.size, args.liability, args.maturities)
    classical = dqi_satisfaction_stats(B, v, code, 1)
    _emit({
        "size": int(args.size),
        "m_field": int(n_syn),
        "n_positions": int(m),
        "liability_index": int(args.liability),
        "optimum": int(classical["max_satisfied"]),
        "classical_mean_satisfied": float(classical["mean_satisfied"]),
        "random_mean": float(m / 2),
        "random_p": float(1.0 / (1 << n_syn)),
        "B": _jsonable(np.asarray(B) % 2),
        "v": _jsonable(np.asarray(v) % 2),
        "bonds": _bond_table(ladder, m),
    })


def cmd_simulate(args):
    from qiskit_aer import AerSimulator
    from dqi_portfolio.bases import SIM_BASIS

    B, v, code, ladder, m, n_syn = _build_instance(
        args.size, args.liability, args.maturities)
    qc, _a = _measured_circuit(B, v)
    sim = AerSimulator()
    qct = transpile(qc, sim, basis_gates=SIM_BASIS)
    counts = sim.run(qct, shots=args.shots).result().get_counts()
    out = _score(counts, B, v, code, ladder)
    out.update({
        "mode": "simulator",
        "size": int(args.size),
        "liability_index": int(args.liability),
        "bonds": _bond_table(ladder, m),
    })
    _emit(out)


def _ibm_service():
    token = os.environ.get("IBM_API_KEY")
    if not token:
        _fail("IBM_API_KEY not set in environment")
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)


def cmd_submit(args):
    B, v, code, ladder, m, n_syn = _build_instance(
        args.size, args.liability, args.maturities)
    qc, _a = _measured_circuit(B, v)

    svc = _ibm_service()
    backend = svc.backend(args.backend)
    # best-of-seeds transpile, keep lowest 2q (cz/ecr) routing.
    best = None
    for seed in range(args.seeds):
        isa = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
        ops = isa.count_ops()
        twoq = ops.get("cz", 0) + ops.get("ecr", 0)
        if best is None or twoq < best[0]:
            best = (twoq, isa, seed)
    twoq, isa, seed = best

    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run([isa], shots=args.shots)
    _emit({
        "job_id": job.job_id(),
        "backend": backend.name,
        "routed_2q": int(twoq),
        "depth": int(isa.depth()),
        "shots": int(args.shots),
        "size": int(args.size),
        "liability_index": int(args.liability),
        "seed": int(seed),
    })


def cmd_result(args):
    B, v, code, ladder, m, n_syn = _build_instance(
        args.size, args.liability, args.maturities)
    svc = _ibm_service()
    job = svc.job(args.job_id)
    status = str(job.status())
    done = status in ("DONE", "JobStatus.DONE")
    if not done:
        _emit({"status": status, "done": False, "job_id": args.job_id})
        return
    counts = job.result()[0].data.c.get_counts()
    out = _score(counts, B, v, code, ladder)
    out.update({
        "mode": "hardware",
        "status": status,
        "done": True,
        "job_id": args.job_id,
        "backend": getattr(job, "backend", lambda: None)().name
            if callable(getattr(job, "backend", None)) else None,
        "size": int(args.size),
        "liability_index": int(args.liability),
        "bonds": _bond_table(ladder, m),
    })
    _emit(out)


# ---------------------------------------------------------------------------
# Argparse.
# ---------------------------------------------------------------------------
def _add_common(p, with_shots=True):
    p.add_argument("--size", type=int, required=True, choices=sorted(SIZE_TO_MFIELD))
    p.add_argument("--liability", type=int, default=0,
                   help="index of the 'ideal' liability bond (0-based)")
    p.add_argument("--maturities", type=str, default=None,
                   help="comma-separated real maturities (years); must match size")
    if with_shots:
        p.add_argument("--shots", type=int, default=4096)


def _parse_maturities(args):
    if args.maturities:
        args.maturities = [s for s in args.maturities.split(",") if s.strip()]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(prog="webapp_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inst = sub.add_parser("instance")
    _add_common(p_inst, with_shots=False)

    p_sim = sub.add_parser("simulate")
    _add_common(p_sim)
    p_sim.set_defaults(shots=16384)

    p_sub = sub.add_parser("submit")
    _add_common(p_sub)
    p_sub.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
    p_sub.add_argument("--seeds", type=int, default=6)

    p_res = sub.add_parser("result")
    _add_common(p_res, with_shots=False)
    p_res.add_argument("--job-id", dest="job_id", type=str, required=True)

    args = _parse_maturities(parser.parse_args(argv))
    try:
        {
            "instance": cmd_instance,
            "simulate": cmd_simulate,
            "submit": cmd_submit,
            "result": cmd_result,
        }[args.cmd](args)
    except SystemExit:
        raise
    except Exception as e:  # any failure -> structured JSON error
        _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
