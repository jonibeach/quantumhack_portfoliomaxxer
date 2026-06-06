"""Shared Q50/LUMI runner mechanics for the q50_run_*.py scripts.

The three runners (7-bond, 3-qubit binary, 5-qubit binary GF(2^5)) all did the
same plumbing: open the live backend, pick a shot count, sweep transpiler seeds
keeping the lowest-CZ routing, snapshot calibration, and dump a raw-counts JSON.
That mechanics lives here once; each runner keeps its own orchestration/printing
(so the committed stdout/`*_raw.json` formats are unchanged).

DEPLOYMENT: this file is copied to the LUMI scratch dir ALONGSIDE the runner it
serves (see submit_q50_*.sh). It uses ONLY qiskit + iqm + stdlib — the exact set
available in the fiqci-vtt-qiskit-JQH module — and must NOT import dqi_portfolio
or numpy.
"""
import os
import json
import time

from qiskit import transpile
from iqm.qiskit_iqm import IQMProvider


def connect_backend(quantum_computer="q50", url_env="Q50_CORTEX_URL"):
    """Open the live backend and print the standard plumbing lines.

    Returns ``(backend, url)``; ``url`` is recorded in the result JSON.
    """
    url = os.environ[url_env]
    provider = IQMProvider(url, quantum_computer=quantum_computer)
    backend = provider.get_backend()
    print("[plumbing] backend.name =", backend.name)
    try:
        print("[plumbing] backend.num_qubits =", backend.num_qubits)
    except Exception as e:
        print("[plumbing] num_qubits err:", e)
    return backend, url


def pick_shots(backend, cap):
    """Shots = min(backend.max_shots or 2000, cap)."""
    return min(getattr(backend, "max_shots", 2000) or 2000, cap)


def bare_best_transpile(circ, backend, seeds):
    """Transpile against the live backend over ``seeds``; keep the lowest-CZ result.

    Returns ``(cz, tqc, seed, sweep)`` where ``sweep`` is a per-seed list of
    ``{seed, cz, depth}`` records (used by runners that log the full sweep; the
    others simply ignore it).
    """
    best = None
    sweep = []
    for seed in seeds:
        t = transpile(circ, backend=backend, optimization_level=3,
                      seed_transpiler=seed)
        cz = t.count_ops().get("cz", 0)
        sweep.append({"seed": seed, "cz": cz, "depth": t.depth()})
        if best is None or cz < best[0]:
            best = (cz, t, seed)
    cz, tqc, seed = best
    return cz, tqc, seed, sweep


def ems_best_transpile(circ, ems, seeds=None):
    """Transpile against the EMS backend; keep the lowest-CZ result.

    ``seeds=None`` does a single default-seed transpile (the 7-bond / 3-qubit
    runners); pass a ``range`` to sweep seeds (the 5-qubit runner). Returns
    ``(cz, tqc, seed)`` — ``seed`` is ``None`` in the single-transpile case.
    """
    if seeds is None:
        tqc = transpile(circ, ems, optimization_level=3)
        return tqc.count_ops().get("cz", 0), tqc, None
    best = None
    for s in seeds:
        te = transpile(circ, ems, optimization_level=3, seed_transpiler=s)
        cze = te.count_ops().get("cz", 0)
        if best is None or cze < best[0]:
            best = (cze, te, s)
    return best


def backend_calibration(backend):
    """Best-effort backend name/num_qubits snapshot for the result JSON."""
    cal = {}
    try:
        cal["backend_name"] = backend.name
        cal["num_qubits"] = backend.num_qubits
    except Exception as e:
        cal["err"] = str(e)
    return cal


def utc_now():
    """Current UTC timestamp in the runners' ISO-Z format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_raw(filename, out):
    """Dump the raw-result dict to ``filename`` and print the WROTE line."""
    with open(filename, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", filename)
