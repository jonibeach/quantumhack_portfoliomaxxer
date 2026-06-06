"""LUMI runner: load portable QASM3, run bare + EMS on Q50, dump RAW COUNTS.

Uses ONLY qiskit + iqm + fiqci.ems + stdlib (all in fiqci-vtt-qiskit-JQH) plus
the co-located ``iqm_runner`` helper (copied to scratch alongside this file).
No project imports. No interpretation here — counts are interpreted LOCALLY.
"""
import qiskit.qasm2 as q2

from iqm_runner import (
    connect_backend,
    pick_shots,
    bare_best_transpile,
    ems_best_transpile,
    backend_calibration,
    utc_now,
    write_raw,
)

# QASM2 (not QASM3): LUMI's qiskit lacks qiskit_qasm3_import so q3.load fails;
# qasm2.load is always available. Circuit is pure standard gates, so QASM2 is
# exact. Registers renamed (er/sol/an) to avoid qelib1.inc gate-name clashes.
circ = q2.load("q50_immunization_7bond.qasm2")  # measurements included

backend, url = connect_backend()

SHOTS = pick_shots(backend, 5000)

# ---- BARE run: multi-seed transpile against LIVE backend, pick lowest CZ ----
cz, tqc, seed, _sweep = bare_best_transpile(circ, backend, range(6))
print("[bare] routed cz=", cz, "depth=", tqc.depth(), "seed=", seed, "shots=", SHOTS)
job = backend.run(tqc, shots=SHOTS)
bare_job_id = getattr(job, "job_id", lambda: "?")()
print("[bare] job id=", bare_job_id)
bare_counts = job.result().get_counts()
print("[bare_counts]", bare_counts)

# ---- EMS run (mitigation_level=1) ----
ems_counts = None
ems_routed_cz = None
ems_depth = None
ems_job_id = None
ems_error = None
try:
    from fiqci.ems import FiQCIBackend
    ems = FiQCIBackend(backend, mitigation_level=1)
    ems_routed_cz, tqc_e, _ems_seed = ems_best_transpile(circ, ems)
    ems_depth = tqc_e.depth()
    print("[ems] routed cz=", ems_routed_cz, "depth=", ems_depth)
    job_e = ems.run(tqc_e, shots=SHOTS)
    ems_job_id = getattr(job_e, "job_id", lambda: "?")()
    print("[ems] job id=", ems_job_id)
    ems_result = job_e.result()
    ems_counts = ems_result.get_counts()
    print("[ems_counts]", ems_counts)
except Exception as e:
    import traceback
    ems_error = repr(e)
    print("[ems] FAILED:", ems_error)
    traceback.print_exc()

# ---- calibration snapshot ----
cal = backend_calibration(backend)

out = {
    "timestamp_utc": utc_now(),
    "shots": SHOTS,
    "bare_routed_cz": cz,
    "bare_depth": tqc.depth(),
    "bare_seed": seed,
    "bare_job_id": str(bare_job_id),
    "bare_counts": bare_counts,
    "ems_routed_cz": ems_routed_cz,
    "ems_depth": ems_depth,
    "ems_job_id": str(ems_job_id) if ems_job_id is not None else None,
    "ems_counts": ems_counts,
    "ems_error": ems_error,
    "calibration": cal,
    "qpu_url": url,
}
write_raw("q50_immunization_raw.json", out)
