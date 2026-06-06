"""LUMI runner: load the 3-qubit BINARY immunization circuit (QASM2),
run bare + (optional) EMS on Q50, dump RAW COUNTS to JSON.

Uses ONLY qiskit + iqm + fiqci.ems + stdlib (all in fiqci-vtt-qiskit-JQH) plus
the co-located ``iqm_runner`` helper (copied to scratch alongside this file).
No project imports (dqi_portfolio does NOT import under this module).
Counts are interpreted LOCALLY afterward.
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

# QASM2 only (LUMI qiskit lacks qiskit_qasm3_import). Circuit uses a `sol`
# register (3 qubits, the syndrome register) + creg c[3].
circ = q2.load("q50_immunization_binary.qasm2")  # measurements included
print("[circ] qubits=", circ.num_qubits, "ops=", circ.count_ops())

backend, url = connect_backend()

SHOTS = pick_shots(backend, 2000)

# ---- BARE run: multi-seed transpile against LIVE backend, pick lowest CZ ----
cz, tqc, seed, _sweep = bare_best_transpile(circ, backend, range(8))
print("[bare] routed cz=", cz, "depth=", tqc.depth(), "seed=", seed, "shots=", SHOTS)

# capture physical layout
layout_info = {}
try:
    fl = tqc.layout
    if fl is not None:
        # initial physical qubits used
        try:
            il = fl.initial_layout
            layout_info["initial_layout"] = {str(k): v for k, v in il.get_virtual_bits().items()}
        except Exception as e:
            layout_info["initial_err"] = str(e)
        try:
            phys = sorted({q._index if hasattr(q, "_index") else q for q in []})
        except Exception:
            pass
    # active physical qubits via the dag/qubit indices
    used = sorted({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits})
    layout_info["physical_qubits_used"] = used
except Exception as e:
    layout_info["err"] = str(e)
print("[bare] layout=", layout_info)

job = backend.run(tqc, shots=SHOTS)
bare_job_id = getattr(job, "job_id", lambda: "?")()
print("[bare] job id=", bare_job_id)
bare_counts = job.result().get_counts()
print("[bare_counts]", bare_counts)

# ---- EMS run (mitigation_level=1) — SECONDARY ----
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

cal = backend_calibration(backend)

out = {
    "timestamp_utc": utc_now(),
    "circuit": "q50_immunization_binary.qasm2",
    "shots": SHOTS,
    "bare_routed_cz": cz,
    "bare_depth": tqc.depth(),
    "bare_seed": seed,
    "bare_layout": layout_info,
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
write_raw("q50_binary_result_raw.json", out)
