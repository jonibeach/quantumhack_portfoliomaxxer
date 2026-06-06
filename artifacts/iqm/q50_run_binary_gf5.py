"""LUMI runner: load the 5-qubit BINARY GF(2^5) immunization circuit (QASM2),
run bare + (optional) EMS on Q50, dump RAW COUNTS to JSON.

Uses ONLY qiskit + iqm + fiqci.ems + stdlib (all in fiqci-vtt-qiskit-JQH) plus
the co-located ``iqm_runner`` helper (copied to scratch alongside this file).
No project imports (dqi_portfolio does NOT import under this module).
Counts are interpreted LOCALLY afterward.

This is the SCALED-UP sibling of the 7-bond run: 5 qubits => LAYOUT matters.
We transpile against the FULL live backend over a 16-seed sweep and keep the
lowest-CZ result, so the mapper can pick a compact ring-like patch (~34 CZ,
under the 40-CZ wall) instead of a bare line (~47 CZ).
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
# register (5 qubits, the syndrome register) + creg c[5].
circ = q2.load("q50_immunization_binary_gf5.qasm2")  # measurements included
print("[circ] qubits=", circ.num_qubits, "ops=", circ.count_ops())

backend, url = connect_backend()

SHOTS = pick_shots(backend, 2000)


def capture_layout(tqc):
    """Return the REAL device physical qubits for each virtual qubit.

    Prefers initial_layout (true device qubit numbers); falls back to the
    relabeled find_bit indices only if initial_layout is unavailable.
    """
    info = {}
    try:
        fl = tqc.layout
        if fl is not None:
            try:
                il = fl.initial_layout
                vbits = il.get_virtual_bits()  # {Qubit: phys_index}
                # virtual qubit position -> device physical qubit
                mapping = {}
                for vq, phys in vbits.items():
                    idx = getattr(vq, "_index", None)
                    if idx is None:
                        try:
                            idx = circ.find_bit(vq).index
                        except Exception:
                            idx = str(vq)
                    mapping[str(idx)] = int(phys)
                info["initial_layout_virt_to_phys"] = mapping
                info["initial_physical_qubits"] = sorted(set(int(p) for p in vbits.values()))
            except Exception as e:
                info["initial_err"] = repr(e)
            # final layout (after routing swaps), if present
            try:
                fin = fl.final_index_layout()
                info["final_index_layout"] = list(fin)
            except Exception as e:
                info["final_err"] = repr(e)
    except Exception as e:
        info["layout_err"] = repr(e)
    # relabeled indices actually carrying ops (NOT the real device qubits, but
    # recorded for comparison with the prior 7-bond run's field)
    try:
        used = sorted({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits})
        info["relabeled_qubits_used"] = used
    except Exception as e:
        info["used_err"] = repr(e)
    return info


# ---- BARE run: 16-seed transpile against LIVE backend, pick lowest CZ ----
cz, tqc, seed, sweep = bare_best_transpile(circ, backend, range(16))
print("[bare] seed sweep:", sweep)
print("[bare] CHOSEN routed cz=", cz, "depth=", tqc.depth(), "seed=", seed, "shots=", SHOTS)

layout_info = capture_layout(tqc)
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
    # seed sweep on EMS too, keep lowest CZ
    ems_routed_cz, tqc_e, ems_seed = ems_best_transpile(circ, ems, seeds=range(16))
    ems_depth = tqc_e.depth()
    print("[ems] routed cz=", ems_routed_cz, "depth=", ems_depth, "seed=", ems_seed)
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
    "circuit": "q50_immunization_binary_gf5.qasm2",
    "shots": SHOTS,
    "bare_routed_cz": cz,
    "bare_depth": tqc.depth(),
    "bare_seed": seed,
    "bare_seed_sweep": sweep,
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
write_raw("q50_binary_gf5_result_raw.json", out)
