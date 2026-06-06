"""Decoupled IBM submit / poll / fetch so nothing blocks indefinitely.

  submit  : transpile + submit all 4 circuits as SEPARATE jobs, print job ids,
            write artifacts/ibm/ibm_job_ids.json, exit immediately (no result()).
  status  : print status of every recorded job id.
  fetch   : for every DONE job, pull counts, score, write ibm_results_all.json.

Creds from env (IBM_API_KEY). Backend overridable via IBM_BACKEND env.
Usage:  set -a; . ./.env; set +a; .venv/bin/python artifacts/ibm/ibm_jobs.py submit
"""
import json
import os
import sys
from pathlib import Path

from qiskit import QuantumCircuit, transpile
import qiskit.qasm2 as q2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dqi_portfolio import dqi_satisfaction_stats  # noqa: E402
from dqi_portfolio.immunization import build_immunization_instance  # noqa: E402
from dqi_portfolio.readout import score_counts  # noqa: E402

OUT = Path(__file__).resolve().parent
IDS = OUT / "ibm_job_ids.json"
SHOTS = 4096


def svc():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform",
                               token=os.environ["IBM_API_KEY"])


def bell():
    qc = QuantumCircuit(2, name="bell")
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    return qc


CIRCUITS = [
    ("bell", lambda: bell(), "meas", None),
    ("collapsed7", lambda: q2.load(str(ROOT / "artifacts/iqm/binary/q50_immunization_binary.qasm2")), "c", (3, 1, 0)),
    ("bond31", lambda: q2.load(str(ROOT / "artifacts/iqm/binary_gf5/q50_immunization_binary_gf5.qasm2")), "c", (5, 1, 0)),
    ("full7", lambda: q2.load(str(ROOT / "artifacts/iqm/circuit/q50_immunization_7bond.qasm2")), "c", (3, 1, 7)),
]


def do_submit():
    s = svc()
    bname = os.environ.get("IBM_BACKEND")
    backend = s.backend(bname) if bname else s.least_busy(operational=True, simulator=False, min_num_qubits=11)
    print(f"[ibm] backend = {backend.name}")
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    rec = {"backend": backend.name, "shots": SHOTS, "jobs": {}}
    for name, mk, creg, _score in CIRCUITS:
        qc = mk()
        isa = transpile(qc, backend, optimization_level=3, seed_transpiler=5)
        ops = isa.count_ops(); twoq = ops.get("cz", 0) + ops.get("ecr", 0)
        job = sampler.run([isa], shots=SHOTS)
        rec["jobs"][name] = {"job_id": job.job_id(), "creg": creg, "routed_2q": twoq,
                             "depth": isa.depth()}
        print(f"[ibm] submitted {name}: id={job.job_id()} routed2q={twoq} depth={isa.depth()}")
    IDS.write_text(json.dumps(rec, indent=2))
    print(f"[ibm] wrote {IDS}")


def do_status():
    s = svc()
    rec = json.loads(IDS.read_text())
    for name, j in rec["jobs"].items():
        try:
            st = s.job(j["job_id"]).status()
        except Exception as e:
            st = f"err {str(e)[:60]}"
        print(f"{name:12} {j['job_id']:24} {st}")


def _score(name, counts, spec, creg):
    if spec is None:  # bell
        tot = sum(counts.values())
        insub = sum(c for k, c in counts.items() if k.replace(" ", "") in ("00", "11")) / tot
        return {"in_subspace": insub, "counts": counts}
    mf, t, off = spec
    B, v, code, _ = build_immunization_instance(mf, t)
    opt = dqi_satisfaction_stats(B, v, code, 1)["max_satisfied"]; m = B.shape[0]
    r = score_counts(counts, B, v, x_offset=off)
    p = r["hist"].get(opt, 0) / r["total"]
    rb = 1 / 2 ** (m.bit_length() if False else (3 if mf == 3 else 5))
    rb = 1 / 2 ** (B.shape[1])  # 2^n_syndrome
    return {"mean": r["mean"], "m": int(m), "opt": int(opt), "p_opt": p,
            "random_p": rb, "lift": p / rb, "counts": counts}


def do_fetch():
    s = svc()
    rec = json.loads(IDS.read_text())
    res = {"backend": rec["backend"], "shots": rec["shots"], "results": {}}
    for name, j in rec["jobs"].items():
        spec = dict((c[0], c[3]) for c in CIRCUITS)[name]
        creg = j["creg"]
        try:
            job = s.job(j["job_id"]); st = job.status()
            if str(st) not in ("DONE", "JobStatus.DONE"):
                print(f"{name}: not done ({st}) — skip"); continue
            counts = getattr(job.result()[0].data, creg).get_counts()
            sc = _score(name, counts, spec, creg)
            sc["routed_2q"] = j["routed_2q"]
            res["results"][name] = sc
            if "in_subspace" in sc:
                print(f"{name}: in-subspace={sc['in_subspace']:.4f} (routed2q={j['routed_2q']})")
            else:
                print(f"{name}: routed2q={j['routed_2q']} mean={sc['mean']:.3f}/{sc['m']} "
                      f"P(opt={sc['opt']})={sc['p_opt']:.4f} lift={sc['lift']:.2f}x")
        except Exception as e:
            print(f"{name}: fetch error {type(e).__name__}: {str(e)[:80]}")
    (OUT / "ibm_results_all.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"[ibm] wrote {OUT/'ibm_results_all.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"submit": do_submit, "status": do_status, "fetch": do_fetch}[cmd]()
