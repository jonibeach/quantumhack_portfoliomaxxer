"""Run the DQI immunization hardware circuits on IBM Quantum (IBM Cloud platform).

Submits three circuits as one SamplerV2 job (single queue wait):
  1. Bell smoke   -> in-subspace fidelity (per-2q-gate health check)
  2. collapsed 7-bond  (3q, t=1 bijection)   -> P(opt), mean satisfied
  3. 31-bond GF(2^5)   (5q, t=1 bijection)   -> P(opt), mean satisfied

Credentials come from the environment (IBM_API_KEY, optional IBM_IAM_ID as the
instance); they are never printed. Scores reuse the project scorer so the IBM
numbers are computed identically to the LUMI/Q50 runs.

Run:  set -a; . ./.env; set +a; .venv/bin/python artifacts/ibm/run_ibm.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2 as q2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dqi_portfolio import dqi_satisfaction_stats  # noqa: E402
from dqi_portfolio.immunization import build_immunization_instance  # noqa: E402
from dqi_portfolio.readout import score_counts  # noqa: E402

OUT = Path(__file__).resolve().parent
SHOTS = 4096


def connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ["IBM_API_KEY"]
    instance = os.environ.get("IBM_IAM_ID") or None
    last = None
    for kwargs in ({"instance": instance} if instance else {}, {}):
        try:
            svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=token, **kwargs)
            # force a call that needs the instance to be resolved
            _ = svc.backends(operational=True)
            print(f"[ibm] connected (instance={'set' if kwargs.get('instance') else 'auto'})")
            return svc
        except Exception as e:
            last = e
            print(f"[ibm] connect attempt failed: {type(e).__name__}: {str(e)[:160]}")
    raise SystemExit(f"[ibm] could not connect: {last}")


def bell():
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def main():
    svc = connect()
    backend = svc.least_busy(operational=True, simulator=False, min_num_qubits=5)
    print(f"[ibm] backend = {backend.name}  nq={backend.num_qubits}")

    # instances for scoring (deterministic, seed=0) — must match the baked QASM
    B7, v7, code7, _ = build_immunization_instance(3, 1)
    B31, v31, code31, _ = build_immunization_instance(5, 1)
    opt7 = dqi_satisfaction_stats(B7, v7, code7, 1)["max_satisfied"]
    opt31 = dqi_satisfaction_stats(B31, v31, code31, 1)["max_satisfied"]

    qc_bell = bell()
    qc7 = q2.load(str(ROOT / "artifacts/iqm/binary/q50_immunization_binary.qasm2"))
    qc31 = q2.load(str(ROOT / "artifacts/iqm/binary_gf5/q50_immunization_binary_gf5.qasm2"))

    jobs = [("bell", qc_bell, "meas"), ("collapsed7", qc7, "c"),
            ("bond31", qc31, "c")]
    isas = []
    for name, qc, creg in jobs:
        isa = transpile(qc, backend, optimization_level=3)
        cz = isa.count_ops().get("cz", 0) + isa.count_ops().get("ecr", 0)
        print(f"[ibm] {name}: routed 2q={cz} depth={isa.depth()} creg={creg}")
        isas.append(isa)

    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isas, shots=SHOTS)
    print(f"[ibm] job id = {job.job_id()}  submitted, waiting for result ...")
    res = job.result()

    def get_counts(pub, creg):
        return getattr(pub.data, creg).get_counts()

    out = {"backend": backend.name, "shots": SHOTS, "job_id": job.job_id()}

    # ---- Bell ----
    bc = get_counts(res[0], "meas")
    tot = sum(bc.values())
    insub = sum(c for k, c in bc.items() if k.replace(" ", "") in ("00", "11")) / tot
    print(f"\n[BELL] counts={bc}")
    print(f"[BELL] in-subspace (00+11) fraction = {insub:.4f}  -> per-CZ health")
    out["bell"] = {"counts": bc, "in_subspace": insub}

    # ---- collapsed 7-bond ----
    c7 = get_counts(res[1], "c")
    r7 = score_counts(c7, B7, v7, x_offset=0)
    p7 = r7["hist"].get(opt7, 0) / r7["total"]
    m7 = B7.shape[0]
    print(f"\n[COLLAPSED-7] counts={c7}")
    print(f"[COLLAPSED-7] mean satisfied = {r7['mean']:.3f}/{m7} (random {m7/2})")
    print(f"[COLLAPSED-7] P(opt={opt7}/{m7}) = {p7:.4f}  (random {1/2**3:.4f}, "
          f"lift {p7/(1/2**3):.2f}x)")
    out["collapsed7"] = {"counts": c7, "mean": r7["mean"], "p_opt": p7,
                         "opt": int(opt7), "m": int(m7),
                         "random_p": 1 / 2 ** 3}

    # ---- 31-bond ----
    c31 = get_counts(res[2], "c")
    r31 = score_counts(c31, B31, v31, x_offset=0)
    p31 = r31["hist"].get(opt31, 0) / r31["total"]
    m31 = B31.shape[0]
    print(f"\n[BOND-31] counts={c31}")
    print(f"[BOND-31] mean satisfied = {r31['mean']:.3f}/{m31} (random {m31/2})")
    print(f"[BOND-31] P(opt={opt31}/{m31}) = {p31:.4f}  (random {1/2**5:.4f}, "
          f"lift {p31/(1/2**5):.2f}x)")
    out["bond31"] = {"counts": c31, "mean": r31["mean"], "p_opt": p31,
                     "opt": int(opt31), "m": int(m31), "random_p": 1 / 2 ** 5}

    (OUT / "ibm_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[ibm] wrote {OUT/'ibm_results.json'}")


if __name__ == "__main__":
    main()
