"""Run the ORIGINAL full t=1 7-bond DQI circuit (the one that decohered to uniform
on Q50 at 154 routed CZ) on IBM Quantum, to see whether IBM's better 2q fidelity
lets the second-order interference survive where Q50's ~0.9% per-CZ killed it.

Circuit: artifacts/iqm/circuit/q50_immunization_7bond.qasm2  (11q: er[7]+sol[3]+an[1]).
Solution register = c[7:10]  ->  score with x_offset=7.

Usage:
  set -a; . ./.env; set +a
  .venv/bin/python artifacts/ibm/run_ibm_full.py [--transpile-only]
"""
import json
import os
import sys
from pathlib import Path

from qiskit import transpile
import qiskit.qasm2 as q2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dqi_portfolio import dqi_satisfaction_stats  # noqa: E402
from dqi_portfolio.immunization import build_immunization_instance  # noqa: E402
from dqi_portfolio.readout import score_counts  # noqa: E402

OUT = Path(__file__).resolve().parent
SHOTS = 4096
FULL_QASM = ROOT / "artifacts/iqm/circuit/q50_immunization_7bond.qasm2"


def connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ["IBM_API_KEY"]
    inst = os.environ.get("IBM_IAM_ID") or None
    for kw in ([{"instance": inst}] if inst else []) + [{}]:
        try:
            svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=token, **kw)
            svc.backends(operational=True)
            return svc
        except Exception as e:
            print(f"[ibm] connect attempt failed: {type(e).__name__}: {str(e)[:120]}")
    raise SystemExit("[ibm] could not connect")


def main():
    transpile_only = "--transpile-only" in sys.argv
    svc = connect()
    backend = svc.least_busy(operational=True, simulator=False, min_num_qubits=11)
    print(f"[ibm] backend = {backend.name}  nq={backend.num_qubits}")

    qc = q2.load(str(FULL_QASM))
    print(f"[full] logical: {qc.num_qubits}q ops={dict(qc.count_ops())}")
    best = None
    for seed in range(8):
        isa = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
        ops = isa.count_ops()
        twoq = ops.get("cz", 0) + ops.get("ecr", 0)
        if best is None or twoq < best[0]:
            best = (twoq, isa, seed)
    twoq, isa, seed = best
    print(f"[full] routed on {backend.name}: 2q(cz/ecr)={twoq} depth={isa.depth()} "
          f"(best of 8 seeds, seed={seed})  [Q50 was 154 routed CZ]")
    if transpile_only:
        return

    B7, v7, code7, _ = build_immunization_instance(3, 1)
    opt = dqi_satisfaction_stats(B7, v7, code7, 1)["max_satisfied"]
    m = B7.shape[0]

    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run([isa], shots=SHOTS)
    print(f"[ibm] job id = {job.job_id()}  waiting ...")
    counts = job.result()[0].data.c.get_counts()

    r = score_counts(counts, B7, v7, x_offset=m)  # solution reg = c[7:10]
    p_opt = r["hist"].get(opt, 0) / r["total"]
    # per-pattern solution distribution (8 strings) to see uniform vs peaked
    sol = {}
    for k, c in counts.items():
        bits = k.replace(" ", "")[::-1]
        x = bits[m:m + 3]
        sol[x] = sol.get(x, 0) + c
    tot = sum(sol.values())
    soldist = {k: round(v / tot, 3) for k, v in sorted(sol.items())}

    print(f"\n[FULL-7 on {backend.name}] routed 2q={twoq}, {SHOTS} shots")
    print(f"  mean satisfied = {r['mean']:.3f}/{m}  (random {m/2})")
    print(f"  P(opt={opt}/{m}) = {p_opt:.4f}  (random {1/8:.4f}, lift {p_opt/(1/8):.2f}x)")
    print(f"  3-bit solution-register distribution: {soldist}")
    print("  (Q50 ran this at 154 CZ -> decohered to uniform, P(opt)~0.122)")

    out = {"backend": backend.name, "routed_2q": int(twoq), "depth": int(isa.depth()),
           "shots": SHOTS, "job_id": job.job_id(), "mean": r["mean"],
           "p_opt": p_opt, "opt": int(opt), "m": int(m), "random_p": 1 / 8,
           "solution_distribution": soldist, "counts": counts}
    (OUT / "ibm_full_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[ibm] wrote {OUT/'ibm_full_results.json'}")


if __name__ == "__main__":
    main()
