import json, numpy as np
from pathlib import Path

from dqi_portfolio.readout import score_counts

here = Path(__file__).parent
meta = json.load(open(here/"q50_immunization_binary_meta.json"))
raw  = json.load(open(here/"q50_binary_result_raw.json"))

B = np.array(meta["B"]) % 2
v = np.array(meta["v"]) % 2
n = meta["n"]; m = meta["m"]; opt = meta["optimum"]
xstar = np.array(meta["financial_readout"]["best_bitstring"])

def satisfied(x):
    return int(np.sum((B @ x) % 2 == v))

def analyze(counts, label):
    if counts is None:
        print(f"\n=== {label}: NONE ===")
        return
    # counts may be floats (EMS quasiprob) or ints; the 5-qubit solution register
    # is measured alone, so the shared readout reads x off offset 0.
    r = score_counts(counts, B, v, x_offset=0)
    total = r["total"]
    hist = {k: 0.0 for k in range(m + 1)}
    for s, w in r["hist"].items():
        hist[s] = w
    mean_sat = r["mean"]
    p_opt = r["hist"].get(opt, 0.0) / total
    sol_dist = {"".join(str(int(b)) for b in key): w
                for key, w in r["sol_hist"].items()}
    print(f"\n=== {label} (total weight={total:.1f}) ===")
    print(f"  mean satisfied = {mean_sat:.4f}   (random 3.5, sim 6.325)")
    print(f"  P(opt=7/7)     = {p_opt:.4f}   (random 0.125, sim 0.831)")
    print(f"  amplification factor over random = {p_opt/0.125:.2f}x")
    print("  satisfied-count histogram (counts):")
    for s in range(m+1):
        frac = hist[s]/total
        bar = "#"*int(frac*50)
        print(f"    {s}/7: {hist[s]:8.1f}  {frac*100:6.2f}%  {bar}")
    print("  solution-register x distribution (top):")
    for key, c in sorted(sol_dist.items(), key=lambda kv:-kv[1]):
        s = satisfied(np.array([int(b) for b in key]))
        star = " <-- x* (opt 7/7)" if key == "".join(str(b) for b in xstar) else ""
        print(f"    x={key}  sat={s}/7  weight={c:8.1f}  {c/total*100:6.2f}%{star}")
    return mean_sat, p_opt

print("optimum x* =", "".join(str(b) for b in xstar), " satisfies", satisfied(xstar), "/7")
print("bare routed cz =", raw["bare_routed_cz"], " depth =", raw["bare_depth"], " seed =", raw["bare_seed"])
print("bare layout physical qubits used =", raw["bare_layout"].get("physical_qubits_used"))
print("bare job id =", raw["bare_job_id"])
print("ems job id  =", raw.get("ems_job_id"), " ems routed cz =", raw.get("ems_routed_cz"))
analyze(raw["bare_counts"], "BARE (PRIMARY)")
analyze(raw.get("ems_counts"), "EMS mitigation_level=1 (SECONDARY)")
