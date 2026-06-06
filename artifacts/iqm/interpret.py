"""LOCAL interpretation of Q50 raw counts with the validated readout.

Run: uv run python artifacts/iqm/interpret.py
"""
import json
import numpy as np

BASE = "/Users/joni/personal/hackathons/quantumhack/artifacts/iqm"
meta = json.load(open(f"{BASE}/circuit/q50_immunization_meta.json"))
raw = json.load(open(f"{BASE}/q50_immunization_raw.json"))

B = np.array(meta["B"]) % 2
v = np.array(meta["v"]) % 2
m, n = meta["m"], meta["n"]
OPT = meta["optimum"]  # 7


def interpret(counts):
    """Same readout as the validated simulator path. Returns (mean, p_opt, hist, tot).

    counts values may be floats (EMS quasi-probabilities). We weight by value and
    clip negatives to 0 for the histogram/normalisation (standard for a quasi-prob
    readout); total uses the clipped weights.
    """
    tot = 0.0
    wt = 0.0
    hist = {}
    for bs, c in counts.items():
        c = float(c)
        if c <= 0:
            continue
        bits = bs.replace(" ", "")[::-1]
        x = np.array([int(bits[m + i]) for i in range(n)])
        s = int(np.sum((B @ x) % 2 == v))
        hist[s] = hist.get(s, 0.0) + c
        tot += c
        wt += s * c
    mean = wt / tot if tot else 0.0
    p_opt = hist.get(OPT, 0.0) / tot if tot else 0.0
    return mean, p_opt, {k: hist[k] / tot for k in sorted(hist)}, tot


bare_mean, bare_popt, bare_hist, bare_tot = interpret(raw["bare_counts"])
ems_mean, ems_popt, ems_hist, ems_tot = interpret(raw["ems_counts"])

rand_mean = meta["random_mean_satisfied"]
rand_popt = meta["random_p_opt"]
sim_mean = meta["simulated_mean_satisfied"]
sim_popt = meta["simulated_p_opt"]

print("=== Q50 immunization interpretation ===")
print(f"bare: mean={bare_mean:.4f}/7  P(opt)={bare_popt:.4f}  (sum_w={bare_tot:.1f})")
print(f"   hist={ {k: round(bare_hist[k],4) for k in bare_hist} }")
print(f"ems : mean={ems_mean:.4f}/7  P(opt)={ems_popt:.4f}  (sum_w={ems_tot:.1f})")
print(f"   hist={ {k: round(ems_hist[k],4) for k in ems_hist} }")
print(f"random: mean={rand_mean:.4f}/7  P(opt)={rand_popt:.4f}")
print(f"sim   : mean={sim_mean:.4f}/7  P(opt)={sim_popt:.4f}")

results = {
    "job_slurm": "19082355",
    "iqm_job_bare": raw["bare_job_id"],
    "iqm_job_ems": raw["ems_job_id"],
    "timestamp_utc": raw["timestamp_utc"],
    "shots": raw["shots"],
    "bare_routed_cz": raw["bare_routed_cz"],
    "bare_depth": raw["bare_depth"],
    "bare_seed": raw["bare_seed"],
    "ems_routed_cz": raw["ems_routed_cz"],
    "ems_depth": raw["ems_depth"],
    "calibration": raw["calibration"],
    "qpu_url": raw["qpu_url"],
    "table": {
        "random":    {"mean_over_7": rand_mean, "p_opt": rand_popt},
        "simulator": {"mean_over_7": sim_mean,  "p_opt": sim_popt},
        "q50_bare":  {"mean_over_7": bare_mean, "p_opt": bare_popt},
        "q50_ems":   {"mean_over_7": ems_mean,  "p_opt": ems_popt},
    },
    "bare_satisfied_histogram": bare_hist,
    "ems_satisfied_histogram": ems_hist,
    "ems_note": ("EMS get_counts returned quasi-probability weights (floats, "
                 "many zero/negative from error-mitigation post-processing). "
                 "mean/P(opt) computed by weighting bitstrings by their positive "
                 "EMS weight; negatives clipped to 0."),
}
with open(f"{BASE}/q50_immunization_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nwrote q50_immunization_results.json")
