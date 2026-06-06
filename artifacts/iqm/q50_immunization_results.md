# Q50 7-bond DQI immunization — hardware run results

**Instance:** 7-bond fixed-income immunization as a binary-BCH DQI max-XORSAT
instance (GF(2^3), t=1, ell=1). 7 constraints, n=3 solution variables.
Circuit: `build_dqi_circuit_algebraic(..., ell=1, use_ancilla=True,
mcx_mode="v-chain")` — 11 qubits, 86 IQM-CZ (local estimate).

**Backend:** VTT Q50 (`IQMBackend`, 53 qubits) via FiQCI on LUMI, URL
`https://qx.vtt.fi/`.

## Pipeline self-check (local, AerSimulator)
Reproduced the known-good simulated numbers with the validated readout
(reverse measured bitstring, take solution register `x = bits[7:10]`,
satisfied = `sum((B@x)%2 == v)`):
- **mean satisfied = 6.32/7**, **P(opt=7/7) = 0.830** (targets 6.34 / 0.836). PASS.
- Shipped QASM2 re-loaded + re-simulated to the same P(opt). PASS.

## Baselines (brute force over all 2^3 = 8 assignments of x)
- Brute-force optimum: **7/7**, with **1** optimal assignment.
- Random mean satisfied: **3.50/7**.
- Random P(opt): **1/8 = 0.125**.

## THE RESULT TABLE

| source            | mean / 7 | P(opt = 7/7) |
|-------------------|----------|--------------|
| random baseline   | 3.500    | 0.125        |
| **simulator**     | 6.320    | **0.830**    |
| **Q50 bare**      | 3.486    | **0.122**    |
| **Q50 + EMS**     | 3.546    | **0.136**    |

Hardware histograms (satisfied-count -> fraction). Every measured string maps to
satisfied-count 3 or 7 (the n=3 readout admits only these two values):
- **Q50 bare** (2000 shots): {3: 0.879, 7: 0.121}
- **Q50 + EMS** (sum of positive quasi-prob weights 1466): {3: 0.864, 7: 0.136}

## Run metadata
- Slurm job: **19082355** (partition debug, reservation JQH2026). Queue wait ~0;
  started RUNNING immediately, total wall < 2 min.
- IQM QPU job (bare): `7e30aa29-f24f-47a5-8941-3302e3975905`
- IQM QPU job (EMS):  `ac03217c-4173-4b94-b63b-ceda17f8d0c5`
- Shots: 2000 (backend `max_shots`).
- Bare: routed **CZ = 154**, depth **226**, transpile seed 3 (best of 6).
- EMS (mitigation_level=1): routed **CZ = 167**, depth **256**.
- Calibration snapshot: backend `IQMBackend`, 53 qubits.

## Verdict (honest)

**DQI did NOT amplify on real hardware.** Q50 bare landed at the random baseline:
mean 3.49/7 vs random 3.50/7, and P(opt) 0.122 vs random 0.125 — statistically
indistinguishable from uniform. The simulator's strong amplification (P(opt) =
0.83, ~6.6x over random) was **completely washed out by decoherence**: the routed
circuit is 154 CZ at depth 226 on a NISQ device, far past the coherence budget for
this 11-logical-qubit DQI circuit, so the output collapsed to noise.

**EMS helped only marginally and did not recover amplification.** With FiQCI
mitigation_level=1 the mean rose to 3.55/7 and P(opt) to 0.136 — a hair above
random but nowhere near the 0.83 target. (EMS `get_counts` returns
quasi-probability weights; mean/P(opt) computed by weighting bitstrings by their
positive EMS weight, negatives clipped.) The lift is within noise of the random
baseline and is not evidence of recovered signal.

**Bottom line:** the circuit, readout, and access path are all validated and
correct (simulator reproduces 0.83), but Q50 at this depth/CZ count cannot sustain
the DQI amplification — the hardware result is a genuine null at the random
baseline. Not dressed up: bare 0.122 ≈ random 0.125, EMS 0.136 marginally above.
