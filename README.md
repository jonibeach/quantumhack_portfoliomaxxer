# DQI for fixed-income immunization

Code and experimental artifacts for *“Towards DQI for finance applications.”*

Decoded Quantum Interferometry (DQI) is a new *fully quantum* (not hybrid)
optimization algorithm whose advantage hinges on a single requirement: the
problem must be expressible as the parity-check matrix of an error-correcting
code. For generic problems this structure is absent. We identify **fixed-income
immunization** as a finance problem that natively fulfills it. The fundamental
goal of immunization is to ensure a financial institution can always meet its
future payment obligations (liabilities), regardless of whether interest rates
rise or fall — and because a portfolio's interest-rate sensitivities (duration,
convexity, key-rate durations) are successive *moments of maturity*, moment
matching is a **Vandermonde / locator** system, i.e. a BCH/Reed–Solomon parity
check decodable by **Berlekamp–Massey**.

We formalize a finite-field, combinatorial surrogate of immunization, cast it as
a max-XORSAT instance, and show via **statevector simulation** that DQI
amplifies good solutions *only* when this algebraic code structure is present,
at low circuit cost. We then run the circuits on **two real superconducting
processors**. On IQM's 53-qubit **VTT Q50**, the `t=1` bijection collapse
amplifies above random — **6.0× at 7 bonds** (5 routed two-qubit gates) and
**4.9× at 31 bonds** (29 routed gates) — fitting under the device's ~40-gate
coherence wall. On IBM's 156-qubit **Heron r2** (`ibm_marrakesh`), the same
collapsed instances reach **6.26× and 15.7× over random**, and the **full,
un-collapsed 11-qubit DQI circuit** — 224 routed two-qubit gates, including the
one-hot error register and reversible decode/uncompute — **retains a signal
through the entire decoder** (1.69× over random, peaked on the true optimum),
where it had decohered to uniform on Q50. Finally, in simulation we build and
verify the **genuine multi-error (`t=2`) reversible Berlekamp–Massey decoder** —
the real DQI, where the decoder *is* the circuit run in superposition —
preserving full amplification (**23.8× lift**) at 28 qubits / 632 two-qubit
gates with polynomial scaling.

These results do not constitute a production immunization optimizer; rather,
they establish the problem as a finance-motivated structure in which DQI's
decoding requirement is naturally satisfied, and demonstrate above-random
amplification on present-day hardware.

## Layout

```
dqi_portfolio/            core package
  dqi_algebraic.py        ← Berlekamp–Massey decoder, GF(2^m) arithmetic, BCH,
                            t=1/t=2 datapaths, RS-Forney, resource estimation  (the contribution)
  immunization.py         maturity → locator mapping; immunization == BCH instance
  dqi.py                  belief-propagation (BP) regime  [wraps external/bcg-dqi]
  dqi_gje.py              Gauss–Jordan (GJE) regime        [wraps external/DQI-Circuit]
  metrics.py              gate/depth counting
  _vendor.py              puts the two submodules on sys.path (they aren't pip-installable)
external/                 BP & GJE baselines as pinned git submodules
scripts/                  one script per reported result (see table below)
artifacts/
  lumi/                   statevector decoder-regime sweep (CSV/JSON) — paper Table 2
  iqm/                    VTT Q50 runs: QASM, raw results, submission scripts, circuit renders
  ibm/                    IBM Heron r2 runs: submit/poll/fetch jobs, raw counts
```

The three `build_dqi_circuit*` functions share a signature so the decoder can be
swapped on a fixed instance — that swap *is* the paper's central experiment.

## Install

```bash
git submodule update --init        # fetch the BP & GJE baselines (external/)
uv sync                            # or:  python -m venv .venv && pip install -e .
```

Core deps are just `qiskit`, `qiskit-aer`, `numpy`, `scipy`, `matplotlib`.
The two submodules under `external/` supply the BP and GJE decoder baselines;
they are not pip-installable, so `dqi_portfolio/_vendor.py` adds them to the path.
Running on the QPU additionally needs the `qpu` extra (`pip install -e '.[qpu]'`).

## Reproduce the results

Run from the repo root as modules (so cross-script imports resolve):

| Script | Paper result |
|---|---|
| `python -m scripts.algebraic_prototype` | algebraic decoder, BCH-7/15 `t=1` (regime sweep) |
| `python -m scripts.algebraic_t2_prototype` | `t=2` exact + measured BM sub-blocks |
| `python -m scripts.bm_datapath_t2` | **genuine reversible Berlekamp–Massey** `t=2` (bch-7-t2, 28q/632 CZ) ⚠ heavy |
| `python -m scripts.rs_forney_symbol` | symbol-level Reed–Solomon Forney decoder (rs-sym, 37q/668 CZ) |
| `python -m scripts.bp_vs_gje` | BP vs GJE gate-count baselines |
| `python -m scripts.verify_gje` | GJE control (stays at random floor) |
| `python -m scripts.immunization_binary` | **7-bond `t=1` collapse** → Q50 (154→5 CZ) |
| `python -m scripts.immunization_binary_gf5` | **31-bond `t=1` collapse** → Q50 (29 routed CZ) |
| `python -m scripts.immunization_prototype` | immunization driver / structural-match check |
| `python -m scripts.render_t1_7bond_circuit` | the 11-qubit circuit figures |
| `python -m scripts.iqm_access_check` | Q50 routing check (needs `qpu` extra) |
| `python -m scripts.dqi_t2_deliverables` | `t=2` validation/resource/noise tables |

Hardware runs themselves are not submitted from here; the Q50 QASM, runners, and
SLURM scripts live under `artifacts/iqm/` (results in
`artifacts/iqm/q50_immunization_results.md`).

## Scope

This solves a **finite-field combinatorial surrogate** of immunization
(“which weight-`t` subset of bonds is the odd-one-out for the binarised moment
syndrome over GF(2^m)?”), faithful to immunization's *algebra* but not its
real-valued cash flows. See the paper's Discussion for the full list of
limitations. The BP and GJE regimes are vendored third-party baselines; everything in `dqi_algebraic.py`, `immunization.py`, and
`scripts/` is original.

## References, credits & thanks

This work stands on other people's code and papers. Thank you to the authors below.

### Vendored code

The two in-circuit decoder baselines we compare against are pinned git
submodules — we wrap them, we did not write them:

- **BP regime** — [`bcg-x-official/dqi`](https://github.com/bcg-x-official/dqi),
  accompanying Sabater, El Harzli, Besjes, Erdmann, Klepsch, Hiltrop, Bobier,
  Cao & Riofrío, *“Towards Solving Industrial Integer Linear Programs with
  Decoded Quantum Interferometry,”* [arXiv:2509.08328](https://arxiv.org/abs/2509.08328)
  (2025). Quantum belief-propagation decoder + ILP→max-XORSAT pipeline.
- **GJE regime** — [`BankNatchapol/DQI-Circuit`](https://github.com/BankNatchapol/DQI-Circuit),
  accompanying *“Quantum Circuit Design for Decoded Quantum Interferometry,”*
  [arXiv:2504.18334](https://arxiv.org/abs/2504.18334). Gauss–Jordan /
  BPQM / USD circuit implementations of DQI.

### Foundational paper

- S. P. Jordan, N. Shutty, M. Wootters, A. Zalcman, A. Schmidhuber, R. King,
  S. V. Isakov & R. Babbush (Google Quantum AI),
  *“Optimization by Decoded Quantum Interferometry,”*
  [arXiv:2408.08292](https://arxiv.org/abs/2408.08292); Nature **646** (2025),
  [doi:10.1038/s41586-025-09527-5](https://doi.org/10.1038/s41586-025-09527-5).
  DQI itself, and the OPI Reed–Solomon advantage witness our immunization
  mapping mirrors.

### DQI literature we drew on

- *Optimization of Quadratic Constraints by DQI* — [arXiv:2510.08061](https://arxiv.org/abs/2510.08061)
  (closest machinery to portfolio risk; the diagonality gap).
- *A nearly linear-time DQI algorithm for OPI* — [arXiv:2601.15171](https://arxiv.org/abs/2601.15171).
- *Tight inapproximability of max-LINSAT and implications for DQI* — [arXiv:2603.04540](https://arxiv.org/abs/2603.04540).
- *Algebraic Geometry Codes and DQI* — [arXiv:2510.06603](https://arxiv.org/abs/2510.06603).
- *DQI for Weighted Optimization Problems* — [arXiv:2605.10666](https://arxiv.org/abs/2605.10666).
- *Verifiable Quantum Advantage via Optimized DQI Circuits* — [arXiv:2510.10967](https://arxiv.org/abs/2510.10967).
- *DQI Under Noise* — [arXiv:2508.10725](https://arxiv.org/abs/2508.10725);
  [doi:10.1088/2058-9565/ae4536](https://doi.org/10.1088/2058-9565/ae4536).

### Hardware & compute

- **VTT Q50** — the 53-qubit superconducting processor the `t=1` collapse ran on,
  accessed through the **LUMI** supercomputer (CSC / EuroHPC JU; JQH2026
  reservation).
- **IBM Quantum** — additional hardware runs (`ibm_marrakesh`).
- Built with [Qiskit](https://www.ibm.com/quantum/qiskit) and
  [Qiskit Aer](https://github.com/Qiskit/qiskit-aer), plus IQM's Qiskit-on-IQM
  stack for Q50 access.

Parts of this codebase were developed with the assistance of [Claude](https://claude.com/claude-code).
