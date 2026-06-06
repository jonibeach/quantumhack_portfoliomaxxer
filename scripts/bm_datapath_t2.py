"""GENUINE reversible Berlekamp-Massey datapath at t=2 — build, verify, cost.

This is the REAL multi-error DQI immunization decoder (not pattern enumeration):
given the binary syndrome in the solution register, it computes the error-locator
polynomial by FIELD ARITHMETIC and uncomputes the error register y via a
reversible Chien search, with every scratch qubit returned to |0>. It scales
POLYNOMIALLY in t (for t=2: two GF(2^m) multiplies + linear maps), where the
pattern-enumeration tier costs C(m,t) multi-controlled gates.

Three checks, each reproducing every number reported:

  CHECK A — decoder block vs classical Berlekamp-Massey (correctness).
    Feed every weight-<=2 error pattern's binary syndrome into the BM block
    (computational basis), confirm it flips EXACTLY the planted error positions
    and restores all ancillas to |0>. Compare to berlekamp_massey_decode.

  CHECK B — full DQI circuit (superposition) vs the ideal amplified distribution.
    Build the end-to-end DQI circuit with decoder="bm", statevector-simulate, and
    confirm the solution-register histogram matches dqi_satisfaction_stats (the
    faithful DQI output) to within shot noise. Proves amplification survives the
    GENUINE datapath, not just enumeration.

  CHECK C — measured gate budget (IQM basis), BM vs enumeration.

Run:  uv run python scripts/bm_datapath_t2.py
On LUMI:  ~28 qubits, statevector ~4 GB — fits one CPU node comfortably.
"""

import itertools
import warnings

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from dqi_portfolio import (
    BCHCode,
    berlekamp_massey_decode,
    dqi_satisfaction_stats,
    build_dqi_circuit_algebraic,
    gate_stats,
)
from dqi_portfolio.bases import IQM_BASIS, SIM_BASIS
from dqi_portfolio.dqi_algebraic import _append_bm_decode_t2
from dqi_portfolio.immunization import build_immunization_instance
from dqi_portfolio.readout import score_counts
from dqi_portfolio.report import section

warnings.filterwarnings("ignore")
# ccx kept un-decomposed for exact reversible-arithmetic verification
# (mcx with >2 controls decomposes into ccx/cx, noancilla mode).
VERIFY_BASIS = ["cx", "x", "ccx"]


def classical_reversible_eval(qc, init_bits):
    """Evaluate a CLASSICAL reversible circuit (x/cx/ccx/mcx only) on a basis state.

    The BM decoder block has NO Hadamards/rotations (verified: it is a permutation
    of basis states), so we can simulate it exactly by walking the gate list over a
    bit array — instant and memory-free, with NO 2^n statevector. This validates
    the EXACT emitted qiskit circuit, gate for gate.
    """
    bits = list(init_bits)
    qubit_index = {q: i for i, q in enumerate(qc.qubits)}
    for inst in qc.data:
        name = inst.operation.name
        qs = [qubit_index[q] for q in inst.qubits]
        if name == "x":
            bits[qs[0]] ^= 1
        elif name == "cx":
            bits[qs[1]] ^= bits[qs[0]]
        elif name == "ccx":
            bits[qs[2]] ^= bits[qs[0]] & bits[qs[1]]
        elif name == "mcx":
            ctrl = qs[:-1]
            bits[qs[-1]] ^= int(all(bits[c] for c in ctrl))
        elif name in ("barrier", "measure"):
            continue
        else:
            raise ValueError(f"non-classical gate {name!r} in decoder block")
    return bits


def check_a_decoder_vs_classical(m_field=3):
    section(f"CHECK A — BM decoder block vs classical Berlekamp-Massey (GF(2^{m_field}), t=2)")
    code = BCHCode(m_field, 2)
    B = code.H.T
    m, n = B.shape                      # m = positions (=code.n), n = n_checks
    n_anc = 5 * code.m
    nq = m + n + n_anc
    print(f"  decoder block: y={m}q + syndrome={n}q + BM ancilla={n_anc}q = {nq} qubits")

    # Build the decode block ONCE (it is input-independent); we then evaluate it
    # classically on each error pattern's basis state (no statevector, no heat).
    block = QuantumCircuit(nq)
    _append_bm_decode_t2(
        block, code,
        y_idx=list(range(m)),
        syn_idx=[m + i for i in range(n)],
        anc_idx=[m + n + i for i in range(n_anc)],
    )

    Bt = B.T                            # syndrome s = Bt @ e   (n x m)
    ok = tot = 0
    anc_restored = True
    for w in range(code.t + 1):
        for supp in itertools.combinations(range(m), w):
            e = np.zeros(m, dtype=int)
            for p in supp:
                e[p] = 1
            s = (Bt @ e) % 2            # binary syndrome of this error

            # initial basis state |y=e>|syn=s>|anc=0>
            init = [0] * nq
            for p in range(m):
                init[p] = int(e[p])
            for i in range(n):
                init[m + i] = int(s[i])
            out = classical_reversible_eval(block, init)
            y_out = out[:m]
            anc_out = out[m + n:m + n + n_anc]

            # classical reference: BM must recover the same support
            e_classical = berlekamp_massey_decode(code, e)
            tot += 1
            y_uncomputed = all(b == 0 for b in y_out)
            anc_clean = all(b == 0 for b in anc_out)
            classical_ok = (e_classical is not None
                            and np.array_equal(e_classical % 2, e % 2))
            ok += int(y_uncomputed and anc_clean and classical_ok)
            anc_restored = anc_restored and anc_clean

    print(f"  weight<=2 patterns: {ok}/{tot} fully decoded "
          f"(y uncomputed to |0> AND ancillas restored to |0> AND match classical BM)")
    print(f"  all ancillas restored to |0>: {anc_restored}")
    assert ok == tot, "BM decoder block does NOT match classical Berlekamp-Massey!"
    print("  => GENUINE reversible BM datapath is EXACT and clean.")
    return code, B


def check_b_full_circuit(m_field=3, ell=2):
    section(f"CHECK B — full DQI circuit (decoder='bm') vs ideal amplified distribution")
    B, v, code, ladder = build_immunization_instance(m_field, 2)
    m, n = B.shape
    st = dqi_satisfaction_stats(B, v, code, ell)
    ideal = st["histogram"]
    opt = st["max_satisfied"]

    qc_meas = build_dqi_circuit_algebraic(code, B, v, ell=ell, decoder="bm",
                                          with_measurements=True)
    print(f"  full DQI circuit: {qc_meas.num_qubits} qubits "
          f"(y={m} + syndrome={n} + BM ancilla={5*code.m})")

    sim = AerSimulator()
    tq = transpile(qc_meas, sim, basis_gates=SIM_BASIS)
    shots = 400_000
    counts = sim.run(tq, shots=shots).result().get_counts()
    r = score_counts(counts, B, v, x_offset=m)
    tot = r["total"]
    hist = {k: r["hist"][k] / tot for k in r["hist"]}
    mean_c = r["mean"]

    print(f"\n   f      ideal-DQI   circuit(BM)")
    fs = sorted(set(list(ideal) + list(hist)), reverse=True)
    for f in fs:
        print(f"   {f}/{m}   {ideal.get(f, 0):8.4f}   {hist.get(f, 0):8.4f}")
    maxdiff = max(abs(ideal.get(f, 0) - hist.get(f, 0)) for f in fs)
    floor = 1 / np.sqrt(tot)
    print(f"\n  mean satisfied : ideal {st['mean_satisfied']:.4f}   circuit {mean_c:.4f}")
    print(f"  P(opt={opt})       : ideal {ideal.get(opt,0):.4f}   circuit {hist.get(opt,0):.4f}")
    print(f"  max |ideal-circuit| over histogram = {maxdiff:.4f}  "
          f"(shot-noise floor ~{floor:.4f})")
    assert maxdiff < 5 * floor, "BM circuit distribution does NOT match the ideal!"
    print("  => amplification SURVIVES the genuine BM datapath (matches ideal).")


def check_c_cost(m_field=3, ell=2):
    section("CHECK C — measured gate budget (IQM basis): BM datapath vs enumeration")
    B, v, code, ladder = build_immunization_instance(m_field, 2)
    qc_bm = build_dqi_circuit_algebraic(code, B, v, ell=ell, decoder="bm")
    qc_en = build_dqi_circuit_algebraic(code, B, v, ell=ell, decoder="enumeration")
    bm = gate_stats(transpile(qc_bm, basis_gates=IQM_BASIS, optimization_level=3))
    en = gate_stats(transpile(qc_en, basis_gates=IQM_BASIS, optimization_level=3))
    print(f"  GF(2^{m_field}) t=2, ell={ell}:")
    print(f"    decoder            qubits   CZ     depth")
    print(f"    enumeration (C(m,t)) {en['qubits']:>3}   {en['2q_gates']:>5}   {en['depth']:>5}")
    print(f"    GENUINE BM datapath  {bm['qubits']:>3}   {bm['2q_gates']:>5}   {bm['depth']:>5}")
    ratio = en['2q_gates'] / bm['2q_gates'] if bm['2q_gates'] else float('inf')
    print(f"    -> BM is {ratio:.2f}x cheaper in CZ at this instance "
          f"(and scales polynomially, not as C(m,t))")
    return {"bm": bm, "enum": en}


def main():
    check_a_decoder_vs_classical(3)
    check_b_full_circuit(3, ell=2)
    check_c_cost(3, ell=2)
    section("DONE — genuine multi-error (t=2) DQI immunization decoder verified")


if __name__ == "__main__":
    main()
