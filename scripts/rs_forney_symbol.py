"""Symbol-level Reed-Solomon decode with FORNEY magnitudes — closing the
real-valued-immunization caveat at the circuit level.

The binary-BCH DQI path (scripts/bm_datapath_t2.py) answers a COMBINATORIAL
question — "which bond is the odd-one-out in the moment-syndrome bit pattern" —
where every error magnitude is 1, so Forney's algorithm is trivial and never
built. A LITERAL Reed-Solomon immunization instead carries a GF(2^m) SYMBOL per
bond (the bond's real-valued amount, reduced into the field): each error has a
LOCATION *and* a MAGNITUDE. Recovering the magnitude is Forney's algorithm — the
genuinely-new reversible primitive demonstrated here.

  CHECK 1 — classical symbol-level RS decode (locator + Chien + FORNEY) is exact
            for t=2 over GF(2^3): location AND magnitude recovered for every
            weight-<=2 symbol error. This is the spec the reversible circuit
            must reproduce (and the documented t=2 scale-up target).

  CHECK 2 — reversible t=1 symbol-level RS decoder (_append_rs_decode_t1_symbol):
            given the field syndromes S_1, S_2 of a weight-<=1 SYMBOL error, it
            computes the Forney magnitude Y = S_1^2 / S_2 by reversible field
            arithmetic and XORs it back out of the symbol register, uncomputing
            the error. Verified by classical bit-walk (the block is pure
            x/cx/ccx/mcx — a permutation — so NO statevector is needed) on every
            (position, nonzero magnitude) single-symbol error plus the zero
            error, checking ys -> 0 AND all ancillas restored to |0>.

  CHECK 3 — measured gate cost of the reversible Forney decoder (IQM basis).

Run:  uv run python scripts/rs_forney_symbol.py
No statevector simulation; safe to run on a laptop.
"""

import itertools
import warnings
from functools import partial

import numpy as np
from qiskit import QuantumCircuit, transpile

from dqi_portfolio import BCHCode, gate_stats  # noqa: F401
from dqi_portfolio.bases import IQM_BASIS
from dqi_portfolio.dqi_algebraic import GF2m, _append_rs_decode_t1_symbol
from dqi_portfolio.report import section as _section
from scripts.bm_datapath_t2 import classical_reversible_eval

warnings.filterwarnings("ignore")
section = partial(_section, width=76)


# ---------------------------------------------------------------------------
# CHECK 1 — classical symbol-level RS decode with Forney (the spec).
# ---------------------------------------------------------------------------
def classical_rs_decode(gf, t, supp, vals):
    """Locator (2x2/closed form) + Chien + Forney over GF(2^m); returns {pos: mag}."""
    n = gf.n
    alphas = [gf.alpha_pow(i) for i in range(n)]
    inv0 = lambda a: 0 if a == 0 else gf.inv(a)

    def peval(coeffs, xv):
        acc, xp = 0, 1
        for c in coeffs:
            if c:
                acc ^= gf.mul(c, xp)
            xp = gf.mul(xp, xv)
        return acc

    S = []
    for k in range(1, 2 * t + 1):
        acc = 0
        for p, v in zip(supp, vals):
            acc ^= gf.mul(v, gf.pow(alphas[p], k))
        S.append(acc)

    if t == 1:
        S1, S2 = S
        if S1 == 0:
            sigma = [1]
        else:
            sigma = [1, gf.mul(S2, inv0(S1))]     # sigma_1 = S2/S1 = X
    else:
        S1, S2, S3, S4 = S
        det = gf.mul(S1, S3) ^ gf.mul(S2, S2)
        if det == 0:
            sigma = [1] if S1 == 0 else [1, gf.mul(S2, inv0(S1))]
        else:
            s2 = gf.mul(gf.mul(S3, S3) ^ gf.mul(S4, S2), inv0(det))
            s1 = gf.mul(gf.mul(S1, S4) ^ gf.mul(S2, S3), inv0(det))
            sigma = [1, s1, s2]

    roots = [p for p in range(n) if peval(sigma, inv0(alphas[p])) == 0]

    # Forney: Omega = S(x) sigma(x) mod x^{2t}; sigma'(x) keeps odd-degree terms.
    Sx = [S[k] for k in range(2 * t)]
    Om = [0] * (2 * t)
    for i in range(len(sigma)):
        for j in range(len(Sx)):
            if i + j < 2 * t:
                Om[i + j] ^= gf.mul(sigma[i], Sx[j])
    sigp = [0] * max(1, len(sigma) - 1)
    for k in range(1, len(sigma)):
        if k % 2 == 1:
            sigp[k - 1] = sigma[k]
    mags = {}
    for p in roots:
        Xinv = inv0(alphas[p])
        mags[p] = gf.mul(peval(Om, Xinv), inv0(peval(sigp, Xinv)))
    return mags


def check1_classical(m_field=3, t=2):
    section(f"CHECK 1 — classical symbol-level RS decode (Chien + FORNEY), GF(2^{m_field}) t={t}")
    gf = GF2m(m_field)
    n = gf.n
    ok = tot = 0
    rng = np.random.default_rng(0)
    for w in range(t + 1):
        for supp in itertools.combinations(range(n), w):
            for _ in range(8):
                vals = [int(rng.integers(1, gf.q)) for _ in supp]
                mags = classical_rs_decode(gf, t, list(supp), vals)
                ref = dict(zip(supp, vals))
                tot += 1
                ok += int(mags == ref)   # both location AND magnitude must match
    print(f"  weight<=t symbol errors: {ok}/{tot} recovered (location AND Forney magnitude)")
    assert ok == tot
    print("  => symbol-level RS decode with Forney magnitudes is EXACT (the scale-up spec).")


# ---------------------------------------------------------------------------
# CHECK 2 — reversible t=1 symbol-level RS decoder (bit-walk verified).
# ---------------------------------------------------------------------------
def check2_reversible_t1(m_field=3):
    section(f"CHECK 2 — reversible t=1 symbol-level RS decoder (Forney magnitude), GF(2^{m_field})")
    gf = GF2m(m_field)
    m, n = gf.m, gf.n
    # qubit layout: ys (n symbols x m) | S1 (m) | S2 (m) | anc (3m+1)
    ys = [[p * m + k for k in range(m)] for p in range(n)]
    base = n * m
    S1 = [base + k for k in range(m)]
    S2 = [base + m + k for k in range(m)]
    anc = [base + 2 * m + k for k in range(3 * m + 1)]
    nq = base + 2 * m + (3 * m + 1)
    print(f"  decoder block: ys={n}x{m}={n*m}q + S1={m}q + S2={m}q + anc={3*m+1}q = {nq} qubits")

    block = QuantumCircuit(nq)
    _append_rs_decode_t1_symbol(block, gf, ys, S1, S2, anc)

    ok = tot = 0
    anc_clean_all = True
    # all weight-<=1 symbol errors: zero error + (position p, nonzero magnitude Y)
    patterns = [([], [])]
    for p in range(n):
        for Yv in range(1, gf.q):
            patterns.append(([p], [Yv]))
    for supp, vals in patterns:
        init = [0] * nq
        # plant symbol error in ys
        for p, Yv in zip(supp, vals):
            for k in range(m):
                init[ys[p][k]] = (Yv >> k) & 1
        # syndromes S1 = Y*alpha^p, S2 = Y*alpha^{2p}
        s1v = s2v = 0
        for p, Yv in zip(supp, vals):
            s1v ^= gf.mul(Yv, gf.alpha_pow(p % gf.n))
            s2v ^= gf.mul(Yv, gf.alpha_pow((2 * p) % gf.n))
        for k in range(m):
            init[S1[k]] = (s1v >> k) & 1
            init[S2[k]] = (s2v >> k) & 1
        out = classical_reversible_eval(block, init)
        ys_zero = all(out[ys[p][k]] == 0 for p in range(n) for k in range(m))
        anc_zero = all(out[a] == 0 for a in anc)
        tot += 1
        ok += int(ys_zero and anc_zero)
        anc_clean_all = anc_clean_all and anc_zero
    print(f"  weight<=1 symbol errors: {ok}/{tot} uncomputed "
          f"(symbol register -> |0> AND ancillas restored to |0>)")
    print(f"  all ancillas restored to |0>: {anc_clean_all}")
    assert ok == tot
    print("  => reversible FORNEY magnitude decode is EXACT and clean "
          "(genuine symbol-level, real-valued amounts).")
    return block


def check3_cost(block):
    section("CHECK 3 — measured gate cost of the reversible Forney decoder (IQM basis)")
    g = gate_stats(transpile(block, basis_gates=IQM_BASIS, optimization_level=3))
    print(f"  reversible t=1 symbol-level RS (Forney) decode block:")
    print(f"    {g['qubits']} qubits   {g['2q_gates']} CZ   depth {g['depth']}")
    print("  (binary-BCH t=1 needs NO Forney; this is the extra cost of carrying")
    print("   genuine field-symbol MAGNITUDES = real-valued immunization amounts.)")


def main():
    check1_classical(3, 2)
    block = check2_reversible_t1(3)
    check3_cost(block)
    section("DONE — symbol-level RS decode with Forney magnitudes verified")


if __name__ == "__main__":
    main()
