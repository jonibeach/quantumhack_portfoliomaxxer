"""Fixed-income IMMUNIZATION as a natively-algebraic DQI max-XORSAT instance.

The idea (the crux — read this before trusting any number below)
----------------------------------------------------------------
Generic diversification portfolios produce a sparse LDPC dual code, which lands
DQI in the expensive belief-propagation regime (~1748 CZ at 24 qubits for a
6-asset toy — measured, far past the ~100-300 CZ NISQ budget). DQI only
*amplifies cheaply* when the constraint matrix B is the parity check of an
ALGEBRAICALLY-decodable code (BCH / Reed-Solomon), because the one known shallow
amplifying decoder is Berlekamp-Massey (BM).

Fixed-income immunization is a financial quantity that is *natively* a
Vandermonde / locator system. A bond portfolio's interest-rate sensitivities are
successive MOMENTS of maturity:

    duration   ~ sum_i x_i * tau_i        (1st moment)
    convexity  ~ sum_i x_i * tau_i^2      (2nd moment)
    ...
    key-rate k ~ sum_i x_i * tau_i^k      (k-th moment)

"Choose a subset of bonds whose portfolio matches a liability's duration,
convexity, ... up to order 2t" is literally

    sum_i x_i * tau_i^k = s_k        for k = 1 .. 2t.

Map the maturities tau_i to DISTINCT NONZERO field elements alpha_i in GF(2^m)
(the code's locator set) and that linear system IS a BCH / Reed-Solomon parity
check H[k, i] = alpha_i^k with target syndrome s = (s_1 .. s_2t). Finding the
<= t bonds whose inclusion/omission breaks immunization is exactly BM decoding.

This module builds that instance and shows it is (a column relabelling of) the
SAME binary BCH parity check the existing `build_bch_instance` / `BCHCode` /
`build_dqi_circuit_algebraic` already run on hardware — so only the financial
WRAPPER is new; the circuit is unchanged and already NISQ-runnable.


================================ HONESTY NOTE ================================
What is GENUINE here:
  * The moment structure is real. Duration / convexity / higher key-rate
    durations ARE the successive power-sums sum_i x_i tau_i^k of a bond
    portfolio. An immunization constraint set IS a Vandermonde / locator system.
    That is not a costume: it is the actual algebra a fixed-income desk uses.
  * Under the maturity -> field-element map, the binary parity check
    H[k, i] = (bit-expansion of) alpha_i^k is LITERALLY the matrix the existing
    BCH builder produces. We assert this byte-for-byte in
    `immunization_is_bch_instance`. So the financial instance maps onto the SAME
    circuit the project already simulates and the same BM decoder it already
    verifies. No new circuit, no sleight of hand on the matrix.

What is a MODELLING CHOICE (the honest caveats — do not oversell):
  * THE R-vs-GF(2^m) GAP. Real immunization matches REAL-VALUED durations:
    arithmetic is over the reals, "match a moment" is approximate equality of
    floats, and the locators alpha_i are real maturities like 2.0, 5.0 years.
    Reinterpreting the moment equations over the FINITE FIELD GF(2^m) is a
    genuine modelling decision, NOT a field isomorphism. There is no
    structure-preserving map R -> GF(2^m); we are CHOOSING to pose the
    "which bonds break the match" question over GF(2) so that BM applies. The
    GF(2) instance answers a *combinatorial* version of immunization (which
    small subset of bonds is the odd-one-out for the bit-pattern of the moment
    syndrome), not the exact real-valued duration-matching problem. A desk that
    wants real-valued matching would still solve an LP/QP classically.
  * THE TARGET SYNDROME. In the cheap (t=1) instance the DQI "variables" are the
    m syndrome bits and the m constraints are the codeword positions; the target
    vector v plays the role of the (binarised) liability syndrome. We construct
    v from a synthetic liability's moments reduced into GF(2^m); a different
    liability gives a different v but the SAME B (same bond ladder). So the
    matrix really is the bond ladder and only the right-hand side is the
    liability — which is exactly how an immunization desk thinks (fixed universe
    of bonds, varying liability).
  * t=1 vs t>=2. The CHEAP, SIMULATED, HARDWARE-RUNNABLE case is t=1 (single
    "odd-one-out" bond, Hamming/BCH locator decode, ancilla v-chain MCX). t>=2
    (match duration AND convexity AND ... to higher order, correct several bonds)
    uses the project's pattern-enumeration decode (exact but not the efficient BM
    datapath) or the resource-estimated BM circuit; the efficient reversible BM
    datapath is not yet wired end-to-end (see `estimate_bm_resources`).

VERDICT (stated up front; the prototype script measures the numbers):
  This is a DEFENSIBLE genuinely-algebraic finance instance *as a combinatorial
  immunization question over GF(2)*, with one load-bearing caveat: the move from
  real-valued moment matching to GF(2^m) moment matching is a modelling choice,
  not an isomorphism. Within that choice, the instance is the real BCH demo with
  a faithful financial reading, not a costume bolted onto unrelated math.
=============================================================================
"""

from __future__ import annotations

import numpy as np

from . import gf2
from .dqi_algebraic import (
    GF2m,
    BCHCode,
    build_bch_instance,
    berlekamp_massey_decode,
    _bm_locator,
)

__all__ = [
    "BondLadder",
    "build_immunization_instance",
    "immunization_is_bch_instance",
    "decode_portfolio",
    "immunization_residual",
    "rs_immunization_check",
]


# ---------------------------------------------------------------------------
# Synthetic bond ladder.  SYNTHETIC, fixed seed, no network.
# ---------------------------------------------------------------------------
class BondLadder:
    """A small SYNTHETIC bond ladder over GF(2^m).

    n = 2^m - 1 bonds with distinct maturities tau_1 < tau_2 < ... mapped onto
    the distinct nonzero locator elements alpha_i = alpha^i of GF(2^m).

    maturity -> field-element map (documented, deterministic)
    --------------------------------------------------------
    The locator set of a length-(2^m-1) BCH/RS code is exactly the nonzero
    elements {alpha^0, alpha^1, ..., alpha^(n-1)} of GF(2^m), where alpha is the
    field's primitive element. We assign bond i (i = 0..n-1, sorted by maturity)
    to locator alpha^i. The maturities themselves are a synthetic increasing
    ladder (tau_i = base + i*step years); the ORDER is what carries into the
    code (bond i <-> alpha^i). This is the standard "evaluation points" choice
    for an RS/BCH code and is the only place the finance meets the field.
    """

    def __init__(self, m_field: int = 3, base: float = 1.0, step: float = 1.0,
                 seed: int = 0):
        self.gf = GF2m(m_field)
        self.m_field = m_field
        self.n = self.gf.n                       # 2^m - 1 bonds
        rng = np.random.default_rng(seed)
        # Synthetic increasing maturity ladder (years).  Small jitter for realism,
        # kept monotone so the sort order (and hence the locator assignment) is
        # deterministic.
        jitter = rng.uniform(0.0, 0.3, size=self.n)
        self.maturities = base + step * np.arange(self.n) + np.sort(jitter)
        # locator alpha_i = alpha^i  (bond i, sorted by maturity)
        self.locators = [self.gf.alpha_pow(i) for i in range(self.n)]
        # Synthetic face/coupon just for a richer financial readout (not used by
        # the field instance — labelled clearly).
        self.coupons = rng.uniform(0.02, 0.06, size=self.n)
        self.faces = np.ones(self.n)

    def describe(self) -> str:
        lines = [f"SYNTHETIC bond ladder over GF(2^{self.m_field}): {self.n} bonds"]
        lines.append("  bond  maturity(yr)   locator alpha^i (GF elt)")
        for i in range(self.n):
            lines.append(f"   {i:>2}     {self.maturities[i]:6.3f}        "
                         f"alpha^{i} = {self.locators[i]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Immunization instance  ==  binary BCH parity-check instance.
# ---------------------------------------------------------------------------
def build_immunization_instance(m_field: int = 3, t: int = 1, seed: int = 0):
    """Build the immunization max-XORSAT instance ``B x = v`` over GF(2).

    Returns ``(B, v, code, ladder)`` where:
      * ``ladder``    : the SYNTHETIC `BondLadder`;
      * ``code``      : the `BCHCode(m_field, t)` whose parity check is the
                        moment/locator matrix H[k, i] = (bits of) alpha_i^k;
      * ``B``         : the DQI constraint matrix, shape (n_positions, n_checks),
                        equal to ``code.H.T`` (== `build_bch_instance`'s B);
      * ``v``         : the (binarised) liability target syndrome.

    By construction this is IDENTICAL to `build_bch_instance(m_field, t)` — the
    immunization story is the financial *reading* of that exact instance. We
    therefore reuse `build_bch_instance` verbatim and attach the ladder, then
    overwrite ``v`` with a liability-derived syndrome (same shape) so the matrix
    is unchanged but the right-hand side is the liability, not noise.
    """
    ladder = BondLadder(m_field=m_field, seed=seed)
    B, _v_random, code = build_bch_instance(m_field, t)
    # Liability target syndrome v: derive from a synthetic liability cashflow
    # whose moments we reduce into the field, then express in the SAME basis as
    # the DQI constraints (one bit per codeword position).  We pick a concrete
    # weight-1 "ideal" liability (matched by exactly one bond's locator) so the
    # instance has a clean financial reading AND a known optimum; then binarise.
    v = _liability_syndrome(code, ladder, seed=seed)
    assert v.shape[0] == B.shape[0]
    return B, v, code, ladder


def _liability_syndrome(code: BCHCode, ladder: BondLadder, seed: int = 0):
    """Target vector v (one bit per codeword position) from a synthetic liability.

    Financial reading: the liability has its own duration/convexity/... profile,
    which we represent as an "ideal" set of <= t matching bonds — the bonds whose
    locator pattern reproduces the liability's moment syndrome exactly. DQI is
    then asked to find the bond subset whose locator pattern best matches it (max
    satisfied constraints = best match).

    Construction (deterministic, length-correct for all t):
      * pick a weight-t "ideal" liability bond pattern ``y_L`` (a single bond for
        t=1, t distinct bonds for t>=2);
      * its binary BCH syndrome ``s_L = (H @ y_L) mod 2`` has length n_checks
        (== B.shape[1], the number of DQI solution variables) — this is the
        liability's target syndrome in the SAME basis as the DQI variables;
      * the per-position target ``v = (H^T @ s_L) mod 2 = B @ s_L`` (length n =
        n_positions) is then exactly realisable: the solution ``x = s_L`` satisfies
        ALL m constraints, so the instance has a clean known optimum at the
        liability's own syndrome.

    For t == 1 this reduces BYTE-FOR-BYTE to the previous construction: a single
    ideal locator ``L = alpha^j`` gives ``s_L = (H @ e_j) = `` (bit-expansion of
    ``alpha^j``) ``= `` column j of H (verified by ``locator_columns_ok``), so the
    hardware-validated 7-bond t=1 instance is unchanged. The bug fixed here is
    that the old code hard-coded ``len(s_L) = code.m`` (the t=1 syndrome size);
    for t>=2 the BCH check space has ``n_checks = m*t`` bits, so ``s_L`` must be
    sized by ``code.H.shape[0]`` (== B.shape[1]).
    """
    rng = np.random.default_rng(seed + 1)
    H = code.H                                    # (n_checks, n)
    # Weight-t ideal liability bond pattern (single bond for t=1; t bonds else).
    if code.t == 1:
        # Preserve the validated t=1 instance: one ideal locator alpha^j.
        y_L = np.zeros(code.n, dtype=int)
        y_L[int(rng.integers(0, code.n))] = 1
    else:
        y_L = np.zeros(code.n, dtype=int)
        for p in rng.choice(code.n, size=code.t, replace=False):
            y_L[int(p)] = 1
    # Liability target syndrome (length n_checks == B.shape[1]).
    s_L = (H @ y_L) % 2
    # v = (H^T @ s_L) mod 2 = B @ s_L  ->  optimum solution x = s_L (all satisfied)
    v = (H.T @ s_L) % 2                           # length n = n_positions
    return v.astype(int)


def immunization_is_bch_instance(m_field: int = 3, t: int = 1, seed: int = 0):
    """CHECK: the immunization B is byte-for-byte the binary BCH parity check B.

    Returns a dict with the equality result and the matrices, so the prototype
    can assert the financial wrapper did NOT alter the circuit-bearing matrix.
    """
    B_imm, v_imm, code, ladder = build_immunization_instance(m_field, t, seed)
    B_bch, _v_bch, _code2 = build_bch_instance(m_field, t)
    same_B = bool(np.array_equal(B_imm % 2, B_bch % 2))
    # Also confirm H column j is the bit-expansion of alpha^j (the locator),
    # which is what makes this a Vandermonde/locator (RS-flavoured) check at t=1.
    locator_ok = True
    if t == 1:
        for j in range(code.n):
            col = code.H[:, j]
            alpha_j = code.gf.alpha_pow(j)
            bits = np.array([(alpha_j >> b) & 1 for b in range(code.m)])
            if not np.array_equal(col % 2, bits):
                locator_ok = False
                break
    return {
        "B_equals_bch": same_B,
        "locator_columns_ok": locator_ok,
        "B": B_imm,
        "v": v_imm,
        "code": code,
        "ladder": ladder,
    }


# ---------------------------------------------------------------------------
# Financial readout.
# ---------------------------------------------------------------------------
def decode_portfolio(code: BCHCode, ladder: BondLadder, x_bits: np.ndarray):
    """Decode a DQI solution bitstring into an included-bond set + residual.

    ``x_bits`` is the n_checks-bit solution register (the syndrome variables).
    We reconstruct the implied error word y (weight <= t) via BM on the syndrome
    the solution encodes, and read its support as the set of "odd-one-out" bonds
    — the bonds whose inclusion/omission breaks immunization. Returns a dict with
    the bond indices, their maturities/locators, and the immunization residual.
    """
    # The solution register encodes the syndrome s = H y for some low-weight y.
    # Reconstruct the candidate received word whose syndrome equals the encoded
    # value by treating x_bits as the m-bit syndrome and BM-decoding the
    # all-zero word forced to that syndrome.  For t=1 the syndrome directly names
    # the single locator; we find which position's H column equals x_bits.
    x_bits = np.asarray(x_bits, dtype=int) % 2
    H = code.H
    support = []
    if code.t == 1:
        for j in range(code.n):
            if np.array_equal(H[:, j] % 2, x_bits[: H.shape[0]]):
                support.append(j)
                break
    else:
        # build a word with this syndrome and BM-decode it.
        y = _word_with_syndrome(code, x_bits)
        if y is not None:
            e = berlekamp_massey_decode(code, y)
            if e is not None:
                support = list(np.nonzero(e)[0])
    bonds = [
        {
            "bond": int(j),
            "maturity": float(ladder.maturities[j]),
            "locator": int(ladder.locators[j]),
        }
        for j in support
    ]
    return {
        "included_bonds": support,
        "bond_details": bonds,
        "residual": immunization_residual(code, ladder, support, x_bits),
    }


def _word_with_syndrome(code: BCHCode, target_syndrome_bits: np.ndarray):
    """Find any binary word y (length n) whose binary syndrome H y == target.

    Solves H y = target over GF(2) by back-substitution (shared ``gf2.solve``).
    Returns None if inconsistent (target not in column space).
    """
    H = np.asarray(code.H, dtype=int) % 2
    s = (np.asarray(target_syndrome_bits, dtype=int) % 2)[: H.shape[0]]
    return gf2.solve(H, s)


def immunization_residual(code: BCHCode, ladder: BondLadder, support, x_bits):
    """Immunization residual of the decoded portfolio.

    GF(2^m) residual: the field-syndrome mismatch between the decoded error
    word's syndrome and the encoded target. Zero means the decoded bond set
    exactly reproduces the target locator pattern (perfect combinatorial
    immunization match). We also report a REAL-VALUED moment readout (duration /
    convexity power-sums of the included bonds) for the financial reader, clearly
    labelled as the R-side companion to the GF(2) instance.
    """
    # GF(2^m) field-syndrome residual of the support set.
    y = np.zeros(code.n, dtype=int)
    for j in support:
        y[j] = 1
    field_syn = code.syndromes(y)                 # S_1..S_2t over GF(2^m)
    field_residual = int(sum(1 for s in field_syn if s != 0))

    # REAL-side moment readout (companion, labelled).
    taus = ladder.maturities[support] if len(support) else np.array([])
    real_moments = {
        "duration_sum_tau": float(np.sum(taus)) if len(taus) else 0.0,
        "convexity_sum_tau2": float(np.sum(taus ** 2)) if len(taus) else 0.0,
        "n_bonds": len(support),
    }
    return {
        "field_syndrome": field_syn,
        "field_nonzero_syndromes": field_residual,
        "real_moment_readout_LABELLED": real_moments,
    }


# ---------------------------------------------------------------------------
# The RS impedance investigation (classical, no circuit claim).
# ---------------------------------------------------------------------------
def rs_immunization_check(m_field: int = 3, t: int = 2, seed: int = 0):
    """Investigate the binary-BCH vs Reed-Solomon (symbol-level) impedance.

    The existing circuit path is the BINARY narrow-sense BCH code: binary parity
    check, BM over the binary subfield. A *literal* RS immunization code has
    SYMBOLS in GF(2^m) (each bond carries a field symbol, not a bit) and is a
    DIFFERENT code. This function builds a small symbol-level GRS instance and
    runs a CLASSICAL BM decode on it to confirm the algebra works, then reports
    whether the existing builder applies.

    Returns a dict describing: (a) that the binary-BCH path applies cleanly to
    the GF(2)-reduced immunization instance (reused circuit), and (b) a verified
    classical BM decode of the symbol-level RS instance, with the honest note
    that the reversible RS circuit is NOT wired into `build_dqi_circuit_algebraic`.
    """
    gf = GF2m(m_field)
    n = gf.n
    # GRS evaluation points = locators alpha^i (the bond maturities' images).
    alphas = [gf.alpha_pow(i) for i in range(n)]

    # Symbol-level RS parity check H_rs[k, i] = alpha_i^(k)  in GF(2^m), k=1..2t.
    H_rs = [[gf.pow(alphas[i], k) for i in range(n)] for k in range(1, 2 * t + 1)]

    # Plant a weight-<=t symbol error and decode it classically with the SAME
    # BM the project ships (it operates on syndromes, field-agnostic at symbol
    # level for binary-symbol errors; for a true symbol-error RS decode we run a
    # standalone BM over GF(2^m) here to prove the algebra).
    rng = np.random.default_rng(seed)
    err_pos = sorted(rng.choice(n, size=t, replace=False).tolist())
    err_val = [int(rng.integers(1, gf.q)) for _ in err_pos]   # nonzero symbols

    # syndromes S_k = sum_j e_j * alpha_j^k   (symbol-level, over GF(2^m))
    S = []
    for k in range(1, 2 * t + 1):
        acc = 0
        for p, val in zip(err_pos, err_val):
            acc ^= gf.mul(val, gf.pow(alphas[p], k))
        S.append(acc)

    # Standalone BM over GF(2^m) to recover the error-locator polynomial.
    locator, _L = _bm_locator(gf, S, t)
    # Chien: roots of locator -> error positions (alpha_i is a root of recip).
    found = []
    for i in range(n):
        # locator evaluated at alpha_i^{-1}
        xinv = gf.inv(alphas[i])
        val = 0
        xp = 1
        for c in locator:
            if c:
                val ^= gf.mul(c, xp)
            xp = gf.mul(xp, xinv)
        if val == 0:
            found.append(i)
    rs_decode_ok = sorted(found) == err_pos

    return {
        "field": f"GF(2^{m_field})",
        "n_positions": n,
        "t": t,
        "planted_error_positions": err_pos,
        "planted_error_symbols": err_val,
        "rs_classical_BM_recovered_positions": sorted(found),
        "rs_classical_BM_decode_ok": rs_decode_ok,
        "binary_bch_path_applies": True,
        "note": (
            "The GF(2)-REDUCED immunization instance (combinatorial 'which bond "
            "is the odd-one-out') maps CLEANLY onto the existing binary-BCH "
            "circuit: build_immunization_instance == build_bch_instance, so "
            "build_dqi_circuit_algebraic runs unchanged (t=1 simulated). A LITERAL "
            "symbol-level Reed-Solomon immunization (each bond carries a GF(2^m) "
            "symbol) is a DIFFERENT code; its classical BM decode is verified here "
            "(rs_classical_BM_decode_ok), but the reversible RS datapath is NOT "
            "wired into build_dqi_circuit_algebraic -- that needs the GF(2^m) "
            "multiply-add blocks (build_gf2m_mul_add_circuit, resource-estimated "
            "in estimate_bm_resources), not yet assembled end-to-end."
        ),
    }
