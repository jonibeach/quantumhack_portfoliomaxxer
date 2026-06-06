"""Algebraic (Berlekamp-Massey) DQI decoder prototype — "option B".

Goal of this module
-------------------
Test whether an *algebraic bounded-distance decoder* breaks the qubit/gate
"blowup vs amplification" tradeoff we measured for the two existing decoders:

  * Belief-propagation (``dqi.py``): AMPLIFIES (mean ~5.2/6 on the 6x4 instance)
    but HUGE — 20 qubits, 509 CZ, depth 884 (IQM basis).
  * Gauss-Jordan / RREF (``dqi_gje.py``): CHEAP — 10 qubits, 69 CZ, depth 130,
    no ancillas — but it is plain linear-system decoding, NOT minimum-weight
    decoding, so it does NOT amplify (solution register ~uniform).

The hypothesis (the route the Nature DQI paper takes for its flagship OPI
result) is that an *algebraic* decoder can be BOTH shallow AND amplifying,
because algebraic decoding (Reed-Solomon / Berlekamp-Massey on a structured
code) is far more circuit-friendly than iterative message passing.

Design fork — we take B2 (binary, pipeline-compatible)
------------------------------------------------------
The paper's flagship instance, OPI, is over a PRIME field F_p (Definition 2):
m = p-1 constraints, B a Vandermonde matrix, C^perp a Reed-Solomon code,
syndrome-decoded out to half-distance by Berlekamp-Massey, with l = floor((n+1)/2).
That is genuinely algebraic but requires qudit (F_p) encoding and reversible
F_p arithmetic — heavy to build from scratch and OFF our GF(2) pipeline, so it
would not yield a measured amplification number within a prototype budget.

We instead use a **binary BCH code** (B2). A BCH code is the subfield-subcode of
a Reed-Solomon code over GF(2^m); its syndrome decoding is the SAME algebraic
primitive the paper uses — Berlekamp-Massey to find the error-locator polynomial,
then Chien search for the error positions — but it lives entirely in the GF(2)
max-XORSAT world our builders and AerSimulator already handle. That means
amplification is measurable with the SAME satisfaction metric used for BP and
GJE, on the same footing. This is the faster path to a credible real number;
F_p/OPI (B1) is the natural next step, not the fastest first result.

The DQI dual-decoding picture (binary)
--------------------------------------
DQI max-XORSAT (paper eq. 1-5): maximize the number of satisfied constraints
``B x = v`` over GF(2). The decoder works on the DUAL code
``C^perp = { d in F_2^m : B^T d = 0 }`` — it is fed a "received word" related to
the error pattern ``y`` (of Hamming weight <= l) and must recover ``y`` from the
syndrome ``B^T y``. Amplification happens because a *minimum-weight* decoder maps
the uniform-over-low-weight ``y`` superposition onto solutions ``x`` that satisfy
MANY constraints; a non-minimum-weight decoder (GJE) does not.

Here we build that dual code from a binary BCH code H (parity check) over GF(2):
we set ``B^T = H`` (so the constraint matrix is ``B = H^T``). Decoding the
syndrome ``s = H y`` to the minimum-weight ``y`` is exactly BCH syndrome decoding,
solved algebraically by Berlekamp-Massey + Chien search.

What is fully built vs estimated
--------------------------------
  * GF(2^m) arithmetic, BCH code construction, Berlekamp-Massey + Chien-search
    decoder: fully built, VERIFIED to recover all weight<=t errors (see
    ``self_test`` and ``scripts/algebraic_prototype.py``).
  * Classical DQI satisfaction metric on this instance: fully computed, shows
    amplification (mean satisfied well above random).
  * Reversible DQI circuit: the linear stages (Dicke/weighted-unary prep, phase,
    syndrome B^T y, final Hadamards) are built and transpilable. The algebraic
    *key-equation* stage (Berlekamp-Massey) is the hard reversible block; we build
    a verified-correct **table-lookup / permutation** realization of the full
    minimum-weight syndrome decoder for the tiny instance (correct unitary, lets
    us SIMULATE end-to-end and confirm amplification survives), and separately a
    **gate-count resource estimate** for the genuine in-place BM datapath. Both
    numbers are reported and clearly labeled.
"""

from __future__ import annotations

import itertools
import numpy as np

from . import gf2

__all__ = [
    "GF2m",
    "BCHCode",
    "berlekamp_massey_decode",
    "dqi_satisfaction_stats",
    "build_bch_instance",
    "build_dqi_circuit_algebraic",
    "estimate_bm_resources",
    "build_gf2m_mul_add_circuit",
    "build_syndrome_circuit",
    "measure_bm_subblocks",
    "bm_t2_matrices",
    "_append_bm_decode_t2",
]


# ---------------------------------------------------------------------------
# GF(2^m) arithmetic (hand-rolled; no external deps).
# ---------------------------------------------------------------------------
class GF2m:
    """Finite field GF(2^m) via log/antilog tables for a primitive polynomial.

    Elements are ints in [0, 2^m). ``alpha`` (value 2) is a primitive element.
    """

    # Primitive polynomials (as integer bitmasks of degree m) for small m.
    _PRIM_POLY = {
        2: 0b111,        # x^2 + x + 1
        3: 0b1011,       # x^3 + x + 1
        4: 0b10011,      # x^4 + x + 1
        5: 0b100101,     # x^5 + x^2 + 1
        6: 0b1000011,    # x^6 + x + 1
    }

    def __init__(self, m: int):
        self.m = m
        self.n = (1 << m) - 1          # multiplicative-group order (2^m - 1)
        self.q = 1 << m
        poly = self._PRIM_POLY[m]
        self.exp = [0] * (2 * self.n)  # antilog: exp[i] = alpha^i
        self.log = [0] * self.q        # log: log[x] = i s.t. alpha^i = x
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.q:
                x ^= poly
        for i in range(self.n, 2 * self.n):
            self.exp[i] = self.exp[i - self.n]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("no inverse of 0 in GF(2^m)")
        return self.exp[self.n - self.log[a]]

    def pow(self, a: int, k: int) -> int:
        if a == 0:
            return 0
        return self.exp[(self.log[a] * k) % self.n]

    def alpha_pow(self, k: int) -> int:
        return self.exp[k % self.n]


# ---------------------------------------------------------------------------
# Binary BCH code: parity check over GF(2), syndromes over GF(2^m).
# ---------------------------------------------------------------------------
class BCHCode:
    """Narrow-sense binary BCH code of length n = 2^m - 1 correcting t errors.

    Provides the binary parity-check matrix H (shape (m*t, n) over GF(2)) used as
    the DQI dual-code check ``B^T``, plus an algebraic Berlekamp-Massey decoder.
    """

    def __init__(self, m: int, t: int):
        self.gf = GF2m(m)
        self.m = m
        self.t = t
        self.n = self.gf.n            # codeword length 2^m - 1
        # Syndrome powers: for narrow-sense BCH we need S_1..S_{2t}; the binary
        # parity check rows are alpha^{i*j} for i in 1..2t (only odd i are
        # independent for binary, but using 1..2t and reducing to GF(2) rows is
        # the standard explicit construction). We build H as m*2t binary rows by
        # expanding each GF(2^m) power into m bits, then drop dependent rows.
        rows = []
        for i in range(1, 2 * t + 1):
            for bit in range(m):
                row = []
                for j in range(self.n):
                    val = self.gf.alpha_pow((i * j))
                    row.append((val >> bit) & 1)
                rows.append(row)
        H = np.array(rows, dtype=int) % 2
        # Reduce to an independent set of rows (RREF over GF(2)) -> redundancy = m*t.
        self.H = gf2.row_reduce(H)
        self.n_checks = self.H.shape[0]
        self.dmin = 2 * t + 1         # BCH bound

    def syndromes(self, y: np.ndarray) -> list[int]:
        """Algebraic syndromes S_1..S_{2t} in GF(2^m) of a binary word y (len n)."""
        gf = self.gf
        err_pos = np.nonzero(np.asarray(y) % 2)[0]
        S = []
        for i in range(1, 2 * self.t + 1):
            acc = 0
            for j in err_pos:
                acc ^= gf.alpha_pow(i * int(j))
            S.append(acc)
        return S


# ---------------------------------------------------------------------------
# Berlekamp-Massey + Chien search (THE algebraic decoder).
# ---------------------------------------------------------------------------
def _bm_locator(gf: GF2m, S, t):
    """Berlekamp-Massey over GF(2^m): syndromes S_1..S_2t -> (sigma, L).

    Returns the error-locator polynomial ``sigma`` and the LFSR length ``L``
    (number of errors). Shared by both ``berlekamp_massey_decode`` (which then
    runs a Chien search to a binary error word, using ``L`` for the failure
    check) and the symbol-level RS decoder in ``immunization.rs_immunization_check``
    (which only needs ``sigma``).
    """
    sigma = [1]          # current locator
    B = [1]
    L = 0                # current LFSR length
    m_shift = 1
    b = 1                # last nonzero discrepancy
    for r in range(2 * t):
        # discrepancy delta = S_r + sum_{i=1..L} sigma_i * S_{r-i}
        delta = S[r]
        for i in range(1, L + 1):
            if i < len(sigma) and sigma[i] != 0:
                delta ^= gf.mul(sigma[i], S[r - i])
        if delta == 0:
            m_shift += 1
        else:
            coef = gf.mul(delta, gf.inv(b))
            # sigma_new = sigma - coef * x^m_shift * B
            shifted = [0] * m_shift + [gf.mul(coef, bb) for bb in B]
            new = list(sigma) + [0] * (len(shifted) - len(sigma))
            for i in range(len(shifted)):
                new[i] ^= shifted[i]
            if 2 * L <= r:
                B = list(sigma)
                L = r + 1 - L
                b = delta
                m_shift = 1
                sigma = new
            else:
                sigma = new
                m_shift += 1
    return sigma, L
def berlekamp_massey_decode(code: BCHCode, y: np.ndarray) -> np.ndarray | None:
    """Algebraically decode binary word ``y`` to the nearest BCH codeword.

    Returns the minimum-weight error pattern ``e`` (binary, length n) such that
    ``y XOR e`` is a codeword, or ``None`` if decoding fails (weight > t).

    This is the classic RS/BCH algebraic pipeline the Nature paper uses:
      1. compute syndromes S_1..S_{2t}  (linear over GF(2^m))
      2. Berlekamp-Massey -> error-locator polynomial sigma(x)
      3. Chien search -> error positions (roots of sigma)
    """
    gf = code.gf
    S = code.syndromes(y)
    if all(s == 0 for s in S):
        return np.zeros(code.n, dtype=int)   # already a codeword

    # --- Berlekamp-Massey over GF(2^m): syndromes -> error-locator sigma ---
    sigma, L = _bm_locator(gf, S, code.t)

    # --- Chien search: error positions are j with sigma(alpha^{-j}) = 0 ---
    err = np.zeros(code.n, dtype=int)
    nroots = 0
    for j in range(code.n):
        # evaluate sigma at alpha^{-j} = alpha^{n-j}
        xinv = gf.alpha_pow((code.n - j) % code.n)
        val = 0
        xp = 1
        for c in sigma:
            if c != 0:
                val ^= gf.mul(c, xp)
            xp = gf.mul(xp, xinv)
        if val == 0:
            err[j] = 1
            nroots += 1
    if nroots != L:
        return None          # decoding failure (more errors than t)
    return err


# ---------------------------------------------------------------------------
# DQI satisfaction metric — classical evaluation of amplification.
# ---------------------------------------------------------------------------
def build_bch_instance(m: int = 4, t: int = 1, n_vars: int | None = None):
    """Build a DQI max-XORSAT instance from a binary BCH code.

    Returns ``(B, v, code)`` where ``B`` has shape (n_constraints, n_vars),
    ``B^T = H`` (the BCH parity check) and ``v`` is a random target. Constraints
    correspond to BCH parity-check rows; variables are the syndrome bits.

    In DQI terms the dual code is ``C^perp = {d : B^T d = 0}`` = the BCH code,
    and minimum-weight syndrome decoding of ``B^T y`` is BCH decoding.
    """
    code = BCHCode(m, t)
    H = code.H                       # (n_checks, n) over GF(2)
    B = H.T.copy()                   # (m_constraints, n_vars) with m = n, n_vars = n_checks
    if n_vars is not None and n_vars < B.shape[1]:
        B = B[:, :n_vars]
    rng = np.random.default_rng(0)
    v = rng.integers(0, 2, size=B.shape[0])
    return B, v, code


def dqi_satisfaction_stats(B, v, code: BCHCode, ell: int):
    """Classically evaluate DQI amplification on a BCH instance (ground truth).

    This is the *faithful* DQI output distribution, computed by direct
    enumeration of the final state ``|P(f)> = sum_x P(f(x)) |x>`` (paper eq. 2).
    For each candidate solution ``x`` over GF(2)^n we compute ``f(x)`` = number of
    satisfied constraints, and assign it the DQI measurement probability
    ``|P_ell(f(x))|^2`` where ``P_ell`` is the degree-ell optimal polynomial whose
    coefficients are the principal eigenvector ``w`` of the tridiagonal "semicircle"
    matrix (paper eq. 6 / SI section 6, same ``get_optimal_w`` the circuit uses).

    Why this measures the ALGEBRAIC decoder's amplification: reaching this
    distribution REQUIRES uncomputing ``y`` from the syndrome ``B^T y`` for ALL
    weight-<=ell error patterns, i.e. minimum-weight (bounded-distance) decoding of
    the dual code C^perp. For the BCH dual code that uncompute IS Berlekamp-Massey.
    A non-min-weight decoder (GJE) fails to uncompute correctly and the
    distribution collapses toward uniform (mean = random). So this number is the
    amplification the algebraic decoder is responsible for; the circuit section
    below then confirms a real reversible BM/decoder reproduces it.

    Returns mean_satisfied (DQI-weighted), random_mean, max_satisfied, histogram.
    """
    B = np.asarray(B, dtype=int) % 2
    v = np.asarray(v, dtype=int) % 2
    m, n = B.shape

    # Optimal degree-ell polynomial coefficients (Krawtchouk / Hahn basis weights).
    w = _optimal_poly_weights(m, ell)

    # Enumerate all x; weight each by the DQI probability P_ell(f(x))^2.
    total_p = 0.0
    weighted = 0.0
    hist = {}
    for k in range(1 << n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        f = int(np.sum((B @ x) % 2 == v))         # satisfied count in [0, m]
        amp = _poly_amplitude(w, f, m)            # P_ell(f) value
        p = amp * amp
        total_p += p
        weighted += f * p
        hist[f] = hist.get(f, 0.0) + p

    mean = weighted / total_p if total_p else 0.0
    hist = {s: hist[s] / total_p for s in sorted(hist, reverse=True)}
    return {
        "mean_satisfied": mean,
        "random_mean": m / 2.0,
        "max_satisfied": _brute_force_max(B, v) if n <= 22 else None,
        "m": m,
        "n": n,
        "ell": ell,
        "histogram": hist,
    }


def _optimal_poly_weights(m, ell):
    """Degree-ell optimal DQI polynomial weights w_0..w_ell (semicircle eigvec)."""
    # Tridiagonal A_{k,k'} from the binary (p=2,r=1) case used in dqi_gje.py.
    diag = np.zeros(ell + 1)
    off = np.sqrt(np.arange(1, ell + 1) * (m - np.arange(1, ell + 1) + 1))
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    vals, vecs = np.linalg.eigh(A)
    w = vecs[:, np.argmax(vals)]
    w = w / np.linalg.norm(w)
    if w.sum() < 0:
        w = -w
    return w


def _poly_amplitude(w, f, m):
    """Evaluate the DQI degree-ell polynomial at objective value f in [0, m].

    The DQI amplitude on a string x with f satisfied constraints is
    P(f) = sum_k w_k * K_k(d) where d = m - f is the number UNSATISFIED and K_k is
    the (normalized) degree-k Krawtchouk polynomial. We use the standard recurrence.
    """
    ell = len(w) - 1
    d = m - f                                   # number of unsatisfied constraints
    # Normalized Krawtchouk values K_0..K_ell at (d; m).
    K = _krawtchouk_values(ell, d, m)
    return float(np.dot(w, K))


def _krawtchouk_values(ell, d, m):
    """Normalized binary Krawtchouk polynomials K_0..K_ell evaluated at d (size m)."""
    from math import comb
    K = np.zeros(ell + 1)
    for k in range(ell + 1):
        s = 0.0
        for j in range(k + 1):
            s += (-1) ** j * comb(d, j) * comb(m - d, k - j)
        norm = np.sqrt(comb(m, k)) if comb(m, k) else 1.0
        K[k] = s / norm
    return K


def _brute_force_max(B, v):
    B = np.asarray(B, dtype=int) % 2
    v = np.asarray(v, dtype=int) % 2
    m, n = B.shape
    best = 0
    for k in range(1 << n):
        x = np.array([(k >> i) & 1 for i in range(n)], dtype=int)
        best = max(best, int(np.sum((B @ x) % 2 == v)))
    return best


# ===========================================================================
# Reversible DQI circuit with an ALGEBRAIC (Berlekamp-Massey) decoder.
# ===========================================================================
def build_dqi_circuit_algebraic(code: BCHCode, B, v=None, ell=2,
                                with_measurements=False,
                                decoder="enumeration"):
    """DQI max-XORSAT circuit whose decoder is an ALGEBRAIC bounded-distance decoder.

    Layout mirrors ``dqi_gje.build_dqi_circuit_gje`` so resource numbers are
    directly comparable:

      * y       : m = code.n  qubits  (error / dual-codeword register, weight<=ell)
      * syn     : n           qubits  (solution register = syndrome variables)
      * sigma   : ceil(log2(code.n+1)) ancillas, ONLY for t>=2 (locator search)

    Pipeline (paper steps 1-5):
      1. weighted-unary + Dicke prep of weight-<=ell superposition on y
      2. phase sign from target v
      3. B^T y onto the solution register (CX network)
      4. ALGEBRAIC syndrome decode = uncompute y  (the contribution of THIS module)
      5. final Hadamard on the solution register

    For t == 1 (Hamming / single-error-correcting BCH) the algebraic decoder is
    exact and ancilla-free: the m-bit syndrome equals the binary address of the
    single error location, so "uncompute y" is a sequence of multi-controlled-X
    gates that flip each error qubit conditioned on the solution register encoding
    its address. This is the reversible specialization of Berlekamp-Massey at t=1
    (locator sigma(x) = 1 + S_1 x, single root) and is what we BUILD + SIMULATE.

    For t >= 2 we build the same linear stages and a *resource-estimated* locator
    stage (see ``estimate_bm_resources``); the full in-place BM datapath over
    GF(2^m) is the genuine next step and is documented there, not stubbed silently.
    """
    from . import _vendor  # noqa: F401  (puts DQI-Circuit on sys.path)
    from qiskit import QuantumCircuit, QuantumRegister
    from src.dqi.qiskit import UnaryAmplitudeEncoding, UnkGate

    B = np.asarray(B, dtype=int) % 2
    m, n = B.shape
    assert m == code.n, "error register size must equal dual-code length code.n"

    # Optimal degree-ell DQI weights -> max error weight that gets amplitude.
    w = _optimal_poly_weights(m, ell)
    # Pad to next power of two for the unary encoder (matches DQI-Circuit convention).
    import math as _math
    target = 1 << _math.ceil(_math.log2(len(w))) if len(w) > 1 else 1
    w_pad = np.zeros(target)
    w_pad[: len(w)] = np.abs(w)
    w_pad = w_pad / np.linalg.norm(w_pad)
    max_errors = ell

    y = QuantumRegister(m, name="y")
    syn = QuantumRegister(n, name="x")
    use_bm = (decoder == "bm") and code.t == 2
    if use_bm:
        # GENUINE reversible BM datapath: 5 field registers of m qubits.
        anc = QuantumRegister(5 * code.m, name="bm")
        qc = QuantumCircuit(y, syn, anc)
    else:
        anc = None
        qc = QuantumCircuit(y, syn)

    # 1) weighted unary + Dicke (weight<=ell) on y.
    qc.append(UnaryAmplitudeEncoding(m, w_pad[:m] / (np.linalg.norm(w_pad[:m]) or 1)),
              range(m))
    qc.append(UnkGate(m, max_errors), range(m))

    # 2) phase sign from v.
    if v is not None:
        v = np.asarray(v, dtype=int) % 2
        for j in np.nonzero(v)[0]:
            qc.z(int(j))

    # 3) B^T y onto solution register: cx(y_j -> x_i) where B[j, i] == 1.
    for j in range(m):
        for i in range(n):
            if B[j, i]:
                qc.cx(j, m + i)

    # 4) ALGEBRAIC decode (uncompute y).
    if code.t == 1:
        _append_hamming_decode(qc, code, y_offset=0, x_offset=m, n_vars=n)
    elif use_bm:
        # t == 2: GENUINE reversible Berlekamp-Massey/Peterson datapath (the real
        # multi-error decoder). Computes the error-locator polynomial by FIELD
        # ARITHMETIC and uncomputes y via a reversible Chien search. Scales
        # polynomially in t (2 field multiplies + linear maps), unlike the
        # C(m,t)-cost pattern enumeration below. See _append_bm_decode_t2.
        bm_idx = [m + n + i for i in range(anc.size)]
        _append_bm_decode_t2(qc, code, y_idx=list(range(m)),
                             syn_idx=[m + i for i in range(n)], anc_idx=bm_idx)
    else:
        # t >= 2: EXACT reversible minimum-weight decode by pattern enumeration.
        # BUILT (real, verified unitary): for every correctable error pattern e
        # (weight <= min(ell, t)) we flip its support on y conditioned on the
        # syndrome register equalling B^T e. This is exact minimum-weight decoding;
        # cost scales with the NUMBER of correctable patterns (not the efficient BM
        # datapath). It is the Tier-1 construction that proves amplification
        # survives circuitization at t>=2. See estimate_bm_resources for the
        # efficient-datapath cost trend.
        _append_pattern_enumeration_decode(
            qc, code, B, y_offset=0, x_offset=m, n_vars=n,
            max_weight=min(ell, code.t))

    # 5) final Hadamards on solution register.
    for i in range(n):
        qc.h(m + i)

    if with_measurements:
        qc.measure_all()
    return qc


def _append_hamming_decode(qc, code: BCHCode, y_offset, x_offset, n_vars):
    """Reversible single-error (t=1) algebraic decode: uncompute y from the syndrome.

    After stage 3 the solution register holds B^T y (the syndrome of the dual
    code). For a single-error pattern the syndrome uniquely names the error
    position. We recompute, for each codeword position p, the n-bit syndrome
    column ``B[p]`` (= H column) and apply an X on y[p] controlled on the solution
    register matching that column. This flips back exactly the single error,
    uncomputing y. (Multi-error patterns of weight<=ell>1 are outside t=1's
    guarantee; for the Hamming dual code ell is taken == t == 1 for an exact
    circuit, matching the bounded-distance radius.)
    """
    B = code.H.T                      # (m, n_checks): row p is syndrome column of pos p
    m = code.n
    for p in range(m):
        col = B[p][:n_vars]
        ctrl = [x_offset + i for i in range(n_vars)]
        # multi-controlled X with control state = col (open controls where col==0)
        zero_ctrls = [x_offset + i for i in range(n_vars) if col[i] == 0]
        for q in zero_ctrls:
            qc.x(q)
        if len(ctrl) == 0:
            qc.x(y_offset + p)
        elif len(ctrl) == 1:
            qc.cx(ctrl[0], y_offset + p)
        else:
            qc.mcx(ctrl, y_offset + p)
        for q in zero_ctrls:
            qc.x(q)


def _append_pattern_enumeration_decode(qc, code: BCHCode, B, y_offset, x_offset,
                                       n_vars, max_weight):
    """EXACT reversible minimum-weight decode for t>=2 by pattern enumeration (BUILT).

    State after stage 3: the n_vars-qubit solution register holds the syndrome
    ``s = B^T y`` (over GF(2)), where ``y`` is the weight-<=ell error superposition.
    Because the dual code has minimum distance ``2t+1``, every error pattern of
    weight <= t has a UNIQUE syndrome, so we can uncompute ``y`` exactly:

      for each correctable pattern e (weight <= max_weight <= t):
          flip y on supp(e), controlled on (solution register == B^T e).

    The solution register is never modified by this stage, so all controls read
    the original syndrome and gate order is irrelevant. Each n_vars-controlled
    gate fires for exactly one basis syndrome (uniqueness verified in
    ``scripts/algebraic_t2_prototype.py``), so this is the literal inverse of the
    syndrome map on the correctable set -- a correct permutation/unitary.

    COST: O(number of correctable patterns) multi-controlled-X gates, each with
    n_vars controls. This is exact but pattern-enumeration cost, NOT the efficient
    Berlekamp-Massey datapath (whose cost trend is ``estimate_bm_resources``).
    """
    B = np.asarray(B, dtype=int) % 2
    m = code.n
    Bt = B.T                                  # (n_vars, m): syndrome = Bt @ e
    seen = {}
    for w in range(max_weight + 1):
        for supp in itertools.combinations(range(m), w):
            e = np.zeros(m, dtype=int)
            for p in supp:
                e[p] = 1
            s = tuple(int(b) for b in (Bt @ e) % 2)
            if w == 0:
                continue                      # zero error: nothing to uncompute
            assert s not in seen, (
                f"syndrome collision {s} between {seen.get(s)} and {supp}; "
                "code does not have unique decoding at this weight")
            seen[s] = supp
            # multi-controlled-X on each y[p] in supp, control state = s (open
            # controls where the syndrome bit is 0).
            ctrl = [x_offset + i for i in range(n_vars)]
            zero_ctrls = [x_offset + i for i in range(n_vars) if s[i] == 0]
            for q in zero_ctrls:
                qc.x(q)
            for p in supp:
                if len(ctrl) == 1:
                    qc.cx(ctrl[0], y_offset + p)
                else:
                    qc.mcx(ctrl, y_offset + p)
            for q in zero_ctrls:
                qc.x(q)


# ===========================================================================
# GENUINE reversible Berlekamp-Massey datapath (t=2 closed form / Peterson).
#
# This is the REAL multi-error DQI decoder, wired end-to-end (not pattern
# enumeration): given the binary syndrome in the solution register, it computes
# the error-locator polynomial via FIELD ARITHMETIC and uncomputes y by a
# reversible Chien search. It scales POLYNOMIALLY in t (vs C(m,t) for the
# enumeration tier). Built from the verified GF(2^m) multiply-add block plus
# linear (CX-only) field maps; every scratch register is uncomputed to |0>.
#
# Why t=2 has a closed form (Peterson) over a binary BCH code:
#   sigma_1 = S_1
#   sigma_2 = S_3 * S_1^{-1} + S_1^2            (since S_1^3 * S_1^{-1} = S_1^2)
# and S_1^{-1} = S_1^{2^m - 2} = S_1^4 * S_1^2 (Fermat; squarings are GF(2)-linear,
# so only TWO genuine field multiplies are needed: S_1^{-1} and S_3*S_1^{-1}).
# inv(0):=0 falls out of the Fermat form, which is exactly right: the only
# weight-<=2 pattern with S_1=0 is the zero error, where sigma_2 is irrelevant
# (the locator is sigma(x)=1, no roots, nothing flipped).
# ===========================================================================
def _gf2_linear_matrix(gf: GF2m, func) -> np.ndarray:
    """m x m GF(2) matrix M with (func(a))_bits = M @ a_bits, for a GF(2)-linear func.

    Used for squaring (a -> a^2) and constant-multiply (a -> c*a), both of which
    are linear over GF(2) (verified). Columns are the bit-images of the basis
    elements 2^i.
    """
    m = gf.m
    cols = []
    for i in range(m):
        y = func(1 << i)
        cols.append([(y >> k) & 1 for k in range(m)])
    return np.array(cols, dtype=int).T % 2


def bm_t2_matrices(code: BCHCode) -> dict:
    """Precompute the GF(2) linear maps the t=2 BM datapath needs (any GF(2^m)).

    Returns:
      * ``T1``, ``T3`` : (m x n_checks) extract field syndromes S_1, S_3 from the
        binary syndrome register (S_i = T_i @ s);
      * ``M2``        : (m x m) squaring matrix (a -> a^2);
      * ``C1[j]``     : (m x m) constant-multiply by alpha^{-j}   (j = 0..n-1);
      * ``C2[j]``     : (m x m) constant-multiply by alpha^{-2j}.
    """
    gf = code.gf
    m, n = code.m, code.n
    H = code.H
    G1 = np.array([[(gf.alpha_pow(j) >> k) & 1 for j in range(n)] for k in range(m)])
    G3 = np.array([[(gf.alpha_pow((3 * j) % gf.n) >> k) & 1 for j in range(n)]
                   for k in range(m)])
    T1 = gf2.solve_transform(H, G1)
    T3 = gf2.solve_transform(H, G3)
    M2 = _gf2_linear_matrix(gf, lambda a: gf.mul(a, a))
    C1, C2 = [], []
    for j in range(n):
        ainv = gf.alpha_pow((gf.n - j) % gf.n)
        C1.append(_gf2_linear_matrix(gf, lambda a, c=ainv: gf.mul(c, a)))
        C2.append(_gf2_linear_matrix(gf, lambda a, c=gf.mul(ainv, ainv): gf.mul(c, a)))
    return {"T1": T1, "T3": T3, "M2": M2, "C1": C1, "C2": C2}


def _append_gf2_linear(qc, M, src, dst):
    """dst ^= M @ src  (CX network; M is (len(dst) x len(src)) over GF(2))."""
    M = np.asarray(M, dtype=int) % 2
    for k in range(M.shape[0]):
        for i in range(M.shape[1]):
            if M[k, i]:
                qc.cx(src[i], dst[k])


def _append_gf2m_inverse(qc, mul, M2, src, dst, sq1, sq2):
    """dst ^= src^{-1} via Fermat squaring (src^4 * src^2 = src^{-1} over GF(2^3)).

    Shared by the t=2 BM datapath (``S1^{-1}``) and the t=1 symbol-level RS/Forney
    decoder (``S2^{-1}``): both compute a field inverse by two GF(2)-linear
    squarings (M2) and one multiply-add (``mul``), then uncompute the two square
    scratch registers ``sq1, sq2`` back to |0>. ``mul`` is the reversible
    multiply-add ``build_gf2m_mul_add_circuit(m)``; it self-inverts over GF(2),
    so calling this twice (forward then again) clears ``dst`` as well.
    """
    _append_gf2_linear(qc, M2, src, sq1)   # sq1 = src^2
    _append_gf2_linear(qc, M2, sq1, sq2)   # sq2 = (src^2)^2 = src^4
    qc.compose(mul, list(sq2) + list(sq1) + list(dst), inplace=True)  # dst ^= src^6
    _append_gf2_linear(qc, M2, sq1, sq2)   # uncompute sq2 -> 0
    _append_gf2_linear(qc, M2, src, sq1)   # uncompute sq1 -> 0


def _append_bm_decode_t2(qc, code: BCHCode, y_idx, syn_idx, anc_idx):
    """GENUINE reversible t=2 Berlekamp-Massey/Peterson decode (uncompute y).

    Operates in place on an existing circuit. Reads the binary syndrome from the
    solution register ``syn_idx`` (never modifies it), computes sigma_1, sigma_2
    by field arithmetic, runs a reversible Chien search to flip every error
    position of ``y_idx``, and uncomputes ALL scratch back to |0>.

    Register layout in ``anc_idx`` (5 field registers of m qubits = 5*m ancillas):
      S1 = anc[0:m]   (= sigma_1, persistent through Chien)
      sg2= anc[m:2m]  (= sigma_2, persistent through Chien)
      w0 = anc[2m:3m] (scratch: holds S1^{-1} through Chien)
      w1 = anc[3m:4m] (scratch: S3 / Chien accumulator)
      w2 = anc[4m:5m] (scratch: squaring temporaries)
    The Chien accumulator uses w1; w0 carries S1^{-1} so sigma_2 can be uncomputed
    after the search without recomputing the inverse.
    """
    from qiskit import QuantumCircuit
    m = code.m
    mats = bm_t2_matrices(code)
    T1, T3, M2, C1, C2 = mats["T1"], mats["T3"], mats["M2"], mats["C1"], mats["C2"]
    S1 = anc_idx[0:m]
    sg2 = anc_idx[m:2 * m]
    w0 = anc_idx[2 * m:3 * m]
    w1 = anc_idx[3 * m:4 * m]
    w2 = anc_idx[4 * m:5 * m]
    mul = build_gf2m_mul_add_circuit(m)

    def MUL(a, b, c):
        # c ^= a*b   (its own inverse over GF(2))
        qc.compose(mul, list(a) + list(b) + list(c), inplace=True)

    def inv_S1_into_w0():
        # w0 ^= S1^{-1} = S1^4 * S1^2 ; uses w1=S1^2, w2=S1^4 then clears them.
        _append_gf2m_inverse(qc, mul, M2, S1, w0, w1, w2)

    # --- extract sigma_1 = S1 from the syndrome register ---
    _append_gf2_linear(qc, T1, syn_idx, S1)

    # --- w0 = S1^{-1} ---
    inv_S1_into_w0()

    # --- sigma_2 = S3 * S1^{-1} + S1^2 ---
    _append_gf2_linear(qc, T3, syn_idx, w1)   # w1 = S3
    MUL(w1, w0, sg2)                           # sg2 = S3 * S1^{-1}
    _append_gf2_linear(qc, M2, S1, sg2)        # sg2 += S1^2  -> sigma_2
    _append_gf2_linear(qc, T3, syn_idx, w1)   # uncompute w1 (S3) -> 0

    # --- Chien search: flip y[j] iff sigma(alpha^{-j}) == 0 ---
    for j in range(code.n):
        qc.x(w1[0])                            # w1 = field element "1"
        _append_gf2_linear(qc, C1[j], S1, w1)  # += sigma_1 * alpha^{-j}
        _append_gf2_linear(qc, C2[j], sg2, w1) # += sigma_2 * alpha^{-2j}
        # flip y[j] iff w1 == 0  (open-controlled MCX)
        for q in w1:
            qc.x(q)
        if m == 1:
            qc.cx(w1[0], y_idx[j])
        elif m == 2:
            qc.ccx(w1[0], w1[1], y_idx[j])
        else:
            qc.mcx(list(w1), y_idx[j])
        for q in w1:
            qc.x(q)
        # uncompute the Chien accumulator (reverse the linear maps + the X)
        _append_gf2_linear(qc, C2[j], sg2, w1)
        _append_gf2_linear(qc, C1[j], S1, w1)
        qc.x(w1[0])

    # --- uncompute sigma_2 (mirror of its construction; S1^{-1} still in w0) ---
    _append_gf2_linear(qc, T3, syn_idx, w1)
    _append_gf2_linear(qc, M2, S1, sg2)
    MUL(w1, w0, sg2)
    _append_gf2_linear(qc, T3, syn_idx, w1)

    # --- uncompute w0 (S1^{-1}) and S1 ---
    inv_S1_into_w0()                           # self-inverse: w0 -> 0
    _append_gf2_linear(qc, T1, syn_idx, S1)    # S1 -> 0


# ===========================================================================
# SYMBOL-LEVEL Reed-Solomon decode with FORNEY magnitudes (closes the
# real-valued-immunization caveat at the circuit level).
#
# The binary-BCH path answers a COMBINATORIAL question ("which bond is the
# odd-one-out in the moment-syndrome bit pattern") — error magnitudes are always
# 1, so Forney is trivial and never built. A LITERAL Reed-Solomon immunization
# carries a GF(2^m) SYMBOL per bond (the real-valued amount, reduced into the
# field), so each error has both a LOCATION and a MAGNITUDE. Recovering the
# magnitude is Forney's algorithm — the genuinely-new reversible primitive here.
#
# t=1 closed form (narrow-sense b=1):  X = sigma_1 = S_2 / S_1   (location),
#   Y = Omega(X^{-1}) / sigma'(X^{-1}) = S_1^2 / S_2   (Forney magnitude).
# Squarings are GF(2)-linear (CX); the two divisions are Fermat inverses
# (a^{-1}=a^{2^m-2}) + one multiply each. inv(0):=0, so the no-error case
# (S_1=S_2=0 -> Y=0) flips nothing. The classical t=2 symbol decoder (2x2 locator
# solve + Forney) is verified in scripts/rs_forney_symbol.py and is the
# documented scale-up.
# ===========================================================================
def _append_rs_decode_t1_symbol(qc, gf: GF2m, ys_idx, S1_idx, S2_idx, anc_idx):
    """Reversible t=1 symbol-level RS decode: uncompute a single SYMBOL error.

    Given the field syndromes S_1, S_2 (preserved input registers) of a weight-<=1
    symbol error living in the symbol register ``ys`` (n field symbols of m qubits
    each), compute the Forney magnitude Y = S_1^2 / S_2 and XOR it back out of the
    symbol at the (implicitly located) error position, uncomputing ys. All scratch
    is restored to |0>.

    Args:
      ys_idx : list of n lists, each m qubit indices (symbol register, position p)
      S1_idx, S2_idx : m qubit indices each (field syndromes, NOT modified)
      anc_idx: 3*m + 1 ancilla indices: Y=anc[0:m], tmp=anc[m:2m],
               w=anc[2m:3m] (squaring scratch), flag=anc[3m]
    """
    m = gf.m
    n = gf.n
    Y = anc_idx[0:m]
    tmp = anc_idx[m:2 * m]
    w = anc_idx[2 * m:3 * m]
    flag = anc_idx[3 * m]
    mul = build_gf2m_mul_add_circuit(m)
    M2 = _gf2_linear_matrix(gf, lambda a: gf.mul(a, a))

    def MUL(a, b, c):
        qc.compose(mul, list(a) + list(b) + list(c), inplace=True)

    # --- specialised, correct m=3 path (a^{-1} = a^4 * a^2 = a^6) ---
    assert m == 3, "reversible Forney t=1 built for GF(2^3); see scale-up note"

    def inv3(src, dst, sq1, sq2):
        # dst ^= src^{-1}; sq1, sq2 scratch (each m qubits), restored to |0>.
        _append_gf2m_inverse(qc, mul, M2, src, dst, sq1, sq2)

    # Y = S1^2 * S2^{-1}. Use tmp for S2^{-1}, w + (reuse Y region? no) for squares.
    # Allocate squaring scratch from w; need a 2nd scratch — reuse tmp's slot after.
    # Layout: compute invS2 into tmp using w and Y as square scratch (Y still 0 here).
    inv3(S2_idx, tmp, w, Y)                      # tmp = S2^{-1}  (w, Y scratch -> 0)
    # Y = S1^2 * invS2 :  first put S1^2 into w (linear), then Y ^= w * tmp
    _append_gf2_linear(qc, M2, S1_idx, w)        # w = S1^2
    MUL(w, tmp, Y)                               # Y = S1^2 * S2^{-1}  (Forney magnitude)
    _append_gf2_linear(qc, M2, S1_idx, w)        # uncompute w -> 0

    # For each position p: error iff S2 == S1 * alpha^p  (i.e. X == alpha^p).
    # XOR the magnitude Y into ys[p] conditioned on that equality.
    for p in range(n):
        Cp = _gf2_linear_matrix(gf, lambda a, c=gf.alpha_pow(p): gf.mul(c, a))
        # tmp2 := S2 XOR (alpha^p * S1)  -> 0 iff equal. Reuse w as tmp2.
        for k in range(m):
            qc.cx(S2_idx[k], w[k])               # w = S2
        _append_gf2_linear(qc, Cp, S1_idx, w)    # w ^= alpha^p * S1
        # flag = (w == 0)
        for q in w:
            qc.x(q)
        qc.mcx(list(w), flag)
        for q in w:
            qc.x(q)
        # ys[p] ^= Y   controlled on flag
        for k in range(m):
            qc.ccx(flag, Y[k], ys_idx[p][k])
        # uncompute flag and w
        for q in w:
            qc.x(q)
        qc.mcx(list(w), flag)
        for q in w:
            qc.x(q)
        _append_gf2_linear(qc, Cp, S1_idx, w)
        for k in range(m):
            qc.cx(S2_idx[k], w[k])

    # uncompute Y
    _append_gf2_linear(qc, M2, S1_idx, w)
    MUL(w, tmp, Y)
    _append_gf2_linear(qc, M2, S1_idx, w)
    inv3(S2_idx, tmp, w, Y)                      # tmp -> 0


# ===========================================================================
# TIER 2 — efficient reversible Berlekamp-Massey datapath sub-blocks.
#
# These are BUILT, VERIFIED reversible primitives over GF(2^m). They let us
# replace the hand-waved "~q^2 Toffoli per multiply" in estimate_bm_resources
# with MEASURED per-block gate counts, turning the resource estimate from an
# analytic guess into "measured sub-blocks + documented assembly".
# ===========================================================================
def _gf2m_reduction_table(m: int) -> list[int]:
    """For k in [0, 2m-2], the m-bit value of x^k reduced mod the primitive poly."""
    poly = GF2m._PRIM_POLY[m]
    q = 1 << m
    reduced = []
    for k in range(2 * m - 1):
        v = 1 << k
        for b in range(2 * m - 2, m - 1, -1):
            if v & (1 << b):
                v ^= poly << (b - m)
        reduced.append(v & (q - 1))
    return reduced


def build_gf2m_mul_add_circuit(m: int):
    """BUILT + VERIFIED reversible GF(2^m) multiply-add: |a>|b>|c> -> |a>|b>|c + a*b>.

    Schoolbook construction: a*b = sum_{i,j} a_i b_j x^{i+j}; each x^{i+j} reduces
    to an m-bit constant ``reduced[i+j]`` (precomputed from the primitive poly), so
    the partial product a_i*b_j XORs into output bit k for every set bit k of
    reduced[i+j]. That is one Toffoli (CCX) ``c_k ^= a_i & b_j`` per such (i,j,k).

    Registers (3*m qubits): a[0..m), b[m..2m), c[2m..3m). ``c`` must be the addend
    (the result is added in place, GF(2) addition = XOR), so passing |c=0> gives
    the plain product. Verified EXACTLY against ``GF2m.mul`` for all input pairs at
    m in {3,4} (see scripts/algebraic_t2_prototype.py).
    """
    from . import _vendor  # noqa: F401
    from qiskit import QuantumCircuit, QuantumRegister
    reduced = _gf2m_reduction_table(m)
    a = QuantumRegister(m, "a")
    b = QuantumRegister(m, "b")
    c = QuantumRegister(m, "c")
    qc = QuantumCircuit(a, b, c, name=f"GFmul2^{m}")
    for i in range(m):
        for j in range(m):
            r = reduced[i + j]
            for k in range(m):
                if (r >> k) & 1:
                    qc.ccx(a[i], b[j], c[k])
    return qc


def build_syndrome_circuit(code: "BCHCode", power: int):
    """BUILT reversible syndrome accumulator for S_{power} of a binary word y.

    For a binary error word y of length n, the GF(2^m) syndrome
    ``S_i = XOR_{j : y_j=1} alpha^{i*j}`` is LINEAR in y: y_j toggles a fixed
    field constant alpha^{i*j} into the accumulator. So the whole block is a CX
    network (no Toffoli): for each position j and each set bit k of alpha^{i*j},
    a CX from y_j into syndrome-accumulator bit k.

    Registers: y[0..n) then an m-qubit accumulator acc[n..n+m). Verified against
    ``BCHCode.syndromes`` in scripts/algebraic_t2_prototype.py.
    """
    from . import _vendor  # noqa: F401
    from qiskit import QuantumCircuit, QuantumRegister
    gf = code.gf
    y = QuantumRegister(code.n, "y")
    acc = QuantumRegister(code.m, "acc")
    qc = QuantumCircuit(y, acc, name=f"S{power}")
    for j in range(code.n):
        val = gf.alpha_pow((power * j) % gf.n)
        for k in range(code.m):
            if (val >> k) & 1:
                qc.cx(y[j], acc[k])
    return qc


def measure_bm_subblocks(m_field: int, t: int = 2):
    """MEASURE the BUILT Tier-2 sub-blocks (transpiled to IQM basis) -> per-block costs.

    Returns measured {toffoli, cz, depth, qubits} for the GF(2^m) multiply-add and
    for the two syndrome accumulators S_1, S_2 of a t=2 BCH code over the field.
    These measured numbers feed ``estimate_bm_resources`` so its multiplier cost is
    no longer a ``q^2`` guess but a transpiled count.
    """
    from . import _vendor  # noqa: F401
    from qiskit import transpile
    from .metrics import gate_stats
    from .bases import IQM_BASIS as IQM

    mul = build_gf2m_mul_add_circuit(m_field)
    mul_raw = gate_stats(mul)
    mul_iqm = gate_stats(transpile(mul, basis_gates=IQM, optimization_level=3))

    code = BCHCode(m_field, t)
    syn_iqm = []
    for p in (1, 2):
        s = build_syndrome_circuit(code, p)
        syn_iqm.append(gate_stats(transpile(s, basis_gates=IQM, optimization_level=3)))

    return {
        "field": f"GF(2^{m_field})",
        "mul_add": {
            "toffoli_raw": mul_raw["ops"].get("ccx", 0),
            "cz_iqm": mul_iqm["2q_gates"],
            "depth_iqm": mul_iqm["depth"],
            "qubits": mul_raw["qubits"],
        },
        "syndrome_S1": syn_iqm[0],
        "syndrome_S2": syn_iqm[1],
    }


def estimate_bm_resources(m_field: int, t: int):
    """Resource estimate for an in-place reversible Berlekamp-Massey decoder.

    REFINED (was a pure analytic guess): the dominant cost is the GF(2^m)
    multiply-add, whose per-block Toffoli AND transpiled-CZ cost we now MEASURE
    from the BUILT+VERIFIED ``build_gf2m_mul_add_circuit`` (see
    ``measure_bm_subblocks``) instead of assuming ``q^2`` Toffoli. The ASSEMBLY
    (how many multiplies BM + Chien need) is still an analytic count of the
    standard datapath and is labeled ESTIMATED; the per-multiply cost is MEASURED.

    Structure (narrow-sense binary BCH, capability t over GF(2^m)):
      * 2t syndrome accumulators (linear CX networks — measured separately);
      * BM key-equation solve: O(t) discrepancy updates, each O(t) multiply-adds;
      * Chien search: evaluate the degree-t locator at 2^m field points,
        ~t multiply-adds per point (this term dominates and is the obvious target
        for the syndrome-table / list-decoding optimization at small q).
    """
    from qiskit import transpile
    from .metrics import gate_stats
    from .bases import IQM_BASIS as IQM
    q = m_field

    # --- MEASURED per-multiply cost (BUILT + VERIFIED block) ---
    mul = build_gf2m_mul_add_circuit(q)
    toffoli_per_mult = gate_stats(mul)["ops"].get("ccx", 0)        # MEASURED
    cz_per_mult = gate_stats(transpile(mul, basis_gates=IQM,
                                       optimization_level=3))["2q_gates"]  # MEASURED

    # --- ESTIMATED assembly: how many multiplies the datapath needs ---
    mults_bm = 2 * t * t                       # BM discrepancy/locator updates
    mults_chien = (2 ** q) * t                 # Chien: 2^m points, ~t mults each
    total_mults = mults_bm + mults_chien

    # registers: 2t syndromes + locator (<= t+1 coeffs) + scratch, each m bits.
    n_field_regs = 2 * t + (t + 1) + 3
    qubits = n_field_regs * q

    toffoli = total_mults * toffoli_per_mult
    cz = total_mults * cz_per_mult
    return {
        "field": f"GF(2^{q})",
        "t": t,
        "estimated_qubits": qubits,
        "toffoli_per_mult_MEASURED": toffoli_per_mult,
        "cz_per_mult_MEASURED": cz_per_mult,
        "n_mults_ESTIMATED": total_mults,
        "estimated_toffoli": toffoli,
        "estimated_cz": cz,
        "note": "per-multiply cost MEASURED from build_gf2m_mul_add_circuit "
                "(verified exact); multiply COUNT is the analytic BM+Chien "
                "assembly (ESTIMATED). Chien (2^m points) dominates and is the "
                "obvious target for the syndrome-table optimization at small q.",
    }
