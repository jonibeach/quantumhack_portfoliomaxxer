"""Minimal GF(2) linear algebra shared by the algebraic-decoder modules.

Three hand-rolled Gaussian-elimination variants used to live in
``dqi_algebraic`` (row-space basis, transform solve) and ``immunization``
(syndrome back-substitution). They are consolidated here unchanged — the
algorithms are byte-for-byte identical to the originals so the BCH parity check,
the t=2 BM linear maps, and the decoded portfolios are all preserved exactly.
"""

import numpy as np

__all__ = ["row_reduce", "solve", "solve_transform"]


def row_reduce(H):
    """Return the independent rows of ``H`` over GF(2) (a row-space basis)."""
    A = (np.asarray(H, dtype=int) % 2).copy()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == rows:
            break
    return A[:r]


def solve(A, b):
    """Return one GF(2) solution ``x`` of ``A x = b`` (free vars 0), or None.

    Generalizes the syndrome back-substitution: forward-eliminate ``A`` (and the
    same row ops on ``b``), check consistency, then read the pivot variables.
    Returns ``None`` if ``b`` is not in the column space of ``A``.
    """
    A = (np.asarray(A, dtype=int) % 2).copy()
    b = (np.asarray(b, dtype=int) % 2).copy()
    rows, cols = A.shape
    x = np.zeros(cols, dtype=int)
    r = 0
    pivots = []
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        b[[r, piv]] = b[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
                b[i] ^= b[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if b[i]:
            return None
    for idx, c in enumerate(pivots):
        x[c] = b[idx]
    return x


def solve_transform(H, G):
    """Return ``T`` over GF(2) with ``T @ H == G`` (rows of G in rowspace(H)).

    Used to express the field syndromes ``S_i = G_i @ y`` as linear functions of
    the binary syndrome register ``s = H @ y`` (``S_i = T_i @ s``).
    """
    H = np.asarray(H, dtype=int) % 2
    G = np.asarray(G, dtype=int) % 2
    Ht = H.T % 2
    rows, cols = Ht.shape
    T = []
    for g in G:
        A = np.concatenate([Ht, g.reshape(-1, 1)], axis=1) % 2
        r = 0
        piv = []
        for c in range(cols):
            p = None
            for i in range(r, rows):
                if A[i, c]:
                    p = i
                    break
            if p is None:
                continue
            A[[r, p]] = A[[p, r]]
            for i in range(rows):
                if i != r and A[i, c]:
                    A[i] ^= A[r]
            piv.append(c)
            r += 1
            if r == rows:
                break
        for i in range(r, rows):
            if A[i, -1]:
                raise RuntimeError("syndrome S_i not in rowspace of H — not BCH?")
        x = np.zeros(cols, dtype=int)
        for idx, c in enumerate(piv):
            x[c] = A[idx, -1]
        T.append(x)
    return np.array(T, dtype=int) % 2
