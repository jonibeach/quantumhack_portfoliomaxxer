"""Shared measurement readout: satisfied-constraint scoring of shot counts.

Every simulate/interpret path in the project does the same thing to a qiskit
counts dict: reverse each measured bitstring to little-endian, slice out the
solution register, count satisfied constraints ``(B @ x) % 2 == v``, and build
the satisfied-count histogram (plus the per-solution distribution). This one
helper replaces the ~half-dozen near-identical copies of that loop.
"""

import numpy as np

__all__ = ["score_counts"]


def score_counts(counts, B, v, x_offset=0, clip_nonpositive=False):
    """Score a qiskit ``counts`` dict by satisfied-constraint count.

    Parameters
    ----------
    counts : dict[str, int | float]
        Measured bitstring -> weight. Integer shot counts (AerSimulator/Q50) or
        float quasi-probabilities (EMS) both work; the accumulator preserves the
        value type, so integer inputs yield integer weights exactly as before.
    B, v : array_like
        The max-XORSAT instance (``m`` constraints x ``n`` variables) and target.
    x_offset : int
        Index of the first solution-register bit in the reversed bitstring. ``0``
        when the solution register is measured alone (binary-collapse circuits);
        ``m`` when an error register ``y`` precedes it (the full DQI circuits).
    clip_nonpositive : bool
        Drop entries with weight <= 0 (used for EMS quasi-probabilities).

    Returns
    -------
    dict with keys ``mean`` (DQI-weighted mean satisfied), ``hist`` (satisfied
    count -> weight), ``sol_hist`` (tuple(x) -> weight) and ``total`` (sum of
    weights). ``hist``/``sol_hist`` keys/types match the historical inline loops.
    """
    B = np.asarray(B, dtype=int) % 2
    v = np.asarray(v, dtype=int) % 2
    m, n = B.shape
    hist = {}
    sol_hist = {}
    total = 0
    weighted = 0
    for bs, c in counts.items():
        if clip_nonpositive and c <= 0:
            continue
        bits = bs.replace(" ", "")[::-1]
        x = np.array([int(bits[x_offset + i]) for i in range(n)])
        s = int(np.sum((B @ x) % 2 == v))
        hist[s] = hist.get(s, 0) + c
        key = tuple(x)
        sol_hist[key] = sol_hist.get(key, 0) + c
        total += c
        weighted += s * c
    mean = weighted / total if total else 0.0
    return {"mean": mean, "hist": hist, "sol_hist": sol_hist, "total": total}
