"""Build DQI max-XORSAT circuits with the light Gauss-Jordan (GJE) decoder.

The BCG pipeline (``dqi.py``) uses a *quantum belief-propagation* decoder, whose
reversible message-passing makes the circuit deep (~500 two-qubit gates at 20
qubits). The DQI-Circuit repo offers an alternative: implement the syndrome
decode as a **reversible classical Gauss-Jordan elimination** over GF(2) — just
a fixed sequence of CX/SWAP row operations, with **no ancilla qubits**.

This is the same "decode the dual problem with a structure-matched classical
decoder, run reversibly" idea the Nature paper describes; GJE is the cheapest
such decoder (valid when the dual code's parity-check is full-rank, i.e. the
syndrome determines the error uniquely — unique syndrome decoding).

CAVEAT (measured, see scripts/verify_gje.py): GJE is plain reversible RREF, i.e.
*linear-system* decoding, NOT minimum-weight decoding. It uncomputes the error
register but does not broadly amplify high-satisfaction solutions — on small
random instances the solution register comes out ~uniform, mean satisfied ~=
random, whereas the BP decoder (dqi.py) reaches mean ~5.2/6 on the 6x4 instance.
The lesson matches the paper: the decoder is simultaneously the source of DQI's
optimization power AND its dominant gate cost. Use GJE when you want a
hardware-runnable DQI *circuit* (10 qubits / ~69 CZ vs 20 / ~509 for BP), and a
structured code whose unique decoding equals minimum-weight decoding.
"""

import numpy as np

from . import _vendor  # noqa: F401  (side effect: puts DQI-Circuit on sys.path)
from qiskit import QuantumCircuit, QuantumRegister
from src.dqi.qiskit import UnaryAmplitudeEncoding, UnkGate, GJEGate
from src.dqi.qiskit import get_optimal_w as _gje_optimal_w

__all__ = ["build_dqi_circuit_gje"]


def build_dqi_circuit_gje(B, v=None, ell=2, with_measurements=False):
    """Construct a DQI circuit for ``B x = v`` using the GJE decoder.

    ``B`` follows the same convention as :func:`dqi_portfolio.build_dqi_circuit`:
    shape ``(m, n)`` with ``m`` constraints and ``n`` variables over GF(2).
    (In DQI-Circuit's own naming this matrix is ``H``; its ``B`` is ``H.T``.)

    Qubit count = ``m + n`` exactly — no ancillas, in contrast to the BP decoder
    which needs ``(bp_iters + 1) * m + n + 2*ceil(log2(t+1))``.

    Parameters
    ----------
    B : array_like, shape (m, n)
        Binary constraint matrix.
    v : array_like, optional
        Binary target vector (length m). GJE row-ops do not depend on ``v``;
        it is accepted for API parity and applied as the phase sign on ``y``.
    ell : int
        DQI degree / Dicke weight (decoding depth).
    with_measurements : bool
        If True, append ``measure_all``.
    """
    H = np.asarray(B, dtype=int)          # (m, n): m constraints/checks, n variables
    m, n = H.shape

    # Weighted unary amplitudes for the optimal degree-ell DQI state (p=2, r=1).
    w = _gje_optimal_w(m, ell, 2, 1)
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0

    y = QuantumRegister(m, name="y")              # error/constraint register
    syn = QuantumRegister(n, name="syndrome")     # variable register
    qc = QuantumCircuit(y, syn)

    # 1) Weighted-unary + Dicke state of weight <= max_errors on y.
    qc.append(UnaryAmplitudeEncoding(m, w), range(m))
    qc.append(UnkGate(m, max_errors), range(m))

    # 2) Optional phase sign from the target v (Z on selected y qubits).
    if v is not None:
        v = np.asarray(v, dtype=int)
        for j in np.nonzero(v % 2)[0]:
            qc.z(int(j))

    # 3) B^T y onto the syndrome register: cx(y_j -> syndrome_i) where H[j, i].
    Bt = H.T                                       # (n, m)
    for i in range(n):
        for j in range(m):
            if Bt[i][j] == 1:
                qc.cx(j, m + i)

    # 4) Reversible Gauss-Jordan decode on the y register (no ancillas).
    gje = QuantumCircuit(m, name="GJE")
    gje.append(GJEGate(H), range(m))
    qc.compose(gje, qubits=range(m), inplace=True)

    # 5) Final Hadamard transform on the syndrome register.
    for i in range(n):
        qc.h(m + i)

    if with_measurements:
        qc.measure_all()
    return qc
