"""Build DQI max-XORSAT circuits via the vendored BCG-X pipeline."""

import numpy as np

from . import _vendor  # noqa: F401  (side effect: puts bcg-dqi on sys.path)
from pipelines.DQI_full_circuit import dqi_max_xorsat, get_optimal_w

__all__ = ["build_dqi_circuit", "get_optimal_w"]


def build_dqi_circuit(B, v, ell=2, bp_iters=1, with_measurements=False):
    """Construct the DQI quantum circuit for a max-XORSAT instance ``Bx = v``.

    Parameters
    ----------
    B : array_like, shape (m, n)
        Binary constraint matrix over GF(2): m constraints, n variables.
    v : array_like, shape (m,)
        Binary target vector.
    ell : int
        DQI degree / Dicke weight (decoding depth). Higher biases harder toward
        good solutions but costs post-selection and depth.
    bp_iters : int
        Belief-propagation decoder iterations.
    with_measurements : bool
        If True, append measurements on all qubits (data, bt, ancilla).

    Returns
    -------
    qiskit.QuantumCircuit
        The DQI circuit. Qubit count = (bp_iters + 1) * m + n + 2*ceil(log2(t+1)),
        where t = max nonzeros per row of B.
    """
    B = np.asarray(B, dtype=int)
    v = np.asarray(v, dtype=int)
    W_k = get_optimal_w(B.shape[0], ell)
    qc, data_qubits, bt_qubits = dqi_max_xorsat(B, v, W_k, bp_iters)

    if with_measurements:
        from qiskit import ClassicalRegister

        creg = ClassicalRegister(qc.num_qubits, "meas")
        qc.add_register(creg)
        qc.measure(range(qc.num_qubits), creg)

    return qc
