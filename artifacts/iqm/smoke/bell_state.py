"""Phase 1 smoke test: 2-qubit Bell pair on VTT Q50 via FiQCI.

Goal: verify the LUMI -> Q50 access path works end-to-end before we
commit a real DQI run. Prints plumbing details before submitting so we
can debug from the slurm log if anything goes wrong.
"""

import os
import datetime

from qiskit import QuantumCircuit, QuantumRegister, transpile
from iqm.qiskit_iqm import IQMProvider


def main() -> None:
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    cortex_url = os.environ.get("Q50_CORTEX_URL")
    print(f"[plumbing] timestamp_utc = {timestamp}")
    print(f"[plumbing] Q50_CORTEX_URL = {cortex_url}")

    provider = IQMProvider(cortex_url, quantum_computer="q50")
    backend = provider.get_backend()
    print(f"[plumbing] backend.name = {backend.name}")
    try:
        nq = backend.num_qubits
    except AttributeError:
        nq = getattr(backend, "n_qubits", "?")
    print(f"[plumbing] backend.num_qubits = {nq}")

    shots = 1000
    qreg = QuantumRegister(2, "QB")
    circuit = QuantumCircuit(qreg, name="Bell_pair")
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure_all()

    transpiled = transpile(circuit, backend)
    print(f"[plumbing] transpiled.depth = {transpiled.depth()}")
    print(f"[plumbing] transpiled.count_ops = {dict(transpiled.count_ops())}")
    print(f"[plumbing] transpiled.num_qubits = {transpiled.num_qubits}")

    job = backend.run(transpiled, shots=shots)
    print(f"[plumbing] job submitted, id = {getattr(job, 'job_id', lambda: '?')()}")

    result = job.result()
    counts = result.get_counts()
    print(f"[result] counts = {counts}")


if __name__ == "__main__":
    main()
