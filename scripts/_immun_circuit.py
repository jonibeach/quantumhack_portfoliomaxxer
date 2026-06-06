"""Shared builders for the t=1 binary-collapse immunization circuits.

``immunization_binary.py`` (GF(2^3), 3 qubits) and ``immunization_binary_gf5.py``
(GF(2^5), 5 qubits) were ~80% identical: the GF(2^5) script was just the GF(2^3)
one generalized from a hard-coded ``n=3`` to a parametric ``n = B.shape``. Those
parametric builders now live here once, and both scripts import them; each keeps
only its own narrative/assertions/artifact-packaging ``main``.

The whole construction rests on the t=1 syndrome-basis collapse: for a single
error over GF(2) the location->syndrome map is a bijection, so the error register,
syndrome computation and reversible decode all fold into the n-qubit syndrome
register, and DQI reduces to ``prepare amplitude vector a -> H^n -> measure``.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from dqi_portfolio import gate_stats
from dqi_portfolio.bases import IQM_BASIS, SIM_BASIS
from dqi_portfolio.readout import score_counts

__all__ = [
    "build_amplitude_vector",
    "v_phase_positions",
    "build_structured",
    "build_minimal",
    "statevector_marginal",
    "shots_score",
    "routed_cz",
]


# ---------------------------------------------------------------------------
# DQI prescription -> pre-Hadamard amplitude vector (computed, NOT hardcoded).
# ---------------------------------------------------------------------------
def build_amplitude_vector(B, v, w):
    """Compute the pre-H n-qubit amplitude vector a[s] from the DQI prescription.

    Index s = 0..2^n-1 is the SYNDROME integer. The DQI ell=1 state is a weighted
    sum of a weight-0 (no-error) term and the weight-1 (single-error) terms:

      * weight-0 term lives on syndrome s=0, with magnitude w[0]/sqrt(C(m,0)) =
        w[0] (one degree-0 dual codeword). For uniform w that is 1/sqrt(2).
      * each single error at position p contributes to syndrome s_p = B[p] (a
        BIJECTION onto {1..m}) with magnitude w[1]/sqrt(C(m,1)) = w[1]/sqrt(m)
        and a sign (-1)^{v_p} from the DQI v-phase.

    a is therefore: a[0] = w[0];  a[s_p] = (w[1]/sqrt m) * (-1)^{v_p}. With uniform
    weights it is normalized: w[0]^2 + m*(w[1]/sqrt m)^2 = w[0]^2 + w[1]^2 = 1.
    """
    m, n = B.shape
    w0 = abs(float(w[0]))
    w1 = abs(float(w[1]))
    a = np.zeros(1 << n, dtype=float)
    # weight-0 (no-error) term on syndrome 0.   C(m,0) = 1
    a[0] = w0
    # weight-1 terms: one per bond position, placed at its syndrome with v-sign.
    mag1 = w1 / np.sqrt(m)  # C(m,1) = m
    syndromes = []
    for p in range(m):
        s_p = int(sum(int(B[p][i]) << i for i in range(n)))
        syndromes.append(s_p)
        a[s_p] = mag1 * ((-1.0) ** int(v[p]))
    # ASSERT the t=1 location->syndrome map is a permutation of {1..m}.
    assert sorted(syndromes) == list(range(1, m + 1)), (
        f"syndrome map is not a bijection onto 1..{m}: {sorted(syndromes)}"
    )
    assert abs(np.linalg.norm(a) - 1.0) < 1e-12, "amplitude vector not normalized"
    return a, syndromes


def v_phase_positions(B, v):
    """The set of syndrome ints {s_p : v_p = 1} that the v-phase oracle flips."""
    n = B.shape[1]
    out = []
    for p in range(B.shape[0]):
        if int(v[p]) == 1:
            out.append(int(sum(int(B[p][i]) << i for i in range(n))))
    return sorted(out)


# ---------------------------------------------------------------------------
# Variant (a): STRUCTURED / TRANSPARENT — explicit DQI stages.
# ---------------------------------------------------------------------------
def build_structured(a, flip_syndromes, n, with_measurements=False,
                     name="dqi_binary_structured"):
    """Explicit DQI stages on n qubits so a reviewer sees it is not hardcoded.

    Stage 1: weighted symmetric prep that sets the MAGNITUDES |a| (1/sqrt2 on
             s=0, the rest spread over the nonzero syndromes). Realized via
             StatePreparation of |a| (the unsigned magnitude vector) — the
             magnitudes come straight from the DQI weights, not the answer.
    Stage 2: v-phase oracle = a diagonal +-1 that flips the sign of |s> for
             s in {s_p : v_p = 1}. Implemented as an exact Boolean phase via a
             Walsh-Hadamard expansion over Rz / higher-order Z products.
    Stage 3: decode = IDENTITY. Preparing directly in the syndrome basis already
             folds in the t=1 location->syndrome relabel (the bijection), so the
             reversible Hamming decode of the faithful circuit is a no-op here.
    Stage 4: H on all n qubits.
    """
    qc = QuantumCircuit(n, name=name)

    # --- Stage 1: magnitudes only (sign-free symmetric prep). ---
    mags = np.abs(a)
    qc.append(StatePreparation(mags), list(range(n)))
    qc.barrier()

    # --- Stage 2: v-phase oracle as an explicit Boolean +-1 phase. ---
    _apply_boolean_phase(qc, flip_syndromes, n)
    qc.barrier()

    # --- Stage 3: decode = identity (bijection already applied by basis prep). ---
    # (no gates)

    # --- Stage 4: Hadamard transform. ---
    qc.h(list(range(n)))

    if with_measurements:
        qc.measure_all()
    return qc


def _apply_boolean_phase(qc, flip_syndromes, n):
    """Apply diag(+-1) flipping |s> for s in flip_syndromes, via a Walsh expansion.

    The Boolean phase f(s) in {0,1} (1 = flip) on n qubits expands in the Walsh
    basis: exp(i*pi*f(s)) = prod_T exp(i*theta_T*chi_T(s)), chi_T(s)=(-1)^{|T&s|},
    theta_T = (pi/2^n) * sum_s f(s)*chi_T(s). Each term is exp(i*theta*Z_{T}), a
    product of Pauli-Z over the qubits in subset T, realized exactly by a CX
    ladder + a single Rz. This is exact and Clifford+Rz only (no MCX).
    """
    f = np.zeros(1 << n)
    for s in flip_syndromes:
        f[s] = 1.0
    for mask in range(1 << n):
        # theta_T via Walsh-Hadamard transform of f.
        theta = 0.0
        for s in range(1 << n):
            chi = (-1.0) ** (bin(mask & s).count("1"))
            theta += f[s] * chi
        theta *= np.pi / (1 << n)
        if abs(theta) < 1e-12:
            continue
        bits = [q for q in range(n) if (mask >> q) & 1]
        if len(bits) == 0:
            qc.global_phase += theta
        elif len(bits) == 1:
            # Rz(-2theta) = diag(e^{i theta}, e^{-i theta})
            qc.rz(-2 * theta, bits[0])
        else:
            _multi_z_phase(qc, bits, theta)


def _multi_z_phase(qc, bits, theta):
    """exp(i*theta * prod_{q in bits} Z_q) via a CX ladder + a single Rz."""
    target = bits[-1]
    for b in bits[:-1]:
        qc.cx(b, target)
    qc.rz(-2 * theta, target)
    for b in reversed(bits[:-1]):
        qc.cx(b, target)


# ---------------------------------------------------------------------------
# Variant (b): MINIMAL — direct StatePreparation(a) then H^{(x)n}.
# ---------------------------------------------------------------------------
def build_minimal(a, n, with_measurements=False, name="dqi_binary_minimal"):
    """Smallest gate count for hardware: prepare the SIGNED vector a, then H^n."""
    qc = QuantumCircuit(n, name=name)
    qc.append(StatePreparation(a), list(range(n)))
    qc.h(list(range(n)))
    if with_measurements:
        qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Verification helpers.
# ---------------------------------------------------------------------------
def statevector_marginal(qc):
    """Return P(x) over the n qubits (no measurements on qc).

    probabilities() is indexed with qubit0 = LSB, the SAME convention as the
    syndrome-int index of `a` and the shots readout, so array index x = the
    solution integer x = sum(bit_i << i).
    """
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.probabilities())


def shots_score(qc_meas, B, v):
    """AerSimulator shots: reuse the satisfied-count readout (x -> (B@x)%2==v).

    Returns ``(mean, normalized_hist, sol_hist, total)`` — the solution register
    is measured alone (no error register), so the readout offset is 0.
    """
    sim = AerSimulator()
    qct = transpile(qc_meas, sim, basis_gates=SIM_BASIS)
    counts = sim.run(qct, shots=16384).result().get_counts()
    r = score_counts(counts, B, v, x_offset=0)
    tot = r["total"]
    return r["mean"], {k: r["hist"][k] / tot for k in r["hist"]}, r["sol_hist"], tot


# ---------------------------------------------------------------------------
# Routing under restricted connectivity.
# ---------------------------------------------------------------------------
def routed_cz(qc, edges, seeds=(0,)):
    """Best routed (CZ, depth) over a transpiler-seed sweep.

    ``seeds`` defaults to a single deterministic seed (the 3-qubit case routes
    trivially); pass a wider ``range(...)`` for the larger maps where the SABRE
    router's swap insertion is seed-sensitive and we keep the fewest routed CZ.
    """
    cmap = CouplingMap(couplinglist=edges)
    best_cz, best_d = None, None
    for seed in seeds:
        t = transpile(
            qc,
            basis_gates=IQM_BASIS,
            coupling_map=cmap,
            optimization_level=3,
            seed_transpiler=seed,
        )
        s = gate_stats(t)
        if best_cz is None or s["2q_gates"] < best_cz:
            best_cz, best_d = s["2q_gates"], s["depth"]
    return best_cz, best_d
