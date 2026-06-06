"""Gate-count / depth metrics for DQI circuits (pure Qiskit, no compiler deps)."""

__all__ = ["gate_stats", "compare"]

# Two-qubit gate names worth tracking — these dominate hardware error budgets.
_TWO_QUBIT_GATES = {
    "cx", "cz", "cy", "ecr", "cp", "crx", "cry", "crz",
    "swap", "iswap", "rzz", "rxx", "ryy", "rzx",
}


def gate_stats(qc):
    """Return a dict of {qubits, gates, depth, 2q_gates, ops} for a circuit."""
    ops = dict(qc.count_ops())
    two_q = sum(c for g, c in ops.items() if g in _TWO_QUBIT_GATES)
    return {
        "qubits": qc.num_qubits,
        "gates": sum(ops.values()),
        "depth": qc.depth(),
        "2q_gates": two_q,
        "ops": ops,
    }


def compare(before, after, label_before="before", label_after="after"):
    """Print a before/after gate-reduction table and return the two stat dicts."""
    b, a = gate_stats(before), gate_stats(after)

    def pct(x0, x1):
        return f"{100 * (x0 - x1) / x0:+.1f}%" if x0 else "n/a"

    rows = [
        ("qubits", b["qubits"], a["qubits"], pct(b["qubits"], a["qubits"])),
        ("total gates", b["gates"], a["gates"], pct(b["gates"], a["gates"])),
        ("2q gates", b["2q_gates"], a["2q_gates"], pct(b["2q_gates"], a["2q_gates"])),
        ("depth", b["depth"], a["depth"], pct(b["depth"], a["depth"])),
    ]
    w = max(len(label_before), len(label_after), 8)
    print(f"{'metric':<12} {label_before:>{w}} {label_after:>{w}} {'reduction':>10}")
    print("-" * (12 + 2 * w + 12))
    for name, x0, x1, r in rows:
        print(f"{name:<12} {x0:>{w}} {x1:>{w}} {r:>10}")
    return b, a
