"""Render the ORIGINAL no-shortcut full DQI t=1 7-bond circuit to Qiskit figures.

This is the 11-qubit circuit (y[7]+x[3]+anc[1]) that ran on Q50 at 154 routed CZ
and was later replaced by the 5-CZ binary-collapse bijection trick. We rebuild it
through the genuine code path (build_dqi_circuit_algebraic) and draw:

  1. block-level (custom gates as boxes)            -> t1_7bond_blocklevel.png
  2. decomposed decode stage / full gate-level      -> t1_7bond_gatelevel.png
  3. ASCII text drawing to stdout
"""
import matplotlib
matplotlib.use("Agg")

from dqi_portfolio.immunization import build_immunization_instance
from dqi_portfolio.dqi_algebraic import build_dqi_circuit_algebraic

OUT = "artifacts/iqm/circuit"

# Exact instance + args that produced the shipped Q50 circuit.
B, v, code, *_rest = build_immunization_instance(m_field=3, t=1)
qc = build_dqi_circuit_algebraic(
    code, B, v, ell=1,
    with_measurements=True,
)

print(f"qubits={qc.num_qubits}  depth={qc.depth()}")
print("logical op counts:", dict(qc.count_ops()))
print("\n=== ASCII (block level) ===")
print(qc.draw(output="text", fold=120))

# 1) block-level figure (UnaryAmplitudeEncoding / Unk shown as boxes)
fig = qc.draw(output="mpl", fold=40, scale=0.8,
              style={"name": "iqm"} if False else None)
fig.savefig(f"{OUT}/t1_7bond_blocklevel.png", dpi=200, bbox_inches="tight")
print(f"\nwrote {OUT}/t1_7bond_blocklevel.png")

# 2) fully decomposed gate-level figure (every CX/CRY/Toffoli)
qc_dec = qc.decompose(reps=4)
print(f"decomposed depth={qc_dec.depth()}  ops={dict(qc_dec.count_ops())}")
fig2 = qc_dec.draw(output="mpl", fold=70, scale=0.55)
fig2.savefig(f"{OUT}/t1_7bond_gatelevel.png", dpi=180, bbox_inches="tight")
print(f"wrote {OUT}/t1_7bond_gatelevel.png")
