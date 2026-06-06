"""Basis-gate sets shared by the package and the demo scripts.

Centralizing these kills the dozen verbatim copies of the same two lists that
used to live at the top of every script and inside the resource estimators.
"""

__all__ = ["IQM_BASIS", "SIM_BASIS"]

# IQM-native gate set (the hardware two-qubit gate is CZ). Used for every
# resource/transpile measurement that should reflect the device.
IQM_BASIS = ["cz", "rz", "rx", "ry"]

# AerSimulator basis (CX-based, plus h/x) used when we only need to SIMULATE a
# circuit, not cost it on the device.
SIM_BASIS = ["cx", "rz", "rx", "ry", "h", "x"]
