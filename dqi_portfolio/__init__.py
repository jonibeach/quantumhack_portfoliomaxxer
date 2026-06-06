"""dqi_portfolio — fixed-income immunization as algebraic DQI decoding.

Maps fixed-income immunization (moment matching) onto BCH/Reed--Solomon syndrome
decoding so that Decoded Quantum Interferometry (DQI) can amplify good portfolios
through an algebraic (Berlekamp--Massey) decoder rather than generic belief
propagation.

The package exposes three in-circuit decoder regimes the paper compares on the
*same* max-XORSAT instances:

* ``build_dqi_circuit_algebraic`` — Berlekamp--Massey (the central contribution),
* ``build_dqi_circuit``           — belief propagation (BP baseline),
* ``build_dqi_circuit_gje``       — Gauss--Jordan elimination (structure-blind baseline).

The BP and GJE builders wrap two upstream projects included as git submodules
under ``external/`` (see ``NOTICE.md``); everything else is original.
"""

# The BP and GJE baselines wrap the two third-party submodules under external/.
# Guard their import so the immunization / algebraic / readout path (the only one
# the web app needs) imports cleanly without external/bcg-dqi + external/DQI-Circuit
# present — keeping the deployed container lean. When the submodules ARE present
# (research env), these names resolve exactly as before.
try:
    from .dqi import build_dqi_circuit, get_optimal_w
    from .dqi_gje import build_dqi_circuit_gje
except ImportError:  # external/ submodules absent (e.g. production container)
    build_dqi_circuit = get_optimal_w = build_dqi_circuit_gje = None
from .dqi_algebraic import (
    BCHCode,
    berlekamp_massey_decode,
    build_bch_instance,
    dqi_satisfaction_stats,
    build_dqi_circuit_algebraic,
    estimate_bm_resources,
    build_gf2m_mul_add_circuit,
    build_syndrome_circuit,
    measure_bm_subblocks,
)
from .metrics import gate_stats, compare
from .immunization import (
    BondLadder,
    build_immunization_instance,
    immunization_is_bch_instance,
    decode_portfolio,
    immunization_residual,
    rs_immunization_check,
)

__all__ = [
    "build_dqi_circuit",
    "build_dqi_circuit_gje",
    "build_dqi_circuit_algebraic",
    "build_gf2m_mul_add_circuit",
    "build_syndrome_circuit",
    "measure_bm_subblocks",
    "BCHCode",
    "berlekamp_massey_decode",
    "build_bch_instance",
    "dqi_satisfaction_stats",
    "estimate_bm_resources",
    "get_optimal_w",
    "gate_stats",
    "compare",
    "BondLadder",
    "build_immunization_instance",
    "immunization_is_bch_instance",
    "decode_portfolio",
    "immunization_residual",
    "rs_immunization_check",
]
