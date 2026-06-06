"""Make the (non-pip-installable) BCG-X DQI submodule importable.

``external/bcg-dqi`` is a collection of scripts under a ``pipelines`` package
with absolute imports (``from pipelines.X import ...``). It has no setup.py, so
we put its root on ``sys.path`` as an import side effect.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BCG_DQI = _REPO_ROOT / "external" / "bcg-dqi"
# DQI-Circuit (BankNatchapol) uses absolute `from src.dqi...` imports, so its
# repo root goes on sys.path. It provides the light Gauss-Jordan (GJE) decoder.
_DQI_CIRCUIT = _REPO_ROOT / "external" / "DQI-Circuit"

if not (_BCG_DQI / "pipelines").is_dir():
    raise ImportError(
        f"BCG DQI submodule not found at {_BCG_DQI}. "
        "Run: git submodule update --init --recursive"
    )

if str(_BCG_DQI) not in sys.path:
    sys.path.insert(0, str(_BCG_DQI))

if (_DQI_CIRCUIT / "src").is_dir() and str(_DQI_CIRCUIT) not in sys.path:
    sys.path.insert(0, str(_DQI_CIRCUIT))
