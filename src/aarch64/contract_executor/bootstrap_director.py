"""C <-> Python bridge for the contract executor's branch predictor.

The CE binary embeds CPython (see tage_py.c). At startup the C side adds the directory
containing this file to ``sys.path`` and imports this module *by name*
(``bootstrap_director``); it then constructs a predictor and drives
``predict()`` / ``update()`` / ``reset()`` (and the speculation hooks) on it.

This module is the single, stable seam between the C embedding layer and the Python
predictor model. The C side knows only this module name and the ``create_predictor()``
factory, so the Python side is free to reorganise or add concrete predictor classes
behind it without touching C.

Import design — this file is loaded *both* ways:
  * as a top-level module ``bootstrap_director`` (the CE runtime: its own directory is
    placed on ``sys.path``), and
  * as the package submodule ``aarch64.contract_executor.bootstrap_director`` (tests/tools).
To support both it (a) defensively puts its own directory on ``sys.path`` and (b) imports
the predictor relative-first, then top-level. That also keeps it location-independent, so
the executable + these .py files can be copied into a relocatable remote-fuzzing bundle
anywhere and still load.
"""
import sys
from pathlib import Path

# Make sibling modules importable as top-level names regardless of how we were launched
# (the CE adds this dir too, but doing it here keeps the module self-sufficient).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    # Package context: aarch64.contract_executor.bootstrap_director
    from .saturating_bp import Aarch64NeoverseN3BPU
except ImportError:
    # Top-level context: this directory is on sys.path, no parent package (CE runtime).
    from saturating_bp import Aarch64NeoverseN3BPU

# Logical predictor name -> class. Add microarchitecture models here; the C side selects
# by name via create_predictor() and never needs to know the concrete class.
_PREDICTORS = {
    "neoverse-n3": Aarch64NeoverseN3BPU,
}

DEFAULT_PREDICTOR = "neoverse-n3"


def create_predictor(name: str = DEFAULT_PREDICTOR):
    """Construct and return a fresh predictor instance for ``name``.

    Stable entry point for the CE: keeping construction here lets the C side stay
    model-agnostic (it passes a logical name, not a class). Raises ValueError on an
    unknown name so a misconfiguration fails loudly rather than silently.
    """
    try:
        cls = _PREDICTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown predictor {name!r}; known predictors: {sorted(_PREDICTORS)}"
        )
    return cls()
