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

# Predictor geometry that is NOT reverse-engineered — placeholders that MUST be fully RE-ed before
# the model is trusted; they are NOT confirmed N3 facts:
#   * PHR_FOLD_GROUP_WIDTH — the index/tag fold (its algorithm AND its group width) is a plausible
#     model, not an RE result. (jit.c's 11-bit mask is a different quantity — branch-address
#     placement — not evidence for this width.)
#   * TAG_WIDTH — the tagged-table tag width has no RE basis at all; 8 is arbitrary. (A finite tag
#     does make distinct branches alias, as real TAGE does.)
# Chosen explicitly here, with no default on the model class, so these guesses are visible in one
# place and obviously pending reverse-engineering.
PHR_FOLD_GROUP_WIDTH = 11
TAG_WIDTH = 8

# Logical predictor name -> zero-arg factory. Add microarchitecture models here; the factory owns
# the concrete class and its config, so the C side never needs to know them.
_PREDICTORS = {
    "neoverse-n3": lambda: Aarch64NeoverseN3BPU(
        fold_group_width=PHR_FOLD_GROUP_WIDTH, tag_width=TAG_WIDTH),
}

DEFAULT_PREDICTOR = "neoverse-n3"


def create_predictor(name: str = DEFAULT_PREDICTOR):
    """Construct and return a fresh predictor instance for ``name``.

    Stable entry point for the CE: construction (predictor selection AND its config, e.g. the PHR
    fold/tag widths) lives here, so the C side stays model-agnostic. The CE currently calls this with
    the default; `name` is the seam for selecting among future Python models (not yet plumbed from
    the C-side config). Raises ValueError on an unknown name so a misconfiguration fails loudly.
    """
    try:
        factory = _PREDICTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown predictor {name!r}; known predictors: {sorted(_PREDICTORS)}"
        )
    return factory()
