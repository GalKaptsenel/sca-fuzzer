from __future__ import annotations
import re

# `if <cond> then [<lvalue> =] ConstrainUnpredictable(Unpredictable_<KIND>)`; cond/assignment hold no
# ';' so a preceding `if ... then EndOfDecode;` cannot be mis-captured.
_CONSTRAIN = re.compile(
    r"\bif\s+(?P<cond>[^;]+?)\s+then\s+(?:[\w\s:]+=\s*)?"
    r"ConstrainUnpredictable\(\s*Unpredictable_\w+")
_EQ = re.compile(r"\b([a-z]\w*)\s*==\s*([a-z]\w*)\b")   # reg-number vars are lowercase (t, n, s, t2)
_FALSE_BOOL = re.compile(r"\b(\w+)\s*:\s*boolean\s*=\s*FALSE\b")


def unpredictable_constraints(asl: str, reg_vars: frozenset) -> tuple:
    """Operand reg-var pairs that must differ (CONSTRAINED UNPREDICTABLE if they alias). A guard is
    dropped when an &&-conjunct is a boolean flag this encoding fixes FALSE (e.g. wback = FALSE)."""
    false_bools = set(_FALSE_BOOL.findall(asl))
    pairs = set()
    for m in _CONSTRAIN.finditer(asl):
        cond = m.group("cond")
        if any(conj.strip() in false_bools for conj in cond.split("&&")):
            continue
        for a, b in _EQ.findall(cond):
            if a in reg_vars and b in reg_vars and a != b:
                pairs.add(tuple(sorted((a, b))))
    return tuple(sorted(pairs))
