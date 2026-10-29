"""A chained load feeding a conditional branch, with an indirect call in the taken path.

The call target is materialized into x28; branch-target sealing (canonicality / alignment) later
seals it, so this template drives speculative-exception fuzzing of the indirect call's target. The callee
is unconstrained -- a random body.
"""
from src.aarch64.template import Template, IndirectCall, Mem, specs, X0, X1, X2


class CanonIndirect(Template):
    def build(self, b):
        b.instruction(specs.LDR, X0, Mem(X1))   # x0 = [x1]
        b.instruction(specs.LDR, X2, Mem(X0))   # x2 = [x0], chained on the loaded pointer
        with b.if_():
            b.call(IndirectCall(targets=1))
