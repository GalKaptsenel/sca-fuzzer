"""Object-based Python templates for the AArch64 Revizor generator.

A template is a `Template` subclass. Its `build(self, b)` describes a program by calling methods on the
builder `b` with typed objects -- no assembly strings:

    from src.aarch64.template import Template, Hole, Kind, IndirectCall, Mem, Imm, specs
    from src.aarch64.template import X0, X1, X2      # register operands

    class LeakyLoad(Template):
        def build(self, b):
            gadget = b.function(lambda f: f.hole(Hole(3, kind=Kind.LOAD)))  # a declared callee
            b.instruction(specs.LDR, X0, Mem(X1))              # a fixed instruction
            b.hole(Hole(3, kind=Kind.ALU, regs=[X0, X1, X2]))  # 3 random ALU insns over x0-x2
            b.call(IndirectCall(targets=[gadget]))             # indirect call to the declared callee

Calls: `DirectCall` is a `BL`, `IndirectCall(targets=, dispatch=)` a `BLR` (a single-`ADR` target or a
jump-table dispatch), `Call()` the generator's choice. Targets are declared callees (`b.function`) or
created on demand -- created callees get a random body unless `populate` says otherwise.

The template is only a generator: it produces the program. Run-time transforms (branch-target sealing,
etc.) are applied afterward by the executor. The engine that turns a template into a runnable test case
(hole expansion, passes, printer, assembler) is private and driven by the fuzzer.
"""
from .objects import *  # noqa: F401,F403  -- the typed vocabulary + register constants
from .objects import __all__ as _vocabulary
from .builder import Template, Builder
from .specs import specs

__all__ = ["Template", "Builder", "specs", *_vocabulary]
