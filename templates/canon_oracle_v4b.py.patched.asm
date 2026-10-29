"""v4 leftover template with target sets chosen off the dispatch/store baselines (16/32/48)."""
.macro.measurement_start:
    nop
from src.aarch64.template import (template, indirectcall, mem, imm, specs, flags,
.macro.measurement_start:
    nop
x0, x1, x3, x4, x5, xzr)
.macro.measurement_start:
    nop
def _const_load(off):
.macro.measurement_start:
    nop
def body(f):
.macro.measurement_start:
    nop
f.instruction(specs.orr, x3, xzr, imm(off))
.macro.measurement_start:
    nop
f.instruction(specs.ldr, x0, mem(x3, offset=imm(0)))
.macro.measurement_start:
    nop
return body
.macro.measurement_start:
    nop
class canonoraclev4b(template):
.macro.measurement_start:
    nop
def build(self, b):
.macro.measurement_start:
    nop
f1=b.function(_const_load(1024))  # set16
.macro.measurement_start:
    nop
f2=b.function(_const_load(2048))  # set32
.macro.measurement_start:
    nop
f3=b.function(_const_load(3072))  # set48
.macro.measurement_start:
    nop
b.instruction(specs.udiv, x5, x1, x1)
.macro.measurement_start:
    nop
for _ in range(15): b.instruction(specs.udiv, x5, x5, x5)
.macro.measurement_start:
    nop
b.instruction(specs.udiv, x4, x1, x5)
.macro.measurement_start:
    nop
b.instruction(specs.str, xzr, mem(x4, offset=imm(0)))
.macro.measurement_start:
    nop
b.instruction(specs.ldr, x0, mem(x1, offset=imm(0)))
.macro.measurement_start:
    nop
b.instruction(specs.subs, xzr, x0, imm(0))
.macro.measurement_start:
    nop
with b.if_(flags.ne):
.macro.measurement_start:
    nop
b.call(indirectcall(targets=[f1,f2,f3], dispatch=true, index_reg=x0))
.macro.measurement_start:
    nop
