"""Contract-executor memory-access modelling:

  * pair LDP/STP record BOTH elements (EA and EA+element_size),
  * a load/store whose destination/data register aliases the base (`ldr x0,[x0]`, `str x1,[x1,x2]`)
    is executed correctly and DETERMINISTICALLY across processes — regression for the
    apply_fixups base-register-translation bug, which only manifested across ASLR layouts.

Gated on /dev/executor (the CE needs it). Also runnable as `python3 unit_ce_mem_model.py --trace`,
which prints one fresh-process cache-set sequence (used by the cross-process determinism check).

The input is generated once and saved to a fixed file so every process traces the SAME input
(AArch64InputGenerator is not deterministic across processes); only then does a differing trace
indicate non-determinism in the CE itself."""
import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_SELF = os.path.abspath(__file__)
_INPUT_BIN = "/tmp/_ce_mm_input.bin"

# dest==base load (ldr x0,[x0]) feeds a dependent load so a wrong (garbage) result would change the
# cache-set footprint; LDP/STP exercise the two-element pair path.
_TC_ASM = """.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
LDP x0, x1, [x2]
STP x3, x4, [x5]
LDR x0, [x0]
LDR x3, [x0]
STR x1, [x1, x2]
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
"""


def _ce_runnable():
    return os.path.exists("/dev/executor")


def _load_tc_and_inputs(fz):
    with open("/tmp/_ce_mm.asm", "w") as f:
        f.write(_TC_ASM)
    tc = fz.asm_parser.parse_file("/tmp/_ce_mm.asm")
    fz.executor.load_test_case(tc)
    return fz.input_gen.load([_INPUT_BIN])


def _fuzzer():
    os.chdir(_ROOT)
    from src.config import CONF
    CONF.load("config.yml")
    CONF.contract_execution_clause = ["seq"]
    from src.factory import get_fuzzer
    fz = get_fuzzer("base.json", f"/tmp/_ce_mm_{os.getpid()}", None, "")
    fz.initialize_modules()
    return fz


def _ensure_input(fz):
    """Generate one input and persist it (wire format) so all processes trace identical bytes."""
    if not os.path.exists(_INPUT_BIN):
        fz._save_input(fz.input_gen.generate(1)[0], _INPUT_BIN)


def _trace_metadata():
    """Trace the crafted TC + the fixed input; return the list of per-access InstrMetadata->accesses."""
    fz = _fuzzer()
    _ensure_input(fz)
    inputs = _load_tc_and_inputs(fz)
    _, _, traces = fz.executor.trace_test_case_with_taints(inputs, 5)
    return traces[0]


def _cache_set_seq():
    t = _trace_metadata()
    x29 = t[0].cpu.gpr[29]
    return [((a.effective_address - x29) // 64) % 64
            for ite in t if ite.metadata.has_memory_access
            for a in ite.metadata.accesses()]


@unittest.skipUnless(_ce_runnable(), "contract executor needs /dev/executor (kernel module not loaded)")
class CEMemModelTest(unittest.TestCase):

    def test_pair_records_two_adjacent_elements(self):
        t = _trace_metadata()
        pairs = [ite.metadata for ite in t if ite.metadata.has_memory_access and ite.metadata.is_pair]
        self.assertEqual(len(pairs), 2, "expected LDP and STP to be flagged as pairs")
        for m in pairs:
            self.assertEqual(len(m.accesses()), 2, "a pair must record two accesses")
            self.assertEqual(m.memory_access2.effective_address,
                             m.memory_access.effective_address + m.memory_access.element_size)

    def test_dest_base_deterministic_across_processes(self):
        # create the fixed input here (main process) before spawning subprocesses
        _ensure_input(_fuzzer())
        runs = []
        for _ in range(5):
            out = subprocess.run([sys.executable, _SELF, "--trace"],
                                 capture_output=True, text=True, cwd=_ROOT)
            lines = [ln for ln in out.stdout.splitlines() if ln.startswith("SEQ ")]
            self.assertTrue(lines, f"no SEQ line:\nstdout={out.stdout}\nstderr={out.stderr}")
            runs.append(tuple(json.loads(lines[0][4:])))
        self.assertEqual(len(set(runs)), 1,
                         f"CE trace not deterministic across processes (apply_fixups bug?): {set(runs)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--trace":
        print("SEQ " + json.dumps(_cache_set_seq()))
    else:
        unittest.main()
