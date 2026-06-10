"""
print_code_base correctness tests (live kernel module required).

print_code_base must return the address where the test case actually executes
inside the JIT'd measurement harness — i.e. the view base plus the constant,
per-template offset of the test case — not the staging buffer.

Two checks (matching the design):
  1. Python/sysfs: the reported base is non-zero, differs between P+P and F+R
     (their harness prologues differ in size), and is not the sandbox base. It is
     valid with no test case loaded (the offset is precomputed at module init).
  2. Kernel self-check via a real trace: load_jit_template() verifies that the
     first test-case word lands at the reported offset (view[offset] == tc[0])
     and returns -EFAULT otherwise, which would make tracing raise. So a trace
     completing successfully confirms the offset is correct.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # repo root: config.yml / base.json

_SYSFS = "/sys/executor"
_MODULE_LOADED = os.path.exists(f"{_SYSFS}/print_code_base")

_SAVED_CONF = None


def setUpModule():
    # test_self_check_passes_on_real_trace does CONF.load(); snapshot the Borg
    # singleton so it does not leak config into other test modules.
    global _SAVED_CONF
    from src.config import CONF
    _SAVED_CONF = copy.deepcopy(CONF._borg_shared_state)


def tearDownModule():
    if _SAVED_CONF is not None:
        from src.config import CONF
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(_SAVED_CONF)


def _read_hex(name: str) -> int:
    with open(f"{_SYSFS}/{name}") as f:
        return int(f.read().strip(), 16)


def _set_mode(mode: str) -> None:
    with open(f"{_SYSFS}/measurement_mode", "w") as f:
        f.write(mode)


@unittest.skipUnless(_MODULE_LOADED, "executor kernel module not loaded")
class PrintCodeBaseTest(unittest.TestCase):

    def test_per_template_offset(self):
        """Valid without a TC; non-zero; per-template; distinct from sandbox base."""
        _set_mode("P+P")
        pp = _read_hex("print_code_base")
        _set_mode("F+R")
        fr = _read_hex("print_code_base")
        sandbox = _read_hex("print_sandbox_base")

        self.assertNotEqual(pp, 0)
        self.assertNotEqual(fr, 0)
        self.assertNotEqual(pp, fr, "P+P and F+R harness prologues should differ in size")
        self.assertNotEqual(pp, sandbox, "code base must not equal the sandbox (main_region) base")

    def test_self_check_passes_on_real_trace(self):
        """A real trace runs the kernel self-check; success ⇒ offset is correct."""
        from src.config import CONF
        CONF.load(os.path.join(_ROOT, "config.yml"))
        from src.isa_loader import InstructionSet
        from src import factory

        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        gen = factory.get_program_generator(isa, CONF.program_generator_seed)
        ig = factory.get_input_generator(CONF.input_gen_seed)
        ex = factory.get_executor()

        gen._state = 12345  # pin the generator (update_seed reads _state)
        tc = gen.create_test_case("/tmp/_codebase_tc.asm")
        ex.load_test_case(tc)
        inputs = ig.generate(3)

        htraces, _ = ex.trace_test_case(inputs, 2)  # raises if the self-check fails
        self.assertEqual(len(htraces), len(inputs))


if __name__ == "__main__":
    unittest.main()
