"""The generator's unreachable-flow injection for straight-line-speculation fuzzing: with
unreachable_bb_probability, a random sub-DAG is spliced physically after a non-last BB, reachable only
by speculation past that BB's branch (no main-flow edge targets it). Off by default."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd


class SlsGeneratorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.config import CONF
        CONF.load("config.yml")
        from src.isa_loader import InstructionSet
        from src.factory import get_program_generator
        cls.CONF = CONF
        cls.isa = InstructionSet("base.json", CONF.instruction_categories)
        cls.get_gen = staticmethod(get_program_generator)

    def _configure(self, prob, flow_len, max_succ, bb):
        C = self.CONF
        C.__setattr__("unreachable_bb_probability", prob)
        C.__setattr__("max_unreachable_flow_length", flow_len)
        C.__setattr__("min_successors_per_bb", 1)
        C.__setattr__("max_successors_per_bb", max_succ)
        C.__setattr__("min_bb_per_function", bb)
        C.__setattr__("max_bb_per_function", bb)

    def _gen(self, n, prob, flow_len=3, max_succ=2, bb=3):
        self._configure(prob, flow_len, max_succ, bb)
        gen = self.get_gen(self.isa, 1)
        out = []
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.asm")
            for s in range(n):
                gen._state = s
                out.append(gen.create_test_case(path, disable_assembler=True))
        return out

    @staticmethod
    def _flow_blocks(func):
        return [b for b in func if "_u" in b.name]

    def test_on_injects_flows(self):
        for tc in self._gen(8, prob=1.0):
            self.assertTrue(any(self._flow_blocks(f) for f in tc.functions), "no flow injected")

    def test_flows_are_unreachable(self):
        # No main-flow block (nor the function exit) may branch into a flow block.
        for tc in self._gen(20, prob=1.0):
            for f in tc.functions:
                main = [b for b in f if "_u" not in b.name] + [f.exit]
                for b in main:
                    for succ in b.successors:
                        self.assertNotIn("_u", succ.name, f"{b.name} -> {succ.name}: flow is reachable")

    def test_flow_length_bounded(self):
        for tc in self._gen(20, prob=1.0, flow_len=3):
            for f in tc.functions:
                for b in self._flow_blocks(f):
                    idx = int(b.name.rsplit("_u", 1)[1])
                    self.assertLess(idx, 3, f"{b.name} exceeds max_unreachable_flow_length")

    def test_flows_can_branch_internally(self):
        # With max_successors=2 the injected region is a sub-DAG, not just a chain.
        seen = False
        for tc in self._gen(20, prob=1.0, max_succ=2):
            for f in tc.functions:
                if any(len(b.successors) == 2 for b in self._flow_blocks(f)):
                    seen = True
        self.assertTrue(seen, "flows never branched internally despite max_successors=2")


if __name__ == "__main__":
    unittest.main()
