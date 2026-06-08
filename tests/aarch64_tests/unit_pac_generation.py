"""
Generation behaviour tests — hardware-free.

Verifies:
  1. InstructionSpec.generate(generator) produces correct instructions.
  2. AuthInstructionSpec.generate enforces ptr_reg != ctx_reg for all AUT* specs.
  3. instrument_stage1 never produces committed_insts with ptr_reg == ctx_reg.
  4. generator.generate_instruction delegates to spec.generate.
"""
import copy
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import CONF
from src.interfaces import OT, Instruction
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, PACInstrumentation, AuthInstructionSpec,
)

_ISA = None
_TMPDIR = None


def setUpModule():
    global _ISA, _TMPDIR
    CONF.load("config.yml")
    _ISA = InstructionSet("base.json", CONF.instruction_categories)
    _TMPDIR = tempfile.mkdtemp()


def tearDownModule():
    import shutil
    if _TMPDIR:
        shutil.rmtree(_TMPDIR, ignore_errors=True)


def _make_gen(seed: int) -> Aarch64RandomGenerator:
    return Aarch64RandomGenerator(_ISA, seed)


def _make_pac(gen: Aarch64RandomGenerator) -> PACInstrumentation:
    return PACInstrumentation(gen, xpac_weight=1, auth_weight=1)


def _gen_stage1(seed: int):
    gen = _make_gen(seed)
    pac = _make_pac(gen)
    asm_path = os.path.join(_TMPDIR, f"gen_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        stage1_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
    except Exception:
        return None
    if not fix_points:
        return None
    return fix_points, stage1_tc, pac


class TestInstructionSpecGenerate(unittest.TestCase):
    """InstructionSpec.generate and AuthInstructionSpec constraint tests."""

    @classmethod
    def setUpClass(cls):
        cls.gen = _make_gen(random.randint(0, 2**31))
        cls.pac = _make_pac(cls.gen)

    def _two_reg_auth_specs(self):
        return [s for s in self.pac._auth_specs.values()
                if len(s.operands) == 2
                and s.operands[0].type == OT.REG
                and s.operands[1].type == OT.REG]

    def test_generate_returns_instruction_with_correct_name(self):
        """InstructionSpec.generate produces an Instruction with matching name."""
        spec = next(iter(self.pac._auth_specs.values()))
        inst = spec.generate(self.gen)
        self.assertIsInstance(inst, Instruction)
        self.assertEqual(inst.name.lower(), spec.name.lower())

    def test_generate_fills_all_declared_operands(self):
        """generate() produces exactly as many operands as the spec declares."""
        for spec in list(self.pac._auth_specs.values())[:10]:
            with self.subTest(spec=spec.name):
                inst = spec.generate(self.gen)
                self.assertEqual(len(inst.operands), len(spec.operands))

    def test_auth_specs_are_auth_instruction_spec_instances(self):
        """All _auth_specs values must be AuthInstructionSpec."""
        for spec in self.pac._auth_specs.values():
            self.assertIsInstance(spec, AuthInstructionSpec)

    def test_auth_generate_never_produces_rd_eq_rn(self):
        """AuthInstructionSpec.generate never yields ptr_reg == ctx_reg."""
        norm = self.gen.target_desc.reg_normalized
        specs = self._two_reg_auth_specs()
        self.assertGreater(len(specs), 0, "No 2-reg auth specs found")
        for spec in specs:
            for _ in range(30):
                inst = spec.generate(self.gen)
                r0 = norm.get(inst.operands[0].value, inst.operands[0].value)
                r1 = norm.get(inst.operands[1].value, inst.operands[1].value)
                with self.subTest(spec=spec.name):
                    self.assertNotEqual(r0, r1,
                        f"{spec.name}: ptr_reg={r0} == ctx_reg={r1}")

    def test_generate_instruction_delegates_to_spec_generate(self):
        """generator.generate_instruction(spec) == spec.generate(generator) in effect."""
        spec = next(iter(self.pac._auth_specs.values()))
        inst_via_spec = spec.generate(self.gen)
        inst_via_gen  = self.gen.generate_instruction(spec)
        self.assertIsInstance(inst_via_gen, Instruction)
        self.assertEqual(inst_via_spec.name.lower(), spec.name.lower())
        self.assertEqual(inst_via_gen.name.lower(),  spec.name.lower())

    def test_stage1_committed_insts_never_have_rd_eq_rn(self):
        """instrument_stage1 committed_insts never have ptr_reg == ctx_reg."""
        norm = self.gen.target_desc.reg_normalized
        found_two_reg = False
        for seed in range(200):
            result = _gen_stage1(seed)
            if result is None:
                continue
            fix_points, _, _ = result
            for fp in fix_points:
                ci = fp.committed_inst
                if len(ci.operands) < 2:
                    continue
                found_two_reg = True
                r0 = norm.get(ci.operands[0].value, ci.operands[0].value)
                r1 = norm.get(ci.operands[1].value, ci.operands[1].value)
                with self.subTest(seed=seed, mnemonic=ci.name):
                    self.assertNotEqual(r0, r1,
                        f"seed={seed} {ci.name}: ptr_reg={r0} == ctx_reg={r1}")
        self.assertTrue(found_two_reg,
                        "No 2-reg auth fix_points found across 200 seeds")


if __name__ == '__main__':
    unittest.main()
