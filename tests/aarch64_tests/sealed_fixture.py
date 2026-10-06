"""Shared bootstrap for the sealed/NI executor tests: a real Aarch64NonInterferenceExecutor over a
generated PAC/MTE-sealable test case, exercised against the REAL contract executor (resolve is a
software CE trace; no HW measurement needed). A mixin, so unittest never collects it as a test.
Needs /dev/executor + the CE. Deterministic (fixed seeds), and each test draws its inputs from a fresh
input generator, so a test's inputs never depend on which tests ran before it."""
import copy
import os
import shutil
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.seal.sealer import MtePacSealedTestCase
from src import factory

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_PROGRAM_SEED = 0x5EA1
_INPUT_SEED = 0xC0DE
_MAX_TEST_CASES = 64
_MAX_INPUTS = 16


class SealedExecutorFixture:
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor, ExecutorInput
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac_mte.yml"))
        cls.ExecutorInput = ExecutorInput
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, _PROGRAM_SEED)
        cls.ex = Aarch64NonInterferenceExecutor(cls.gen)
        cls.tmp = tempfile.mkdtemp()
        cls._load_sealable_tc()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    @classmethod
    def _load_sealable_tc(cls):
        """The first generated test case with a PAC/MTE sealing and a decoy-eligible input."""
        for _ in range(_MAX_TEST_CASES):
            tc = cls.gen.create_test_case(os.path.join(cls.tmp, "t.asm"), disable_assembler=True)
            cls.ex.load_test_case(tc)
            assert isinstance(cls.ex._sealed, MtePacSealedTestCase), type(cls.ex._sealed)
            if not (cls.ex._sealed._pac or cls.ex._sealed._mte):
                continue
            igen = factory.get_input_generator(_INPUT_SEED)
            if any(cls.ex.has_decoy(i) for i in igen.generate(_MAX_INPUTS)):
                cls.ex._resolve_cache = {}
                cls.tc = tc
                return
        raise AssertionError(f"no sealable test case with a decoy in {_MAX_TEST_CASES} (seed {_PROGRAM_SEED})")

    def setUp(self):
        self.ex._resolve_cache = {}   # each test starts from a cold cache
        self.igen = factory.get_input_generator(_INPUT_SEED)

    def _input(self):
        return self.igen.generate(1)[0]

    def _decoy_input(self):
        """The next input with a decoy-eligible slot (exists: the test case was chosen for it)."""
        for _ in range(_MAX_INPUTS):
            inp = self._input()
            if self.ex.has_decoy(inp):
                return inp
        raise AssertionError("no decoy-eligible input")
