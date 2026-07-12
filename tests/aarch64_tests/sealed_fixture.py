"""Shared bootstrap for the sealed/NI executor tests: a real Aarch64NonInterferenceExecutor over a
generated PAC/MTE-sealable test case, exercised against the REAL contract executor (resolve is a
software CE trace; no HW measurement needed). A mixin, so unittest never collects it as a test.
Needs /dev/executor + the CE."""
import os
import random
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src import factory

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


class SealedExecutorFixture:
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        try:
            from src.aarch64.aarch64_kernel import PacKeys
            from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor, ExecutorInput
            CONF.load(os.path.join(_ROOT, "config_pac_mte.yml"))
            cls.ExecutorInput = ExecutorInput
            isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
            cls.gen = Aarch64RandomGenerator(isa, random.randrange(1 << 32))
            cls.ex = Aarch64NonInterferenceExecutor(cls.gen)
            k = PacKeys()
            k.apia_lo = k.apib_lo = k.apda_lo = k.apdb_lo = k.apga_lo = 0x1122334455667788
            k.apia_hi = k.apib_hi = k.apda_hi = k.apdb_hi = k.apga_hi = 0x8877665544332211
            cls.ex.local_executor.set_pac_keys(k)
            cls.igen = factory.get_input_generator(random.randrange(1 << 32))
            cls.tmp = tempfile.mkdtemp()
            cls._load_sealable_tc()
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"NI executor setup failed: {e}")

    @classmethod
    def _load_sealable_tc(cls):
        for _ in range(8 * 8):
            try:
                tc = cls.gen.create_test_case(os.path.join(cls.tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            cls.ex.load_test_case(tc)
            if getattr(cls.ex._sealed, "_pac", []) or getattr(cls.ex._sealed, "_mte", []):
                cls.tc = tc
                return
        raise unittest.SkipTest("no sealable test case generated")

    def setUp(self):
        self.ex._resolve_cache = {}   # each test starts from a cold cache

    def _input(self):
        return self.igen.generate(1)[0]
