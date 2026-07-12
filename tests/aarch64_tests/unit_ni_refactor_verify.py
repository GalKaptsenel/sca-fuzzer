"""Systematic verification of the unified NI/sealed executor refactor against the REAL contract
executor (resolve is a software CE trace; no HW measurement needed):
  * _resolve memoizes the pure resolution (one CE run per distinct input; reset per test case).
  * the cached resolution equals a fresh one.
  * variants_for_input is deterministic (priming re-derives the exact slots it ran).
  * trace_test_case dispatches by element type (Input -> genuine baseline; ExecutorInput -> verbatim).
  * reconstruct_enacted_code rebuilds a variant's exact machine code from its recorded relocations.
Needs /dev/executor + the CE.
"""
import os
import sys
import random
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_relocations import apply_relocations
from src import factory


class NiRefactorVerifyTest(unittest.TestCase):
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

    def test_resolve_is_memoized(self):
        inp, other = self._input(), self._input()
        with mock.patch.object(self.ex._sealed, "resolve",
                               wraps=self.ex._sealed.resolve) as spy:
            self.ex.variants_for_input(inp)
            self.ex.variants_for_input(inp)   # cache hit
            self.ex._resolve(inp)             # cache hit
            self.assertEqual(spy.call_count, 1, "one CE resolve per distinct input")
            self.ex.variants_for_input(other)
            self.assertEqual(spy.call_count, 2, "a different input resolves once more")

    def test_load_test_case_clears_cache(self):
        self.ex.variants_for_input(self._input())
        self.assertTrue(self.ex._resolve_cache)
        self.ex.load_test_case(self.tc)
        self.assertEqual(self.ex._resolve_cache, {}, "cache must reset per test case")

    def test_cached_resolve_matches_uncached(self):
        inp = self._input()
        cached = self.ex._resolve(inp)
        fresh = self.ex._sealed.resolve(inp)          # bypass the memo
        self.assertEqual(cached.collapse_key, fresh.collapse_key)
        self.assertEqual(cached.genuine(), fresh.genuine())

    def test_variants_for_input_is_deterministic(self):
        inp = self._input()
        a = self.ex.variants_for_input(inp)
        self.ex._resolve_cache = {}                    # force a genuinely fresh resolve
        b = self.ex.variants_for_input(inp)
        self.assertEqual(list(a), list(b), "same variant names, same order")
        for name in a:
            self.assertEqual(a[name].code_reloc, b[name].code_reloc,
                             f"variant {name} not reproducible — priming would diverge")

    def test_trace_test_case_dispatches_by_type(self):
        inp = self._input()
        variant = next(iter(self.ex.variants_for_input(inp).values()))   # a pre-built ExecutorInput
        seen = {}

        def fake(exec_inputs, n_reps):
            seen["ei"] = list(exec_inputs)
            return [None] * len(exec_inputs)

        with mock.patch.object(self.ex, "_trace_exec_inputs", side_effect=fake):
            self.ex.trace_test_case([inp, variant], 1)
        got = seen["ei"]
        self.assertIs(got[1], variant, "a pre-built ExecutorInput passes through verbatim")
        self.assertIsInstance(got[0], self.ExecutorInput)
        self.assertEqual(got[0].code_reloc, self.ex._resolve(inp).genuine(),
                         "a plain Input is sealed to its genuine baseline")

    def test_reconstruct_enacted_code(self):
        from src.interfaces import Measurement
        inp = self._input()
        _name, ei = next(iter(self.ex.variants_for_input(inp).items()))
        m = Measurement(input_id=0, test_id=0, input_=inp, ctrace=None, htrace=None,
                        test_case=None, relocs=ei.code_reloc)
        rebuilt = self.ex.reconstruct_enacted_code(m)
        expected = apply_relocations(self.ex._sealed.object_code, list(ei.code_reloc))
        self.assertEqual(rebuilt, expected, "enacted code must reproduce from the recorded relocs")

    # -------- NI fuzzer _collect_traces: real-CE variants, mocked HW htraces --------
    def _fuzzer(self):
        from src.fuzzer import NoninterferenceFuzzer
        CONF.analyser = "sets"                       # deterministic set comparison for this test
        fz = NoninterferenceFuzzer.__new__(NoninterferenceFuzzer)
        fz.executor = self.ex
        fz.analyser = factory.get_analyser()
        fz.LOG = mock.Mock()
        return fz

    def _args(self, inputs):
        from src.fuzzer import TracingArguments
        return TracingArguments(inputs=inputs, n_reps=4, model_nesting=CONF.model_max_nesting,
                                ctraces=[], record_stats=False, fast_boosting=False,
                                update_ignore_list=False, reuse_ctraces=False, added_htraces=[])

    def test_collect_traces_flags_intra_input_divergence(self):
        from src.interfaces import HTrace
        fz = self._fuzzer()
        inputs = self.igen.generate(1)
        variants = list(self.ex.variants_for_input(inputs[0]).items())
        if len(variants) < 2:
            self.skipTest("input has no decoy-eligible slot (baseline only)")

        # baseline (flat index 0) diverges from the rest -> exactly one NI violation on input 0
        def fake(flat, n_reps):
            return [HTrace([1] * 4) if i == 0 else HTrace([0] * 4) for i in range(len(flat))], []

        with mock.patch.object(self.ex, "trace_test_case", side_effect=fake):
            violations, _ctraces, _htraces = fz._collect_traces(self._args(inputs))

        self.assertEqual(len(violations), 1)
        ms = violations[0].measurements
        self.assertEqual({m.input_id for m in ms}, {0})
        plans = {name: ei.code_reloc for name, ei in variants}
        for m in ms:                                   # test_id indexes the variant slot order
            self.assertEqual(m.relocs, plans[variants[m.test_id][0]],
                             "measurement relocs must be the enacted variant's plan")

    def test_collect_traces_no_violation_when_variants_agree(self):
        from src.interfaces import HTrace
        fz = self._fuzzer()
        inputs = self.igen.generate(1)
        with mock.patch.object(self.ex, "trace_test_case",
                               side_effect=lambda flat, n: ([HTrace([0] * 4)] * len(flat), [])):
            violations, _c, _h = fz._collect_traces(self._args(inputs))
        self.assertEqual(violations, [], "identical variant htraces must not violate")

    # -------- NI priming (_prime_one): the test-vs-context transpose --------
    def _two_variant_violation(self, H0, H1):
        from src.interfaces import Measurement, Violation, CTrace
        inp = self._input()
        variants = list(self.ex.variants_for_input(inp).items())
        if len(variants) < 2:
            self.skipTest("input has no decoy-eligible slot (baseline only)")
        m0 = Measurement(input_id=0, test_id=0, input_=inp, ctrace=CTrace.get_null(), htrace=H0,
                         test_case=None, relocs=variants[0][1].code_reloc)
        m1 = Measurement(input_id=0, test_id=1, input_=inp, ctrace=CTrace.get_null(), htrace=H1,
                         test_case=None, relocs=variants[1][1].code_reloc)
        viol = Violation.from_measurements(CTrace.get_null(), [m0, m1], [[m0], [m1]], [inp])
        return inp, variants, viol

    def test_priming_genuine_survives(self):
        from src.interfaces import HTrace
        fz = self._fuzzer()
        H0, H1 = HTrace([0] * 4), HTrace([1] * 4)
        inp, variants, viol = self._two_variant_violation(H0, H1)
        base_plan = variants[0][1].code_reloc

        # htrace follows the TEST (the occupant): swapping a variant into a slot changes that slot's
        # htrace -> divergence is real -> priming must keep the violation.
        def follows_test(primer, n_reps):
            return [H0 if ei.code_reloc == base_plan else H1 for ei in primer], []

        with mock.patch.object(self.ex, "trace_test_case", side_effect=follows_test):
            self.assertTrue(fz._prime_one(viol, [inp]),
                            "test-following divergence must survive priming")

    def test_priming_context_is_false_positive(self):
        from src.interfaces import HTrace
        fz = self._fuzzer()
        H0, H1 = HTrace([0] * 4), HTrace([1] * 4)
        inp, variants, viol = self._two_variant_violation(H0, H1)

        # htrace follows the SLOT regardless of occupant -> divergence is context, not the seal -> FP.
        def follows_slot(primer, n_reps):
            return [H0 if i == 0 else H1 for i in range(len(primer))], []

        with mock.patch.object(self.ex, "trace_test_case", side_effect=follows_slot):
            self.assertFalse(fz._prime_one(viol, [inp]),
                             "context-following divergence must be filtered as a false positive")


if __name__ == "__main__":
    unittest.main()
