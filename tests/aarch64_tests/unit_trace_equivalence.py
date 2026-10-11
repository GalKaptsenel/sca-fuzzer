"""A-priori equivalence property: the single resolution trace IS the genuine contract trace — the
genuine binary (correctly pre-signed) traces without an FPAC and produces identical ctrace and taint —
for each sealing primitive. Device-free: drives the CE binary directly. The one-run design reuses this
resolution trace instead of running a separate genuine trace.
"""
import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import CONF
from src.aarch64.aarch64_relocations import apply_relocations
from src.aarch64.aarch64_trace import compute_ctrace, compute_taint
import tests.aarch64_tests.ce_equivalence as eq

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_PROGRAM_SEED = 0x5EA1
_INPUT_SEED = 0xC0DE
_MAX_TEST_CASES = 64


@unittest.skipUnless(eq.ce_available(), "needs the contract-executor binary")
class _EquivalenceBase:
    CONFIG = None
    PRIMITIVES = None
    VA_SIZE = None   # set for canon/branch-target (device-free; otherwise taken from the device)
    EXTRA_CONF = {}
    PROFILE = None   # override the shared PAC profile (signer + CE) for this class; None keeps default
    PROGRAM_SEED = _PROGRAM_SEED
    INPUT_SEED = _INPUT_SEED

    @classmethod
    def setUpClass(cls):
        from src.isa_loader import InstructionSet
        from src.aarch64.aarch64_generator import Aarch64RandomGenerator
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, cls.CONFIG))
        if cls.VA_SIZE is not None:
            CONF.va_size = cls.VA_SIZE
        for k, v in cls.EXTRA_CONF.items():
            setattr(CONF, k, v)
        cls._saved_profile = eq.PROFILE
        if cls.PROFILE is not None:
            eq.PROFILE = cls.PROFILE
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, cls.PROGRAM_SEED)
        cls.sess = eq.CESession(cls.PRIMITIVES)
        cls.sealer = cls.sess.sealer(cls.gen)
        cls.tmp = tempfile.mkdtemp()
        cls._seal_a_test_case()

    @classmethod
    def tearDownClass(cls):
        cls.sess.stop()
        eq.PROFILE = cls._saved_profile
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    @classmethod
    def _has_seals(cls, sealed, resolved):
        from src.aarch64.seal.sealer import SandboxSealing
        return any(not isinstance(s, SandboxSealing) for s in sealed._sealings())

    @classmethod
    def _respawn(cls):
        cls.sess.stop()
        cls.sess = eq.CESession(cls.PRIMITIVES)
        cls.sealer = cls.sess.sealer(cls.gen)

    @classmethod
    def _seal_a_test_case(cls):
        from src import factory
        probe = factory.get_input_generator(cls.INPUT_SEED).generate(1)[0]
        for _ in range(_MAX_TEST_CASES):
            tc = cls.gen.create_test_case(os.path.join(cls.tmp, "t.asm"), disable_assembler=True)
            cls.sess.set_data_size(eq._dispatch_data_size(tc))
            sealed = cls.sealer.seal(tc)
            # A few randomly-generated indirect-call test cases make an ARCHITECTURAL out-of-sandbox
            # data access, which the CE rejects by design (a pre-existing sandbox-completeness limit for
            # indirect calls, orthogonal to sealing). Detect it by tracing the bare placeholder — no
            # no resolution — so the only thing that can fail is the CE being unable to trace the test
            # case at all; such a test case is skipped. Resolution and the resolution-vs-genuine
            # comparison below run unguarded, so a sealing bug surfaces instead of being masked.
            try:
                cls.sess.trace_bytes(sealed.object_code, probe)
            except RuntimeError:
                cls._respawn()
                continue
            resolved = sealed.resolve(probe)
            if cls._has_seals(sealed, resolved):
                cls.sealed = sealed
                return
        raise AssertionError(f"no CE-traceable sealed test case in {_MAX_TEST_CASES} "
                             f"(seed {cls.PROGRAM_SEED})")

    def test_resolution_trace_matches_genuine(self):
        from src import factory
        inputs = factory.get_input_generator(self.INPUT_SEED).generate(8)
        checked = 0
        for inp in inputs:
            resolved = self.sealed.resolve(inp)
            genuine_code = apply_relocations(resolved.object_code, resolved.genuine())
            cer_res = resolved.cer                              # the single resolution trace
            cer_g = self.sess.trace_bytes(genuine_code, inp)    # the pre-signed genuine binary, verify-mode

            # The genuine binary traces without an FPAC (the resolver's signatures are correct) and its
            # contract trace equals the resolution trace in the observable — identical entry count and
            # ctrace. So the one resolution trace is a faithful genuine trace: no separate genuine run.
            self.assertEqual(len(list(cer_res)), len(list(cer_g)),
                             "entry counts differ:\n" + "\n".join(eq.diff_traces(cer_res, cer_g)))
            self.assertEqual(compute_ctrace(cer_res), compute_ctrace(cer_g),
                             "ctrace differs:\n" + "\n".join(eq.diff_traces(cer_res, cer_g)))
            tr, tg = compute_taint(cer_res).tobytes(), compute_taint(cer_g).tobytes()
            if "mte" not in self.PRIMITIVES and CONF.pac_strip_prob == 0:
                # Renders are taint-identical: resolution AUT* == genuine AUT* (no strip), no retag.
                self.assertEqual(tr, tg, "taint differs")
            else:
                # The resolution taint is a safe SUPERSET of the genuine binary's — assert containment.
                #  * MTE: the genuine ADDG #delta gives the pointer the cell tag, so compute_taint's
                #    (effective_address - sandbox_base_register) offset leaves a (delta<<56) term, lands
                #    outside the region, and drops the byte the access reads; the resolution pointer keeps
                #    the base tag and marks it. The genuine-binary oracle UNDER-counts.
                #  * pac_strip_prob>0: a genuine XPAC strip reads no context, while the resolution AUT*
                #    does, so resolution additionally marks the ctx bytes at stripped slots.
                # Production uses the resolution taint, which is the complete, correct one.
                self.assertTrue(all(g == 0 or r != 0 for r, g in zip(tr, tg)),
                                "genuine taint not covered by the resolution taint")
            checked += 1
        self.assertGreater(checked, 0)


class TraceEquivalencePacTest(_EquivalenceBase, unittest.TestCase):
    CONFIG = "config_pac.yml"
    PRIMITIVES = {"pac"}


class TraceEquivalencePacStripTest(_EquivalenceBase, unittest.TestCase):
    # pac_strip_prob=1.0 forces every genuine auth to render as the arch-safe XPAC strip instead of an
    # AUT*, exercising the strip path the default (strip_prob=0) PAC test never reaches. The resolution
    # render still uses the committed AUT* (reads ctx), so its taint is a superset of the strip genuine's.
    CONFIG = "config_pac.yml"
    PRIMITIVES = {"pac"}
    EXTRA_CONF = {"pac_strip_prob": 1.0}


class TraceEquivalencePacMteDeviceProfileTest(_EquivalenceBase, unittest.TestCase):
    # The N3's real PAC profile: tbid1=1 disables TBI for INSTRUCTION keys, so the PAC field extends
    # into the top byte and overlaps the MTE tag [59:56]. The resolver must sign over the pointer the
    # genuine binary presents (its MTE tag stripped by the AUT*/revert XPAC), not the inert placeholder's
    # tagged pointer — otherwise every later AUT* FPACs. The suite's default profile (generic_qarma=0,
    # tbid1=0, constpacfield=0) does NOT exercise this entanglement, so this profile is the regression.
    CONFIG = "config_pac_mte.yml"
    PRIMITIVES = {"pac", "mte"}
    VA_SIZE = 48
    PROFILE = eq.qarma.profile(qarma_version=3, generic_qarma_version=3, auth_level=5,
                               va_size0=48, va_size1=48, tbi0=True, tbi1=True,
                               tbid0=False, tbid1=True, constpacfield=True, mtx0=False, mtx1=False)
    # Seeds whose test case authenticates a pointer that an MTE retag has tagged — the slot that FPACs
    # without the canonicalization splice. A generic seed may not exercise the entanglement.
    PROGRAM_SEED = 469530
    INPUT_SEED = 2665689131


class TraceEquivalenceMteTest(_EquivalenceBase, unittest.TestCase):
    CONFIG = "config_mte.yml"
    PRIMITIVES = {"mte"}


class TraceEquivalencePacMteTest(_EquivalenceBase, unittest.TestCase):
    CONFIG = "config_pac_mte.yml"
    PRIMITIVES = {"pac", "mte"}


# Canonicality and branch-target carry no dynamic value: their genuine baseline is the canonical /
# valid-target NOP placeholder (the flip is the decoy relocation), so the resolution trace already is
# the genuine trace. PTE seals no code (SandboxSealedTestCase); the genuine/decoy difference is the
# page-table environment, not the trace. These confirm the no-value-seal path yields genuine ==
# resolution trace.
class TraceEquivalenceCanonTest(_EquivalenceBase, unittest.TestCase):
    CONFIG = "config_canon.yml"
    PRIMITIVES = {"canon"}
    VA_SIZE = 48


class TraceEquivalenceBranchTargetTest(_EquivalenceBase, unittest.TestCase):
    CONFIG = "config_canon_indirect.yml"
    PRIMITIVES = {"branch_target"}
    VA_SIZE = 48
    EXTRA_CONF = {"indirect_call_probability": 1.0}


class TraceEquivalencePteTest(_EquivalenceBase, unittest.TestCase):
    CONFIG = "config_pte_oracle.yml"
    PRIMITIVES = set()

    @classmethod
    def _has_seals(cls, sealed, resolved):
        return bool(sealed._sandbox)


if __name__ == "__main__":
    unittest.main()
