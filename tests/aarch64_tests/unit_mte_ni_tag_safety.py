"""Regression tests for the MTE non-interference seal's architectural tag safety, over the real CE.

The genuine (baseline) variant must be ARCHITECTURALLY tag-clean: every committed (nesting==0) memory
access must either match its granule's allocation tag or use a TCMA-Unchecked pointer tag (0b0000 /
0b1111). A *checked* architectural mismatch is a synchronous MTE tag-check fault on hardware that aborts
the measurement -- the KASAN crash chased down in this campaign. Its cause: the seal used to retag each
access's pointer to match its cell and then retag BACK to the sandbox base's Unchecked tag; a later
(unsealed) tag store committed that Unchecked tag into a granule the baseline then accessed with a
matched, checked pointer -> arch fault. Dropping the retag-back lets the matched (checked) tag propagate,
so the baseline is tag-clean and every decoy still re-converges with it architecturally (only the
speculative path -- the TikTag channel -- differs). See sealer.MteSealing.

Self-contained: it drives the local contract_executor over a fixed sandbox base, so it needs the CE
binary but no /dev/executor and no hardware. Seeded to reproduce the exact test cases that faulted.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.isa_loader import InstructionSet
from src import factory
import src.aarch64.aarch64_executor as ni_mod
from src.aarch64.aarch64_relocations import apply_relocations

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
# The canonical kernel sandbox base the HW campaign uses (top byte 0xff -> pointer tag 0b1111); the CE
# models every address relative to it, so any fixed value with this layout exercises the same tag logic.
_SANDBOX_BASE = 0xffffff8844fa1000
_SEED = 1        # reproduces the generated test cases (incl. the ones that used to arch-fault: 17/24/27)
_N_TCS = 30
_N_INPUTS = 4


def _arch_tag_mismatches(cer):
    """(pc_offset, ptr_tag, granule_tag, is_write) for every architectural access that MTE-faults on HW:
    a checked pointer tag (not TCMA-Unchecked 0/15) that differs from the granule's allocation tag."""
    out = []
    if not cer:
        return out
    base = cer[0].cpu.pc
    for ite in cer:
        if ite.metadata.speculation_nesting != 0:
            continue
        for ma in ite.metadata.accesses():
            if ma.allocation_tag == 0xFF:                 # CE holds no tag for this address
                continue
            ptr_tag = (ma.effective_address >> 56) & 0xF
            granule_tag = ma.allocation_tag & 0xF
            if ptr_tag in (0, 15) or ptr_tag == granule_tag:
                continue
            out.append((ite.cpu.pc - base, ptr_tag, granule_tag, bool(ma.is_write)))
    return out


def _arch_stream(cer):
    """The committed (nesting==0) instruction+access stream -- identical across a genuine baseline and
    any of its decoys (decoys differ only speculatively)."""
    return [(ite.cpu.pc, ite.cpu.encoding, tuple(ite.cpu.gpr),
             tuple((ma.effective_address, ma.allocation_tag, ma.is_write)
                   for ma in ite.metadata.accesses()))
            for ite in cer if ite.metadata.speculation_nesting == 0]


def _full_stream(cer):
    return [(ite.cpu.pc, ite.metadata.speculation_nesting, ite.cpu.encoding,
             tuple((ma.effective_address, ma.allocation_tag) for ma in ite.metadata.accesses()))
            for ite in cer]


class MteNiTagSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ce_bin = os.path.join(_ROOT, "src", "aarch64", "contract_executor", "contract_executor")
        if not os.path.exists(ce_bin):
            raise unittest.SkipTest("contract_executor binary not built")
        CONF.load(os.path.join(_ROOT, "config_mte.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        gen = factory.get_program_generator(isa, _SEED)
        igen = factory.get_input_generator(_SEED)
        # The HW backend is never used here (the tests only run the local CE over a fixed sandbox base
        # and never measure), so stub it out -> construction contacts no device, local or remote.
        with mock.patch.object(ni_mod, "make_hw_executor", return_value=mock.Mock()):
            ex = factory.get_noninterference_executor(gen)
        ex._sandbox_base = _SANDBOX_BASE          # fixed base -> _ce_trace never reads a device
        gen.set_seed(_SEED); igen.set_seed(_SEED)
        tmp = tempfile.mkdtemp()

        # Trace the genuine baseline and one decoy of every (test case, input) once; the tests below
        # only inspect these cached traces.
        cls.genuine = []          # list of cer
        cls.decoy_pairs = []      # list of (baseline_cer, decoy_cer) for decoy-eligible inputs
        cls.mte_active = False
        for _ in range(_N_TCS):
            tc = gen.create_test_case(os.path.join(tmp, "t.asm"))
            ex.load_test_case(tc)
            for inp in igen.generate(_N_INPUTS):
                resolved = ex._resolve(inp)
                base_cer = ex._ce_trace(apply_relocations(resolved.object_code, list(resolved.genuine())), inp)
                cls.genuine.append(base_cer)
                cls.mte_active = cls.mte_active or any(
                    ma.allocation_tag != 0xFF for ite in base_cer for ma in ite.metadata.accesses())
                plans = ex._variants_for(resolved)
                if resolved.has_decoy() and "decoy0" in plans:
                    dec_cer = ex._ce_trace(apply_relocations(resolved.object_code, list(plans["decoy0"])), inp)
                    cls.decoy_pairs.append((base_cer, dec_cer))

    def test_genuine_baseline_is_arch_tag_clean(self):
        """No genuine baseline makes an architecturally tag-mismatched (checked) access."""
        if not self.mte_active:
            self.skipTest("MTE tag memory never modelled")
        offenders = [(i, _arch_tag_mismatches(cer))
                     for i, cer in enumerate(self.genuine) if _arch_tag_mismatches(cer)]
        self.assertEqual(offenders, [], f"genuine baseline arch tag mismatch(es): {offenders[:3]}")

    def test_decoys_reconverge_and_signal(self):
        """Every decoy re-converges with its baseline architecturally (identical committed stream), is
        itself arch tag-clean, and differs from the baseline somewhere (a real, speculative NI signal)."""
        self.assertGreater(len(self.decoy_pairs), 0, "no decoy-eligible input generated")
        for base_cer, dec_cer in self.decoy_pairs:
            self.assertEqual(_arch_stream(base_cer), _arch_stream(dec_cer),
                             "decoy diverged from the baseline architecturally")
            self.assertEqual(_arch_tag_mismatches(dec_cer), [],
                             "decoy makes an architectural tag-mismatched access")
            self.assertNotEqual(_full_stream(base_cer), _full_stream(dec_cer),
                                "decoy identical to the baseline (no NI signal)")


if __name__ == "__main__":
    unittest.main()
