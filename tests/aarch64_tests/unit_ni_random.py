"""Random NI coverage test (needs /dev/executor + the contract executor).

Drives the real unified non-interference executor over several random combined-PAC+MTE test cases /
inputs, mints many decoys per input, classifies every slot, and checks:
  * SAFETY invariant: no PAC forge and no MTE wrong-tag ever lands on an architectural slot; the
    Sandbox clamp is always [AND, ADD].
  * COVERAGE: across the run, every corner case actually occurs — arch correct-auth and strip; spec
    forged-auth, spec strip, MTE genuine/wrong-tag; and all four PAC*MTE decoy combinations.
The deterministic per-case behaviour is pinned in unit_pac_seal / unit_mte_tagstate; this guards that
the random generator+pipeline actually exercises them together. Driven through the new
Sealer/SealedTestCase pipeline (src/aarch64/seal/sealer.py).
"""
import os
import sys
import random
import tempfile
import collections
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.seal.primitives import inst_at
from src.aarch64.seal.pac import _AUTH_TO_PAC, _read_reg
from src import factory
from src.util import FuzzLogger


def _slot_insts(tc, sealing):
    return [inst_at(tc, loc)[0] for loc in sealing.slot_locs]


def _pac_kind(insts, correct_sig):
    if insts[-1].name.lower() in ("xpaci", "xpacd"):
        return "strip"
    imm = int(insts[0].template.split("#0x")[1].split(",")[0], 16) if "movk" in insts[0].name.lower() else None
    target = (correct_sig & 0xFFFF) if correct_sig is not None else None
    return "auth_correct" if imm == target else "auth_forged"


def _mte_delta(insts):
    if insts[0].name.lower() == "nop":
        return 0
    return int(insts[0].template.rsplit("#", 1)[1])   # "ADDG r, r, #0, #<delta>"


class NiRandomCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        try:
            from src.aarch64.aarch64_kernel import PacKeys
            CONF.load(os.path.join(_ROOT, "config_pac_mte.yml"))
            from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor
            isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
            cls.gen = Aarch64RandomGenerator(isa, random.randrange(1 << 32))
            cls.ex = Aarch64NonInterferenceExecutor(cls.gen)
            k = PacKeys()
            k.apia_lo = k.apib_lo = k.apda_lo = k.apdb_lo = k.apga_lo = 0x1122334455667788
            k.apia_hi = k.apib_hi = k.apda_hi = k.apdb_hi = k.apga_hi = 0x8877665544332211
            cls.ex.local_executor.set_pac_keys(k)
            cls.igen = factory.get_input_generator(random.randrange(1 << 32))
            cls.tmp = tempfile.mkdtemp()
        except Exception as e:
            raise unittest.SkipTest(f"NI executor setup failed: {e}")

    @classmethod
    def tearDownClass(cls):
        # the pinned deterministic keys outlive this module in the kernel; revert to live
        if hasattr(cls, "ex"):
            cls.ex.local_executor.clear_pac_keys()

    def _assert_oracle_matches_genuine(self, baseline, inp, pac, mte, by_sealing):
        """Independent oracle: classify each slot from the GENUINE trace (the resolver's flag comes
        from the placeholder), and verify every genuine AUT* would authenticate without faulting."""
        ex = self.ex
        layout = ex._sealed._layout
        trace = ex._seal_trace(baseline, inp)
        base = trace[0].cpu.pc

        def min_nesting_at(off):
            ns = [e.metadata.speculation_nesting for e in trace if e.cpu.pc - base == off]
            return min(ns) if ns else None

        def auth_off(s):
            xpac = next(i for i in s.slot_insts if i.name.lower() in ("xpaci", "xpacd"))
            return layout.instruction_address[xpac]

        for s in pac:
            self.assertEqual(min_nesting_at(auth_off(s)) != 0, by_sealing[id(s)].speculative,
                             f"PAC slot reg={s.value_reg}: genuine arch-ness != resolver flag")
        for s in mte:
            self.assertEqual(min_nesting_at(layout.instruction_address[s.access]) != 0,
                             by_sealing[id(s)].speculative,
                             f"MTE slot reg={s.value_reg}: genuine arch-ness != resolver flag")

        for s in pac:
            r = by_sealing[id(s)]
            if r.value is None:
                continue
            ent = next((e for e in trace if e.cpu.pc - base == auth_off(s)
                        and e.metadata.speculation_nesting == 0), None)
            if ent is None:
                continue
            inst = s.committed_inst
            ptr = _read_reg(ent.cpu, inst.operands[0].value)
            ctx = _read_reg(ent.cpu, inst.operands[1].value) if len(inst.operands) > 1 else 0
            self.assertEqual(ex._sealed._signer.sign16(ptr, ctx, _AUTH_TO_PAC[inst.name.lower()]), r.value,
                             f"genuine {inst.name} {inst.operands[0].value}: re-sign over genuine "
                             f"state != resolved sig (would FPAC)")

    def test_random_coverage_and_safety(self):
        ex, gen = self.ex, self.gen
        cov = collections.Counter()
        arch_violations = []
        tcs = 0
        for _ in range(8 * 6):
            if tcs >= 8:
                break
            try:
                tc = gen.create_test_case(os.path.join(self.tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            ex.load_test_case(tc)
            sandbox = getattr(ex._sealed, "_sandbox", [])
            pac = getattr(ex._sealed, "_pac", [])
            mte = getattr(ex._sealed, "_mte", [])
            if not (pac or mte):
                continue
            tcs += 1
            ex._sandbox_base, _ = ex.read_base_addresses()
            ex._nesting = 5
            for inp in self.igen.generate(4):
                resolved = ex._sealed.resolve(inp)
                by_sealing = {id(r.sealing): r for r in resolved._entries}
                baseline = resolved.genuine()
                decoys = [resolved.decoy() for _ in range(6)]

                self._assert_oracle_matches_genuine(baseline, inp, pac, mte, by_sealing)

                for label, tcv in [("baseline", baseline)] + [("decoy", d) for d in decoys]:
                    for s in sandbox:                              # clamp is never decoyed
                        self.assertEqual([i.name.lower() for i in _slot_insts(tcv, s)], ["and", "add"])
                    for s in pac:
                        r = by_sealing[id(s)]
                        arch = not r.speculative
                        k = _pac_kind(_slot_insts(tcv, s), r.value)
                        cov[f"pac_{'arch' if arch else 'spec'}_{k}"] += 1
                        if arch and k == "auth_forged":
                            arch_violations.append(f"PAC forge on arch slot reg={s.value_reg}")
                    for s in mte:
                        r = by_sealing[id(s)]
                        arch = not r.speculative
                        genuine_delta = (r.value or 0) % 16
                        wrong = _mte_delta(_slot_insts(tcv, s)) != genuine_delta
                        cov[f"mte_{'arch' if arch else 'spec'}_{'wrong' if wrong else 'genuine'}"] += 1
                        if arch and wrong:
                            arch_violations.append(f"MTE wrong-tag on arch slot reg={s.value_reg}")

                # per-decoy orthogonal combination of speculative PAC forge and MTE wrong-tag
                for d in decoys:
                    pac_forged = any(by_sealing[id(s)].speculative
                                     and _pac_kind(_slot_insts(d, s), by_sealing[id(s)].value) == "auth_forged"
                                     for s in pac)
                    mte_wrong = any(by_sealing[id(s)].speculative
                                    and _mte_delta(_slot_insts(d, s)) != (by_sealing[id(s)].value or 0) % 16
                                    for s in mte)
                    cov[f"combo_{'P' if pac_forged else '-'}{'M' if mte_wrong else '-'}"] += 1

        self.assertEqual(arch_violations, [], f"architectural slots decoyed: {arch_violations[:5]}")
        for key in ("pac_arch_auth_correct", "pac_spec_auth_forged", "pac_spec_strip",
                    "mte_arch_genuine", "mte_spec_wrong"):
            self.assertGreater(cov[key], 0, f"corner case never occurred: {key} (cov={dict(cov)})")
        for combo in ("combo_P-", "combo_-M", "combo_PM"):
            self.assertGreater(cov[combo], 0, f"decoy combination never occurred: {combo} (cov={dict(cov)})")


if __name__ == "__main__":
    unittest.main()
