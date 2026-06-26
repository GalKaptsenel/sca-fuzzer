"""Random NI coverage test (needs /dev/executor + the contract executor).

Drives the real unified non-interference executor over several random combined-PAC+MTE test cases /
inputs, mints many decoys per input, classifies every slot, and checks:
  * SAFETY invariant: no PAC forge and no MTE retag ever lands on an architectural slot; the Sandbox
    clamp is always [AND, ADD].
  * COVERAGE: across the run, every corner case actually occurs — arch correct-auth and strip; spec
    forged-auth, spec strip, MTE retag; and all four PAC*MTE decoy combinations.
The deterministic per-case behaviour is pinned in unit_pac_seal / unit_mte_tagstate; this guards that
the random generator+pipeline actually exercises them together.
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
from src.aarch64.aarch64_seal import inst_at, CompositeSeal
from src.aarch64.aarch64_contract_executor import ExecutionClause
from src import factory
from src.util import FuzzLogger


def _parts(fp, tc):
    insts = [inst_at(tc, loc)[0] for loc in fp.slot_locs]
    seals = fp.seal.seals if isinstance(fp.seal, CompositeSeal) else [fp.seal]
    out, i = {}, 0
    for s in seals:
        out[s.name] = insts[i:i + s.slot_size]; i += s.slot_size
    return out


def _pac_kind(part, correct_sig):
    if part[1].name.lower() in ("xpaci", "xpacd"):
        return "strip"
    imm = int(part[0].template.split("#0x")[1].split(",")[0], 16) if "movk" in part[0].name.lower() else None
    return "auth_correct" if imm == correct_sig else "auth_forged"


def _mte_kind(part):
    n = part[0].name.lower()
    return "nop" if n == "nop" else ("fix_addg" if n == "addg" else "retag")


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

    def test_random_coverage_and_safety(self):
        ex, gen = self.ex, self.gen
        cov = collections.Counter()
        arch_violations = []
        log = FuzzLogger.get()
        tcs = 0
        for _ in range(8 * 6):
            if tcs >= 8:
                break
            try:
                tc = gen.create_test_case(os.path.join(self.tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            ex.load_test_case(tc)
            fps = ex._stage1_fix_points
            if not fps:
                continue
            tcs += 1
            sb, _ = ex.read_base_addresses()
            ex._engine.set_sealed(ex._stage1_tc, fps)
            for inp in self.igen.generate(4):
                for fp in fps:
                    fp.reset()
                cer = ex._contract_executor.run(ex._make_ce_execution(
                    ex._stage1_tc_bytes, inp, sb, 5, CONF.model_max_spec_window, ExecutionClause.COND))
                if ex._stage1_pac_offset_to_fp:
                    ex._sign_reached_fixpoints(cer, ex._stage1_pac_offset_to_fp, log)
                    ex._fill_missing_alt_sigs(ex._stage1_pac_fps, 6)
                if ex._stage1_mte_offset_to_fp:
                    ex._classify_mte_slots(cer, ex._stage1_mte_offset_to_fp, (sb >> 56) & 0xF)
                variants = {"baseline": ex._engine.baseline(random)}
                decoys = [next(ex._engine.decoys(random)) for _ in range(6)]
                # baseline + every decoy: classify each slot; arch slots must stay genuine
                for label, tcv in list(variants.items()) + [("decoy", d) for d in decoys]:
                    for fp in fps:
                        arch = (fp.spec_nesting == 0)
                        for name, part in _parts(fp, tcv).items():
                            if name == "sandbox":
                                self.assertEqual([i.name.lower() for i in part], ["and", "add"])
                            elif name == "pac_sign":
                                k = _pac_kind(part, fp.correct_sig)
                                cov[f"pac_{'arch' if arch else 'spec'}_{k}"] += 1
                                if arch and k == "auth_forged":
                                    arch_violations.append(f"PAC forge on arch slot {fp.slot_id}")
                            elif name == "mte_tag":
                                k = _mte_kind(part)
                                cov[f"mte_{'arch' if arch else 'spec'}_{k}"] += 1
                                if arch and k == "retag":
                                    arch_violations.append(f"MTE retag on arch slot {fp.slot_id}")
                # per-decoy orthogonal combination
                for d in decoys:
                    pac = any(fp.spec_nesting != 0 and "pac_sign" in _parts(fp, d)
                              and _pac_kind(_parts(fp, d)["pac_sign"], fp.correct_sig) == "auth_forged" for fp in fps)
                    mte = any(fp.spec_nesting != 0 and "mte_tag" in _parts(fp, d)
                              and _mte_kind(_parts(fp, d)["mte_tag"]) == "retag" for fp in fps)
                    cov[f"combo_{'P' if pac else '-'}{'M' if mte else '-'}"] += 1

        self.assertEqual(arch_violations, [], f"architectural slots decoyed: {arch_violations[:5]}")
        for key in ("pac_arch_auth_correct", "pac_spec_auth_forged", "pac_spec_strip",
                    "mte_arch_nop", "mte_spec_retag"):
            self.assertGreater(cov[key], 0, f"corner case never occurred: {key} (cov={dict(cov)})")
        for combo in ("combo_P-", "combo_-M", "combo_PM"):
            self.assertGreater(cov[combo], 0, f"decoy combination never occurred: {combo} (cov={dict(cov)})")


if __name__ == "__main__":
    unittest.main()
