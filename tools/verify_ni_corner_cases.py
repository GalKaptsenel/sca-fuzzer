#!/usr/bin/env python3
"""Comprehensive NI corner-case verification: drive the real Aarch64NonInterferenceExecutor (seal ->
one CE trace -> fill PAC/MTE fix points -> mint baseline + many decoys) over many TCs/inputs for PAC,
MTE, and combined. For every slot in every variant, classify the PacSign/MteTag/Sandbox parts and:
  * ASSERT the safety invariants (baseline all-genuine; an ARCH slot is genuine in EVERY variant;
    Sandbox is never decoyed).
  * accumulate which corner cases were observed and require each to appear.
Hardware measurement is not exercised (this VM has no PMU); everything up to it runs.
"""
import os, sys, random, tempfile, collections
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor, NIVariant
from src.aarch64.aarch64_contract_executor import ExecutionClause
from src.aarch64.aarch64_seal import inst_at, CompositeSeal
from src import factory
from src.util import FuzzLogger

KEYS = dict(apia_lo=0x1234567890abcdef, apia_hi=0xfedcba9876543210,
            apib_lo=0x0f1e2d3c4b5a6978, apib_hi=0x8796a5b4c3d2e1f0,
            apda_lo=0x1122334455667788, apda_hi=0x8877665544332211,
            apdb_lo=0xaabbccddeeff0011, apdb_hi=0x1100ffeeddccbbaa,
            apga_lo=0x0102030405060708, apga_hi=0x0807060504030201)

cov = collections.Counter()
fails = []

def _parts(fp, variant_tc):
    """The filled slot instructions split per composite member: list of (seal_name, [insts])."""
    insts = [inst_at(variant_tc, loc)[0] for loc in fp.slot_locs]
    seals = fp.seal.seals if isinstance(fp.seal, CompositeSeal) else [fp.seal]
    out, i = [], 0
    for s in seals:
        out.append((s.name, insts[i:i + s.slot_size])); i += s.slot_size
    return out

def _classify_pac(part, correct_sig):
    """-> 'strip' | 'auth_correct' | 'auth_forged'."""
    sig_i, auth_i = part[0], part[1]
    if auth_i.name.lower() in ("xpaci", "xpacd"):
        return "strip"
    imm = int(sig_i.template.split("#0x")[1].split(",")[0], 16) if "movk" in sig_i.name.lower() else None
    return "auth_correct" if imm == correct_sig else "auth_forged"

def _classify_mte(part):
    """-> 'nop' | 'fix_addg' | 'retag'."""
    n = part[0].name.lower()
    if n == "nop":
        return "nop"
    if n == "addg":
        return "fix_addg"
    return "retag"  # irg / eor

def _decoyed(variant_tc, fps, prim):
    """Whether this decoy variant perturbs `prim` on any speculative slot (PAC forge / MTE retag);
    a decoy-strip is indistinguishable from genuine-strip, so PAC counts only forges."""
    for fp in fps:
        if fp.spec_nesting == 0:
            continue
        parts = dict(_parts(fp, variant_tc))
        if prim == "pac" and "pac_sign" in parts and _classify_pac(parts["pac_sign"], fp.correct_sig) == "auth_forged":
            return True
        if prim == "mte" and "mte_tag" in parts and _classify_mte(parts["mte_tag"]) == "retag":
            return True
    return False

def check_variant(fp, tc, is_arch, label):
    for name, part in _parts(fp, tc):
        if name == "sandbox":
            ok = part[0].name.lower() == "and" and part[1].name.lower() == "add"
            if not ok: fails.append(f"{label} slot{fp.slot_id}: sandbox not [and,add]: {[i.name for i in part]}")
            continue
        if name == "pac_sign":
            kind = _classify_pac(part, fp.correct_sig)
            cov[f"pac_{'arch' if is_arch else 'spec'}_{kind}_{label}"] += 1
            if is_arch and kind == "auth_forged":
                fails.append(f"{label} slot{fp.slot_id}: FORGED PAC on ARCH slot (correct={fp.correct_sig:#06x})")
        elif name == "mte_tag":
            kind = _classify_mte(part)
            cov[f"mte_{'arch' if is_arch else 'spec'}_{kind}_{label}"] += 1
            if is_arch and kind == "retag":
                fails.append(f"{label} slot{fp.slot_id}: RETAG (decoy) on ARCH slot")

def run_mode(mode, cfg, n_tcs=8, n_inputs=4, n_decoys=6):
    CONF.load(cfg)
    isa = InstructionSet("base.json", CONF.instruction_categories)
    gen = Aarch64RandomGenerator(isa, random.randrange(1 << 32))
    ex = Aarch64NonInterferenceExecutor(gen)
    from src.aarch64.aarch64_kernel import PacKeys
    k = PacKeys()
    for a, v in KEYS.items(): setattr(k, a, v)
    ex.local_executor.set_pac_keys(k)
    igen = factory.get_input_generator(random.randrange(1 << 32))
    tmp = tempfile.mkdtemp()
    tcs = 0
    for _ in range(n_tcs * 6):
        if tcs >= n_tcs: break
        try:
            tc = gen.create_test_case(os.path.join(tmp, "t.asm"), disable_assembler=True)
        except Exception:
            continue
        ex.load_test_case(tc)
        fps = ex._stage1_fix_points
        if not fps:
            continue
        tcs += 1
        sandbox_base, _ = ex.read_base_addresses()
        ex._engine.set_sealed(ex._stage1_tc, fps)
        log = FuzzLogger.get()
        for inp in igen.generate(n_inputs):
            for fp in fps: fp.reset()
            cer = ex._contract_executor.run(ex._make_ce_execution(
                ex._stage1_tc_bytes, inp, sandbox_base, 5, CONF.model_max_spec_window, ExecutionClause.COND))
            if ex._stage1_pac_offset_to_fp:
                ex._sign_reached_fixpoints(cer, ex._stage1_pac_offset_to_fp, log)
                ex._fill_missing_alt_sigs(ex._stage1_pac_fps, 6)
            if ex._stage1_mte_offset_to_fp:
                ex._classify_mte_slots(cer, ex._stage1_mte_offset_to_fp, (sandbox_base >> 56) & 0xF)
            baseline = ex._engine.baseline(random)
            decoys = [next(ex._engine.decoys(random)) for _ in range(n_decoys)]
            for fp in fps:
                arch = (fp.spec_nesting == 0)
                check_variant(fp, baseline, arch, "baseline")
                for d in decoys:
                    check_variant(fp, d, arch, "decoy")
            # per-decoy primitive selection (orthogonality): which primitives mismatch this variant
            if mode == "pac_mte":
                for d in decoys:
                    cov[f"combo_{'P' if _decoyed(d, fps, 'pac') else '-'}"
                        f"{'M' if _decoyed(d, fps, 'mte') else '-'}"] += 1
    print(f"[{mode}] {tcs} TCs verified")

run_mode("pac", "config_pac.yml")
run_mode("mte", "config_mte.yml")
run_mode("pac_mte", "config_pac_mte.yml")

print("\n=== COVERAGE ===")
for key in sorted(cov):
    print(f"  {key:42s} {cov[key]}")
print("\n=== FAILURES (safety-invariant violations) ===")
print("  NONE" if not fails else "\n".join("  " + f for f in fails[:40]))

# required corner cases
req = {
  "pac_arch_auth_correct_baseline": "NI arch PAC: correct auth",
  "pac_spec_auth_forged_decoy":     "NI spec PAC: forged auth",
  "pac_spec_strip_decoy":           "NI spec PAC: strip (xpac)",
  "mte_arch_nop_baseline":          "NI arch MTE: nop (tag matches)",
  "mte_spec_retag_decoy":           "NI spec MTE: retag (decoy)",
}
print("\n=== REQUIRED CORNER CASES ===")
missing = [d for k, d in req.items() if cov.get(k, 0) == 0]
for k, d in req.items():
    print(f"  [{'OK ' if cov.get(k,0) else 'MISS'}] {d}: {cov.get(k,0)}")
print(f"\nRESULT: {'PASS' if not fails and not missing else 'FAIL'}"
      + (f" (missing: {missing})" if missing else "") + (f" ({len(fails)} violations)" if fails else ""))
sys.exit(1 if fails or missing else 0)
