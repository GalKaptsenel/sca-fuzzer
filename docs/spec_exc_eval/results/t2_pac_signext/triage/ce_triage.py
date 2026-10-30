#!/usr/bin/env python3
"""Local, SAFE (no HW execution) CE triage of the PAC+sign-ext violation.
Confirms: contract-equivalence (seq), the speculative mechanism (which sealed AUT*/sign-ext slot),
and that the decoy's speculative footprint explains the htrace-divergent cache set."""
import os, re, sys
sys.path.insert(0, "/home/gal_k_1_1998/revizor/sca-fuzzer")
import numpy as np
from src.config import CONF

VD = "/home/gal_k_1_1998/revizor/sca-fuzzer/docs/spec_exc_eval/results/t2_pac_signext/violation-261030-001505"
A, B = 26, 68
CONF.load(f"{VD}/reproduce.yaml")

from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor
from src.aarch64.aarch64_contract_executor import ExecutionClause
from src.aarch64.aarch64_trace import compute_ctrace
from src import factory

ROOT = "/home/gal_k_1_1998/revizor/sca-fuzzer"
isa = InstructionSet(f"{ROOT}/base.json", CONF.instruction_categories)
gen = Aarch64RandomGenerator(isa, 0)
ex = Aarch64SpeculativeExceptionExecutor(gen)

# Load the test case (device staging only — no execution) so the sealer builds the sealed code.
from src.aarch64.aarch64_asm_parser import Aarch64AsmParser
tc = Aarch64AsmParser(gen).parse_file(f"{VD}/generated.asm")
ex.load_test_case(tc)
base, _ = ex.read_base_addresses()
print(f"sandbox base = {base:#x}")

igen = factory.get_input_generator(0)
inA, inB = igen.load([f"{VD}/input_{A:04d}.reif", f"{VD}/input_{B:04d}.reif"])

def simd(inp):
    return inp[0]['simd'].view(np.uint8).copy()

sA, sB = simd(inA), simd(inB)
diff = np.nonzero(sA != sB)[0]
print(f"\nGST image diff bytes (A#{A} vs B#{B}): offsets {list(diff)}")
for off in diff:
    print(f"  off {off}: A={sA[off]:#04x} B={sB[off]:#04x}")

# Which sealing owns each differing lane, and is it speculative?
resolved = ex._resolve(inA)
print(f"\nResolved entries for #{A} (contract hash {compute_ctrace(resolved.cer).hash_}):")
for e in resolved._entries:
    loc = getattr(e.sealing, "location", None)
    rng = loc.byte_range() if loc is not None else None
    name = type(e.sealing).__name__
    inst = getattr(e.sealing, "access_inst", None) or getattr(e.sealing, "committed_inst", None)
    inst_s = getattr(inst, "name", None)
    touches = rng is not None and any(rng.start <= d < rng.stop for d in diff)
    flag = "  <== DIFFERS (decoy lane)" if touches else ""
    print(f"  {name:20s} spec={e.speculative!s:5s} lane={rng} inst={inst_s} "
          f"value={e.value} nalts={len(e.alts)}{flag}")

# htrace-divergent bits from report.txt (per-bit frequency across all pattern/count lines)
def htrace_freq(idx):
    txt = open(f"{VD}/report.txt").read()
    m = re.search(rf"Input #{idx}\b(.*?)(?=Input #|\Z)", txt, re.S)
    freq = {}
    tot = 0
    for pat, cnt in re.findall(r"([ \^\.]{40,})\s*\[(\d+)\]", m.group(1)):
        c = int(cnt); tot += c
        s = pat.rstrip("\n")
        for i, ch in enumerate(s):
            setno = len(s) - 1 - i
            if ch == '^':
                freq[setno] = freq.get(setno, 0) + c
    return {k: v / tot for k, v in freq.items()}, tot

fa, ta = htrace_freq(A)
fb, tb = htrace_freq(B)
allsets = sorted(set(fa) | set(fb))
print(f"\nHTRACE per-bit frequency (A reps={ta}, B reps={tb}); sets with |fA-fB|>0.1:")
divergent = []
for s in allsets:
    da, db = fa.get(s, 0.0), fb.get(s, 0.0)
    if abs(da - db) > 0.1:
        divergent.append(s)
        print(f"  set {s:2d}: fA={da:.2f} fB={db:.2f}  DIVERGENT")
print(f"divergent htrace sets = {divergent}")

# INTEGRITY CHECK 1: do the two executor-inputs differ ONLY in the GVT image? If main/faulty/gpr are
# identical, the htrace divergence CANNOT come from the arch input — only from the seal values.
def frag(inp):
    f = inp[0]
    return {k: bytes(f[k].view(np.uint8)) for k in ("main", "faulty", "gpr")}
fA, fB = frag(inA), frag(inB)
for k in ("main", "faulty", "gpr"):
    same = fA[k] == fB[k]
    nd = 0 if same else int(np.count_nonzero(np.frombuffer(fA[k], np.uint8) != np.frombuffer(fB[k], np.uint8)))
    print(f"  arch-input '{k}': {'IDENTICAL' if same else f'DIFFERS in {nd} bytes'}")

# INTEGRITY CHECK 2: the genuine contract traces (resolve render, emul-injected) must be EQUAL -- the
# contract-equivalence premise of the violation. resolve canonicalizes the AUT* so it does not abort.
hA = compute_ctrace(ex._resolve(inA).cer).hash_
hB = compute_ctrace(ex._resolve(inB).cer).hash_
print(f"\n  genuine contract hash A={hA}")
print(f"  genuine contract hash B={hB}")
print(f"  -> contract-equivalent: {hA == hB}")

# The leak set vs the speculative sealed accesses: list speculative sign-ext/PAC slots and whether
# their genuine speculative access set, or the decoy (bit-55 flip) of it, could reach the leak set.
print("\n  Speculative sealed accesses in the genuine resolve trace of #%d:" % A)
cer = ex._resolve(inA).cer
cb = cer[0].cpu.pc
for ite in cer:
    ma = ite.metadata.memory_access
    if ma is None or not ite.metadata.has_sandbox_memory_access:
        continue
    if ite.metadata.speculation_nesting == 0:
        continue
    ea = ma.effective_address
    s_gen = ((ea - base) // 64) % 64
    s_flip = (((ea ^ (1 << 55)) - base) // 64) % 64
    print(f"    off={ite.cpu.pc-cb:#06x} nest={ite.metadata.speculation_nesting} "
          f"EA={ea:#x} set_genuine={s_gen} set_bit55flip={s_flip}")
