"""Pin the EXACT sealing responsible for a PAC+SIGNEXT NI violation, and the reason — crash-free.

The decoy perturbs several speculative slots at once (PAC sig forgeries + sign-ext flips), so the saved
violation does not say which one leaks. Tracing the decoy aborts the CE (a sign-ext flip drives a
speculative access out of the sandbox). So instead trace the GENUINE (which the CE handles) and reason
about each speculative sealing's guarded access:
  * SIGN-EXT site: the decoy EORs bit 55 into the access's pointer. If that access is reached only
    speculatively (nest>0), the flip sends the SPECULATIVE effective address to EA^(1<<55) — a
    non-canonical, out-of-sandbox address that faults speculatively but leaves a distinct cache
    footprint -> the NI leak. Report the site, its guarded access, the genuine EA and the flipped EA.
  * PAC site: a speculative forged signature makes the AUT* fail (FPAC) speculatively — a different
    mechanism; reported for contrast.

Usage: pin_ni_sealing.py <violation_dir>
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.config import CONF

_BIT55 = 1 << 55


def main(vdir):
    CONF.load(os.path.join(vdir, "reproduce.yaml"))
    CONF.enable_cross_input_priming = True
    CONF.cross_input_leaks_only = False
    from src import factory
    from src.aarch64.aarch64_relocations import apply_relocations
    from src.aarch64.seal.sealer import SignExtSealing, PacSealing
    from src.aarch64.aarch64_trace import _SANDBOX_BASE_GPR

    fz = factory.get_fuzzer(os.path.join(_ROOT, "base.json"), vdir, "", [])
    fz.initialize_modules()
    ex = fz.executor
    tc = fz.asm_parser.parse_file(os.path.join(vdir, "generated.asm"))
    ex.load_test_case(tc)
    layout = ex._sealed._layout
    reif = [f for f in os.listdir(vdir) if f.startswith("input_") and f.endswith(".reif")]
    inputs = fz.input_gen.generate(max(1, len(reif) // CONF.inputs_per_class))
    ids = [int(m) for m in re.findall(r"Input #(\d+)\n\* Hardware trace", open(f"{vdir}/report.txt").read())]
    det = ids[0] % len(inputs)
    inp = inputs[det]
    resolved = ex._resolve(inp)

    # genuine CE trace: per code offset, the accesses (nest, EA) observed.
    cer = list(ex._ce_trace(apply_relocations(resolved.object_code, list(resolved.genuine())), inp))
    pc0 = cer[0].cpu.pc
    base = cer[0].cpu.gpr[_SANDBOX_BASE_GPR]   # the kernel sandbox base the CE traced against
    acc_by_off = {}
    for ite in cer:
        off = ite.cpu.pc - pc0
        for ma in ite.metadata.accesses():
            acc_by_off.setdefault(off, []).append((ite.metadata.speculation_nesting, ma.effective_address,
                                                   ma.is_write))

    def cset(ea):
        return ((ea - base) // 64) % 64

    print(f"violation {os.path.basename(vdir)}  detecting input #{det}  sandbox_base=0x{base:x}")
    sx = [e for e in resolved._entries if isinstance(e.sealing, SignExtSealing)]
    pac = [e for e in resolved._entries if isinstance(e.sealing, PacSealing)]
    print(f"  sign-ext sites: {len(sx)} ({sum(e.speculative for e in sx)} speculative);  "
          f"PAC sites: {len(pac)} ({sum(e.speculative for e in pac)} speculative)")

    culprits = []
    set_to_site = {}   # genuine L1D set -> list of sign-ext sites whose speculative access lands there
    for e in sx:
        if not (e.speculative and e.alts):
            continue
        slot_off = layout.instruction_address[e.sealing.slot_insts[0]]
        acc_off = layout.instruction_address[e.sealing.access_inst]
        hits = acc_by_off.get(acc_off, [])
        spec_hits = [(nest, ea, w) for (nest, ea, w) in hits if nest > 0]
        for nest, ea, w in spec_hits:
            fea = ea ^ _BIT55
            oob = ((fea >> 55) & 1) != ((base >> 55) & 1)   # bit-55 flip crosses the regime -> out of sandbox
            culprits.append((slot_off, e.sealing.value_reg, acc_off, nest, ea, fea))
            set_to_site.setdefault(cset(ea), []).append((slot_off, e.sealing.value_reg, acc_off, nest, w))
            print(f"  [SIGNEXT] flip slot@0x{slot_off:x} reg={e.sealing.value_reg} -> guarded "
                  f"{'WRITE' if w else 'READ'} @+0x{acc_off:x} nest={nest}: genuine EA=0x{ea:x} "
                  f"(set {cset(ea)}) -> decoy EA=0x{fea:x} [bit55 flipped, {'OUT-OF-SANDBOX' if oob else 'in-sandbox'}]")
    for e in pac:
        if not (e.speculative and e.alts):
            continue
        slot_off = layout.instruction_address[e.sealing.slot_insts[0]]
        print(f"  [PAC]     forge slot@0x{slot_off:x} reg={e.sealing.value_reg}: speculative AUT* fails "
              f"(FPAC) on the forged signature -- distinct mechanism (not an OOB address)")
    # Map the OBSERVED HW divergence to the exact sealing: diff the dominant baseline vs decoy htrace
    # (MSb-first: string pos p -> set 63-p) and attribute each differing set to the sign-ext site whose
    # genuine speculative access lands in it.
    rep = open(f"{vdir}/report.txt").read()
    blocks = re.findall(r"Input #(\d+)\n\* Hardware trace:\n(.*?)\n\n", rep, re.S)

    def dom_sets(body):
        line = next(l for l in body.split("\n") if '^' in l)
        s = re.sub(r"\s*\[\d+\]$", "", line).strip()
        return sorted(63 - i for i, c in enumerate(s) if c == '^')
    b, d = dom_sets(blocks[0][1]), dom_sets(blocks[1][1])
    diff = sorted(set(b) ^ set(d))
    print(f"  HW dominant sets: baseline#{blocks[0][0]}={b}  decoy#{blocks[1][0]}={d}")
    print(f"  HW DIVERGENT sets: {diff}")
    for cs in diff:
        who = set_to_site.get(cs)
        if who:
            for slot_off, reg, acc_off, nest, w in who:
                print(f"    set {cs} -> EXACT SEALING: SignExt flip slot@0x{slot_off:x} reg={reg} guarding "
                      f"the speculative {'WRITE' if w else 'READ'} @+0x{acc_off:x} (nest={nest})")
        else:
            print(f"    set {cs} -> no modeled sign-ext speculative access lands here "
                  f"(prefetcher/other channel or PAC-FPAC side effect)")
    print(f"SIGN-EXT CULPRITS (speculative access the bit-55 flip sends out of sandbox): "
          f"{[(hex(s), r, hex(a)) for s, r, a, *_ in culprits]}")


if __name__ == "__main__":
    main(sys.argv[1])
