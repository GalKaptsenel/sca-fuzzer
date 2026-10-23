import sys
sys.path.insert(0, "/tmp/claude-1000/-home-gal-k-1-1998-revizor-sca-fuzzer/cf5e0fed-2c4e-46af-8f5e-49fcf726976a/scratchpad")
from ce_parse import parse, VD, CODE_BASE
ents = parse(f"{VD}/input_0000_trace_log.txt")
arch, spec = {}, {}
for e in ents:
    off = e["pc"] - CODE_BASE
    for ma in e["mem"]:
        s = (ma["ea"] >> 6) & 63          # set index (sandbox base is 4K-aligned)
        base = (e["enc"] >> 5) & 0x1f     # base register field, as accounting.py computes it from the code word
        (spec if e["nest"] else arch).setdefault(s, (off, base))
arch_s, spec_s = set(arch), set(spec) - set(arch)
print("CE arch sets:", sorted(arch_s))
print("CE spec-only sets:", sorted(spec_s), {s: f"pc+{spec[s][0]:#x} base=x{spec[s][1]}" for s in sorted(spec_s)})
def nearest(s):
    best = None
    for D in sorted(arch_s | spec_s):
        k = (s - D) % 64
        if 1 <= k <= 16 and (best is None or k < best[1]): best = (D, k)
    return best
for s in [6, 7, 8, 32, 52, 5, 9, 59, 19]:
    if s in spec_s: cls = "IS spec demand"
    elif s in arch_s: cls = "arch demand"
    else:
        nd = nearest(s)
        if nd and nd[0] in spec_s: cls = f"prefetch of SPEC demand {nd[0]} (+{nd[1]}) base=x{spec[nd[0]][1]}"
        elif nd and nd[0] in arch_s: cls = f"prefetch of arch demand {nd[0]} (+{nd[1]}) base=x{arch[nd[0]][1]}"
        else: cls = "UNACCOUNTED"
    print(f"set {s:2d} -> {cls}")
