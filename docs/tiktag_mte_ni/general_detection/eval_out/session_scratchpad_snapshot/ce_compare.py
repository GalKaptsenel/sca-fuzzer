import sys
sys.path.insert(0, "/tmp/claude-1000/-home-gal-k-1-1998-revizor-sca-fuzzer/cf5e0fed-2c4e-46af-8f5e-49fcf726976a/scratchpad")
from ce_parse import parse, VD, CODE_BASE
runs = {n: parse(f"{VD}/input_{n:04d}_trace_log.txt") for n in (0, 8, 16, 24)}
for n, es in runs.items():
    print(f"input {n}: entries={len(es)} nest-0={sum(e['nest']==0 for e in es)} nest-1={sum(e['nest']==1 for e in es)} max_nest={max(e['nest'] for e in es)}")
# 1) same control-flow (pc, nesting) sequence across variants?
seq = {n: [((e['pc']-CODE_BASE), e['nest']) for e in es] for n, es in runs.items()}
for n in (8, 16, 24):
    same = seq[n] == seq[0]
    print(f"control-flow (pc,nest) sequence identical to #0 for #{n}: {same}")
# 2) which entries differ in memory-access EA beyond the tag byte?  and store EA tags / before/after
def strip_tag(ea): return ea & 0x00ffffffffffffff
for n in (8, 16, 24):
    diffs = []
    for a, b in zip(runs[0], runs[n]):
        ma, mb = a["mem"], b["mem"]
        if len(ma) != len(mb):
            diffs.append((a["idx"], "len")); continue
        for x, y in zip(ma, mb):
            if strip_tag(x["ea"]) != strip_tag(y["ea"]) or x["before"] != y["before"] or x["after"] != y["after"]:
                diffs.append((a["idx"], hex(x["ea"]), hex(y["ea"]), hex(x["before"]), hex(y["before"]), hex(x["after"]), hex(y["after"])))
    print(f"#{n} vs #0: non-tag memory differences = {len(diffs)} {diffs[:5]}")
    tags = []
    for a, b in zip(runs[0], runs[n]):
        for x, y in zip(a["mem"], b["mem"]):
            if (x["ea"] >> 56) != (y["ea"] >> 56):
                tags.append((a["idx"], a["nest"], a["asm"].split(',')[0], (x["ea"]>>56)&0xf, (y["ea"]>>56)&0xf, hex(y["after"])==hex(x["after"]), "after_equal" if x["after"]==y["after"] else "after_differs"))
    print(f"   tag-byte differences #0 vs #{n}: {tags}")
# 3) After values of the spec stores (does the CE write differently per variant?)
for n in (0, 8):
    for a in runs[n]:
        if a["nest"] > 0 and a["mem"]:
            for m in a["mem"]:
                print(f"  #{n} entry {a['idx']} nest{a['nest']} {a['asm']:<26} EA tag {(m['ea']>>56)&0xf} before={m['before']:#x} after={m['after']:#x}")
