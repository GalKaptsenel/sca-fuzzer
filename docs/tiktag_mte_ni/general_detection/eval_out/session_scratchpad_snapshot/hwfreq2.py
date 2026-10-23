import re
REPORT = "/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/collect/session_20261023_011712/violation-261023-021502/report.txt"
lines = open(REPORT).read().split("\n")
start = lines.index("## Counterexample Inputs")
blocks, cur = {}, None
for ln in lines[start:]:
    m = re.match(r"^Input #(\d+)$", ln)
    if m:
        cur = int(m.group(1)); blocks[cur] = []; continue
    if cur is not None and re.match(r"^\s*[\^\.]+\s+\[\d+\]\s*$", ln):
        blocks[cur].append(ln.strip())
    if ln.startswith("* Contract trace"):
        cur = None
ORDER = sorted(blocks)
freq, nrows, tot_all, dist = {}, {}, {}, {}
for k in ORDER:
    tot = 0; per = [0]*64
    rows = blocks[k]
    for r in rows:
        pat, cnt = re.match(r"^(\S+)\s+\[(\d+)\]$", r).groups(); cnt = int(cnt); tot += cnt
        assert len(pat) == 64
        for ci, ch in enumerate(pat):
            if ch == "^": per[63 - ci] += cnt
    freq[k] = [p / tot for p in per]; nrows[k] = len(rows); tot_all[k] = tot
    top = max(int(re.search(r"\[(\d+)\]$", r).group(1)) for r in rows)
    dist[k] = top / tot
print("inputs:", ORDER, " rows:", {k: nrows[k] for k in ORDER}, " total reps:", {k: tot_all[k] for k in ORDER})
print("top-trace share (determinism):", {k: round(dist[k], 3) for k in ORDER})
print("mean # hot sets per trace:", {k: round(sum(freq[k]), 2) for k in ORDER})
print()
print("set  " + " ".join(f"#{k:>2}" for k in ORDER) + "   spread  role")
arch = {0,1,13,20,38,46,61,63}
for s in range(64):
    v = [freq[k][s] for k in ORDER]
    if max(v) < 0.02 and min(v) < 0.02: continue
    spread = max(v) - min(v)
    role = "ARCH/always-hot" if min(v) > 0.95 else ("cold" if max(v) < 0.02 else "VARIABLE" if spread >= 0.05 else "low-var")
    print(f"{s:>3}  " + " ".join(f"{x:4.2f}" for x in v) + f"   {spread:5.2f}   {role}")
