#!/usr/bin/env python3
"""Mechanistic instruction-flow table for an AArch64 Revizor violation.

Runs the contract executor (CE) and reads its STRUCTURED per-instruction trace (the InstrTraceEntry
stream -- NOT the flattened compute_ctrace().raw, which mixes code-fetch PCs with data addresses).
For each executed instruction it prints: the disassembly, the block, the speculation phase
(architectural = speculation_nesting 0, speculative = nesting>0), and for every memory access the
actual effective address, its L1D cache set, load/store/RMW, whether it's a pair (both elements shown),
and the memory contents BEFORE -> AFTER.

RIGOR (this is the whole point): a cache set is classified ONLY from real MemAccess data:
  * a set is ARCHITECTURAL iff some nesting==0 access touches it;
  * a set is SPECULATIVE-ONLY iff some nesting>0 access touches it AND no arch access does.
Code-fetch (PC) addresses are NEVER counted as data sets -- the HW F+R trace is L1D (data) only, so the
comparison uses _cache_sets on data accesses exactly like the model's `l1d` observation clause. The
SPEC-ONLY data sets are intersected with the report.txt htrace-diff to pin the leaking set + instruction.

Usage:
  flow_table.py <violation-dir> [--repo DIR] [--pair A B] [--base 0xHEX]
                [--contract seq|cond|bpas|cond-bpas] [--nest 5] [--input A|B|both] [--all-insns]

--contract picks the speculative model to trace (default cond). Only accesses actually produced under
that contract are shown; nothing is attributed by hand. Base is auto-validated against the report's
ct-seq hash (a wrong base silently shifts every set), aborting if none matches.
"""
import argparse, os, re, sys, collections, html as _html


def find_repo(*starts):
    for start in starts:
        if not start:
            continue
        d = os.path.abspath(start)
        while d != "/":
            if os.path.isdir(os.path.join(d, "src", "aarch64")):
                return d
            d = os.path.dirname(d)
    sys.exit("could not locate repo root (no src/aarch64); pass --repo")


def parse_pair(report):
    idx = [int(m) for m in re.findall(r'Input #(\d+)\n\* Hardware trace:', report)]
    if len(idx) < 2:
        sys.exit("could not find >=2 counterexample inputs in report.txt")
    return idx[0], idx[1]


def report_hash(report, idx):
    m = re.search(rf'Input #{idx}\b.*?Contract trace \(hash\):\s*(\d+)', report, re.S)
    return int(m.group(1)) if m else None


def htrace_freq(report, idx):
    """Per-SET '^' frequency over the FULL distribution. report.txt strings are BIT-REVERSED:
    string position j == cache set (63-j)."""
    m = re.search(rf'Input #{idx}\n\* Hardware trace:(.*?)(?:\* Contract|Input #\d|\Z)', report, re.S)
    acc, tot, W = collections.defaultdict(int), 0, 0
    for mm in re.finditer(r'([.\^]{40,})\s*\[(\d+)\]', m.group(1) if m else ""):
        t, c = mm.group(1), int(mm.group(2)); tot += c; W = max(W, len(t))
        for j, ch in enumerate(t):
            if ch == '^': acc[(W - 1) - j] += c
    return {b: acc[b] / tot for b in acc} if tot else {}


def input_paths(vd, *idxs):
    out = []
    for i in idxs:
        p = next((f"{vd}/input_{i:04d}.{e}" for e in ("reif", "bin")
                  if os.path.exists(f"{vd}/input_{i:04d}.{e}")), None)
        if not p:
            sys.exit(f"input file for #{i} not found (.reif/.bin)")
        out.append(p)
    return out


_CSS = """
*{box-sizing:border-box}
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1a1d21;--mut:#6b7280;--line:#e4e7ec;--mono:#0f172a;
--arch:#eef1f5;--specbg:#fdf6ea;--accent:#b45309;--leak:#dc2626;--leakbg:#fdecec;--good:#0f766e;--hwbar:#cbd5e1;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mut:#9aa4b2;--line:#242c37;
--mono:#d6e2f0;--arch:#161b22;--specbg:#1e1a12;--accent:#f59e0b;--leak:#f87171;--leakbg:#241416;--good:#2dd4bf;--hwbar:#334155;}}
:root[data-theme=dark]{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mut:#9aa4b2;--line:#242c37;
--mono:#d6e2f0;--arch:#161b22;--specbg:#1e1a12;--accent:#f59e0b;--leak:#f87171;--leakbg:#241416;--good:#2dd4bf;--hwbar:#334155;}
:root[data-theme=light]{--bg:#f6f7f9;--panel:#fff;--ink:#1a1d21;--mut:#6b7280;--line:#e4e7ec;
--mono:#0f172a;--arch:#eef1f5;--specbg:#fdf6ea;--accent:#b45309;--leak:#dc2626;--leakbg:#fdecec;--good:#0f766e;--hwbar:#cbd5e1;}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:0 0 18px;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 22px;font-size:12px;color:var(--mut)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:18px 0;overflow:hidden}
.card>h2{font-size:15px;margin:0;padding:14px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;background:var(--leakbg);color:var(--leak)}
.kv{padding:12px 18px;border-bottom:1px solid var(--line);font-size:13px;color:var(--mut)}
.kv b{color:var(--ink)}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:18px 18px 8px}
.scroll{overflow-x:auto;padding:0 6px 6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{position:sticky;top:0;text-align:left;font-weight:600;color:var(--mut);background:var(--panel);padding:7px 10px;
border-bottom:1px solid var(--line);white-space:nowrap;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
td{padding:5px 10px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;color:var(--mono);font-size:12px}
.val{color:var(--mut)}
tr.spec td{background:var(--specbg)}
tr.ni td{color:var(--mut)} tr.ni .ins{opacity:.7}
tr.leak td{background:var(--leakbg)!important;font-weight:600}
tr.leak .set,tr.leakrow .set{color:var(--leak);font-weight:700}
.set{text-align:center;font-variant-numeric:tabular-nums}
.rw{text-align:center;font-weight:600}
.rw.W{color:var(--accent)} .rw.R{color:var(--good)}
.pill{font-size:11px;padding:2px 8px;border-radius:20px;background:var(--arch);color:var(--mut);font-weight:600}
.pill.speconly{background:var(--specbg);color:var(--accent)}
.pill.HWonly{background:transparent;border:1px dashed var(--line);color:var(--mut)}
tr.leakrow td{background:var(--leakbg)} tr.hwonly td{color:var(--mut)}
.hw{min-width:150px}
.barwrap{display:inline-block;width:80px;height:8px;background:var(--hwbar);border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:8px}
.bar{display:block;height:100%;background:var(--accent)}
tr.leakrow .bar{background:var(--leak)}
.pct{font-variant-numeric:tabular-nums;font-size:11px;color:var(--mut)}
.foot{color:var(--mut);font-size:12px;padding:0 18px 16px}
"""


def render_html(name, A, B, base, contract, hf, hdiff, hw_min, collected, per_input_sets, todo, classify):
    e = _html.escape

    def trace_tbl(rows_norm, arch, idx):
        out = ['<div class="scroll"><table>',
               '<thead><tr><th>idx</th><th>ph</th><th>instruction</th><th>rw</th><th>sz</th>'
               '<th>VA (effective)</th><th>set</th><th>mem&nbsp;before</th><th>mem&nbsp;after</th></tr></thead><tbody>']
        for ii, ph, ins, accs in rows_norm:
            if not accs:
                out.append(f'<tr class="ni {ph}"><td>{ii}</td><td>{ph}</td>'
                           f'<td class="mono ins">{e(ins)}</td><td colspan="6"></td></tr>')
                continue
            for k, (s, ea, rw, sz, bef, aft) in enumerate(accs):
                leak = ph == "spec" and s in hdiff and s not in arch
                cls = f'{ph}{" leak" if leak else ""}'
                lab = e(ins) if k == 0 else "↳ pair[2/2]"
                out.append(
                    f'<tr class="{cls}"><td>{ii if k==0 else ""}</td><td>{ph}</td>'
                    f'<td class="mono ins">{lab}</td><td class="rw {rw}">{rw}</td><td>{sz}</td>'
                    f'<td class="mono">0x{ea:012x}</td><td class="set">{s}{" 🔥" if leak else ""}</td>'
                    f'<td class="mono val">0x{bef:016x}</td><td class="mono val">0x{aft:016x}</td></tr>')
        out.append('</tbody></table></div>')
        return "\n".join(out)

    def perset_tbl(set_hits, arch, idx):
        hw = hf[idx]
        hw_only = sorted(s for s, f in hw.items() if f >= hw_min and s not in set(set_hits))
        out = ['<div class="scroll"><table>',
               '<thead><tr><th>set</th><th>class</th><th>HW %</th><th>accessed by (idx·ph)</th><th>note</th></tr></thead><tbody>']
        for s in sorted(set(set_hits) | set(hw_only)):
            f = hw.get(s, 0.0); cls = classify(set_hits, arch, s)
            by = ", ".join(f'{i}·{p}' for i, p in set_hits[s]) if s in set_hits else "—"
            leak = cls == "spec-only" and s in hdiff
            note = {"arch": "architectural — not a leak", "arch+spec": "both phases — not a leak",
                    "spec-only": "THE LEAK" if leak else "spec-only, not distinguishing",
                    "HW-only": "prefetcher / noise floor"}[cls]
            rc = "leakrow" if leak else ("hwonly" if cls == "HW-only" else "")
            pill = cls.replace("+", "").replace("-", "")
            out.append(
                f'<tr class="{rc}"><td class="set">{s}</td><td><span class="pill {pill}">{cls}</span></td>'
                f'<td class="hw"><span class="barwrap"><span class="bar" style="width:{int(round(f*100))}%"></span></span>'
                f'<span class="pct">{100*f:.1f}%</span></td>'
                f'<td class="mono">{e(by)}</td><td>{e(note)}{" 🎯" if leak else ""}</td></tr>')
        out.append('</tbody></table></div>')
        return "\n".join(out), hw_only

    sec = []
    for idx in todo:
        rows_norm, set_hits = collected[idx]
        arch_s, spec_s, so = per_input_sets[idx]
        archset = set(arch_s)
        lk = next(((ii, ins, s, ea) for ii, ph, ins, accs in rows_norm for (s, ea, *_ ) in accs
                   if ph == "spec" and s in hdiff and s not in archset), None)
        banner = ""
        if lk:
            ii, ins, s, ea = lk
            banner = (f'<div class="kv">Leak: <b>idx {ii}</b> <span class="mono">{e(ins)}</span> executes '
                      f'<b>speculatively</b> and writes <span class="mono">0x{ea:012x}</span> → <b>set {s}</b> '
                      f'(HW {100*hf[idx].get(s,0):.1f}%). Architecturally the same store lands elsewhere, so set {s} '
                      f'is touched <b>only under speculation</b>.</div>')
        tt = trace_tbl(rows_norm, archset, idx)
        pt, hw_only = perset_tbl(set_hits, archset, idx)
        leakset = sorted(set(so) & set(hdiff))
        sec.append(f'<div class="card"><h2>Input #{idx} <span class="tag">LEAK set {leakset or "NONE"}</span></h2>'
                   f'{banner}<h3>Execution trace (one row per instruction; pair = two rows)</h3>{tt}'
                   f'<h3>Per-set — model vs observed HW htrace</h3>{pt}'
                   f'<div class="foot">Unexpected (hot in HW, not modelled — prefetcher/noise): {hw_only or "none"}</div></div>')

    head = (f'<div class="wrap"><h1>{e(name)} — leak flow ({contract})</h1>'
            f'<p class="sub">in#{A} vs in#{B} · base 0x{base:x} · HW htrace-diff sets = {hdiff}</p>'
            '<div class="legend">'
            '<span><i class="sw" style="background:var(--specbg)"></i>speculative row</span>'
            '<span><i class="sw" style="background:var(--leakbg)"></i>leaking access</span>'
            '<span>🔥 spec-only set in HW htrace-diff</span></div>')
    return f'<title>{e(name)} — leak flow</title>\n<style>{_CSS}</style>\n{head}{"".join(sec)}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vd")
    ap.add_argument("--repo"); ap.add_argument("--pair", nargs=2, type=int)
    ap.add_argument("--base"); ap.add_argument("--nest", type=int, default=5)
    ap.add_argument("--contract", default="cond")
    ap.add_argument("--input", default="both", help="A|B|both (which counterexample input to trace)")
    ap.add_argument("--hw-min", type=float, default=0.05, help="min HW htrace freq for a HW-only set to be listed")
    ap.add_argument("--html", metavar="PATH", help="also write a rendered HTML page (both tables per input) here")
    ap.add_argument("--no-terminal", action="store_true", help="suppress the terminal tables (use with --html)")
    a = ap.parse_args()
    vd = os.path.abspath(a.vd)
    repo = find_repo(a.repo, vd, __file__)
    sys.path.insert(0, repo)

    from src.config import CONF; CONF.load(f"{vd}/reproduce.yaml")
    from src.factory import get_input_generator
    from src.aarch64.aarch64_generator import Aarch64Generator
    from src.aarch64.aarch64_contract_executor import (
        ContractExecution, ContractExecutorService, ExecutionClause as E, SimArch, SUPPORTED_EXECUTION_CLAUSES)
    from src.aarch64.aarch64_trace import compute_ctrace, _sandbox_base, _cache_sets, _is_speculative
    from src.aarch64.aarch64_disasm import disassemble_instruction
    from src.aarch64.aarch64_executor import _ce_memory_regs

    def clause_name(cl):
        bits = [f.name.lower() for f in E if f and int(f) and (cl & f) == f]
        return "-".join(bits) if bits else "seq"
    CLAUSE = {clause_name(cl): cl for cl in SUPPORTED_EXECUTION_CLAUSES}
    CLAUSE["seq"] = E.SEQ
    if a.contract not in CLAUSE:
        sys.exit(f"unknown contract {a.contract!r}; known: {sorted(CLAUSE)}")
    clause = CLAUSE[a.contract]
    nest = 0 if clause == E.SEQ else a.nest

    report = open(f"{vd}/report.txt").read()
    A, B = a.pair if a.pair else parse_pair(report)
    tc = Aarch64Generator.in_memory_assemble(open(f"{vd}/sandboxed_test_case.asm").read())
    ce = ContractExecutorService(os.path.join(repo, "src/aarch64/contract_executor/contract_executor"))

    def run(inp, cl, nst, base):
        m, r = _ce_memory_regs(inp)
        ex = ContractExecution(tc, m, r, SimArch.RVZR_ARCH_AARCH64, nst,
                               CONF.model_max_spec_window, req_mem_base_virt=base, execution_clauses=cl)
        return ce.run(ex)

    gen = get_input_generator(0)
    pa, pb = input_paths(vd, A, B)
    inA, inB = gen.load([pa, pb])

    # base that reproduces the report's ct-seq hash (base-dependent!)
    cands = ([int(a.base, 16)] if a.base else [])
    try: cands.append(int(open("/sys/executor/print_sandbox_base").read(), 16))
    except Exception: pass
    cands += [0x0, 0x1000000]
    want = report_hash(report, A)
    base = next((c for c in cands if want is None or compute_ctrace(run(inA, E.SEQ, 0, c)).hash_ == want), None)
    if base is None:
        sys.exit(f"no candidate base reproduces report hash {want}; pass --base. tried={[hex(c) for c in cands]}")

    print(f"# {os.path.basename(vd)}  in#{A} vs in#{B}  base={hex(base)}  contract={a.contract} nest={nest}"
          + ("  (base validated vs report hash)" if want else "  (report hash absent)"))
    hf = {A: htrace_freq(report, A), B: htrace_freq(report, B)}
    hdiff = sorted(s for s in set(hf[A]) | set(hf[B]) if abs(hf[A].get(s, 0) - hf[B].get(s, 0)) >= 0.10)
    print(f"# HW htrace-diff sets (>=10%): {hdiff}\n")

    todo = {"A": [A], "B": [B], "both": [A, B]}[a.input]
    inputs = {A: inA, B: inB}
    per_input_sets = {}
    collected = {}                                # idx -> (rows_norm, set_hits) for the HTML pass

    def analyze(idx):
        """Run the CE and normalize the trace: rows_norm = [(idx, phase, ins, [acc...])] where
        acc = (set, ea, rw, size, before, after); plus set_hits and arch/spec set memberships."""
        cer = run(inputs[idx], clause, nest, base)
        arch, spec = set(), set()
        set_hits = collections.defaultdict(list)
        rows_norm = []
        for ite in cer:
            md = ite.metadata; b = _sandbox_base(ite)
            phase = "spec" if _is_speculative(ite) else "arch"
            ins = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc)
            accs = []
            for ma in md.accesses():
                s = list(_cache_sets(ma, b))[0]
                (spec if phase == "spec" else arch).add(s)
                if (md.instr_index, phase) not in set_hits[s]:
                    set_hits[s].append((md.instr_index, phase))
                rw = "RMW" if ma.is_atomic else ("W" if ma.is_write else "R")
                accs.append((s, ma.effective_address, rw, ma.element_size, ma.before, ma.after))
            rows_norm.append((md.instr_index, phase, ins, accs))
        return rows_norm, arch, spec, set_hits

    def classify(set_hits, arch, s):
        in_a = any(p == "arch" for _, p in set_hits.get(s, []))
        in_s = any(p == "spec" for _, p in set_hits.get(s, []))
        return "arch+spec" if in_a and in_s else "arch" if in_a else "spec-only" if in_s else "HW-only"

    for idx in todo:
        rows_norm, arch, spec, set_hits = analyze(idx)
        spec_only = sorted(spec - arch)
        per_input_sets[idx] = (sorted(arch), sorted(spec), spec_only)
        collected[idx] = (rows_norm, set_hits)
        hw = hf[idx]
        hw_only = sorted(s for s, f in hw.items() if f >= a.hw_min and s not in set(set_hits))
        if a.no_terminal:
            continue

        # ---------- TABLE 1: THE TRACE (one row per instruction; a pair = 2 rows) ----------
        print(f"\n############### TRACE  in#{idx}   contract={a.contract}  base={hex(base)} ###############")
        h = (f"{'idx':>4}  {'ph':<4} {'instruction':<32} {'rw':>3} {'sz':>2} {'VA (effective addr)':>18} "
             f"{'set':>4} {'mem_before':>18} {'mem_after':>18}")
        print(h); print("-" * len(h))
        for ii, phase, ins, accs in rows_norm:
            if not accs:
                print(f"{ii:>4}  {phase:<4} {ins:<32}")          # non-memory instruction
                continue
            for k, (s, ea, rw, sz, bef, aft) in enumerate(accs):
                label = ins if k == 0 else "   ↳ pair[2/2]"
                tag = "  <== LEAK (spec-only & in HW-diff)" if (phase == "spec" and s in hdiff and s not in arch) else ""
                print(f"{ii:>4}  {phase:<4} {label:<32} {rw:>3} {sz:>2} 0x{ea:016x} "
                      f"{s:>4} 0x{bef:016x} 0x{aft:016x}{tag}")

        # ---------- TABLE 2: PER-SET (model classification vs the OBSERVED HW htrace for THIS input) ----------
        print(f"\n=============== PER-SET  in#{idx}  (model trace vs observed HW htrace for this input) ===============")
        h2 = (f"{'set':>4}  {'model':<10} {'HW%':>6}  {'leak':<5} {'accessed by  idx[phase]':<40} note")
        print(h2); print("-" * len(h2))
        for s in sorted(set(set_hits) | set(hw_only)):
            f = hw.get(s, 0.0); cls = classify(set_hits, arch, s)
            idxs = ", ".join(f"{i}[{p}]" for i, p in set_hits[s]) if s in set_hits else "(none — not in model trace)"
            leak = "LEAK" if (cls == "spec-only" and s in hdiff) else ""
            note = {"arch": "architectural -> should appear in HW for BOTH inputs (not a leak)",
                    "arch+spec": "touched in BOTH phases -> NOT a leak",
                    "spec-only": ("speculative-only AND in HW htrace-diff -> THE LEAK" if s in hdiff
                                  else "speculative-only, not a distinguishing HW set"),
                    "HW-only": "hot in HW but the model never accessed it -> noise / prefetcher / unmodelled"}[cls]
            print(f"{s:>4}  {cls:<10} {100*f:5.1f}%  {leak:<5} {idxs:<40} {note}")
        arch_s, spec_s, so = per_input_sets[idx]
        hot = sorted(s for s, f in hw.items() if f >= a.hw_min)
        print(f"\n  observed HW htrace hot sets (>= {100*a.hw_min:.0f}%) for in#{idx}: {hot}")
        print(f"  model ARCH sets {arch_s}")
        print(f"  model {a.contract.upper()} sets {spec_s}   SPEC-ONLY(=spec-arch) {so}")
        print(f"  HW htrace-diff {hdiff}  ->  LEAK set(s) for in#{idx}: {sorted(set(so) & set(hdiff)) or 'NONE'}")
        print(f"  UNEXPECTED — HW htrace sets NOT explained by the model (hot >= {100*a.hw_min:.0f}%, "
              f"neither arch nor {a.contract} spec): {hw_only or 'none'}"
              + (f"   [{len(hw_only)} sets: prefetcher / noise floor]" if hw_only else ""))

    if a.html:
        html_doc = render_html(os.path.basename(vd), A, B, base, a.contract, hf, hdiff, a.hw_min,
                               collected, per_input_sets, todo, classify)
        with open(a.html, "w") as fp:
            fp.write(html_doc)
        print(f"\n# wrote HTML -> {a.html}")


if __name__ == "__main__":
    main()
