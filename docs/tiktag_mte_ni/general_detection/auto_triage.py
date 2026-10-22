#!/usr/bin/env python3
"""Full MTE/TikTag non-interference triage for a saved violation, with an explicit verdict.

Orchestrates the proven scripts and applies a conservative classifier:
  1. accounting.py  (local CE, no device): tag-only diff; CE arch vs SPECULATIVE-ONLY sets; and for each
     HW diverging set whether it is a speculative tag-sealed access / a prefetch stream of one, or merely
     an architectural demand or its prefetch stream.
  2. triage_sweep.py (device): for each SPEC-relevant diverging set, locate the gating sealed load and
     sweep its tag 0..15, giving diverging-set residency per delta.

Verdict:
  GENUINE      : tag-only AND at least one diverging set is CE spec-only (or a prefetch stream of a
                 speculative tag-sealed access) AND its sweep separates the matching (genuine) CHECKED
                 tag above every checked mismatch by >= SEP, with match residency >= MATCH_MIN; the
                 separation must NOT be explained only by the TCMA-unchecked tags.
  QUESTIONABLE : spec-relevant set exists and shows tag dependence, but the separation is marginal or is
                 carried only by TCMA-unchecked tags / a gain-flavor increase -> not a clean shrink.
  NOISE        : no spec-relevant diverging set (every divergence is an architectural demand or its
                 prefetch stream), or the sweep fires ~uniformly with no match separation.

Usage: auto_triage.py <cfg> <violation-dir> [--reps N] [--ledger PATH]
"""
import os
import re
import subprocess
import sys

REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/gal_k_1_1998/revizor/revizor-venv/bin/python"
ACCOUNTING = os.path.join(REPO, ".claude/skills/verify-mte-tag-leak/accounting.py")
TRIAGE_SWEEP = os.path.join(HERE, "triage_sweep.py")
SEP = 0.20        # match must beat every checked mismatch by this
MATCH_MIN = 0.30  # and the match residency must be at least this


def _run(args):
    env = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.run([PY] + args, capture_output=True, text=True, env=env)
    return p.stdout + "\n" + p.stderr


def suspect_of(vd):
    """First counterexample input index (the baseline whose decoys are at +K)."""
    try:
        lines = open(f"{vd}/report.txt").read().splitlines()
    except OSError:
        return 0
    insec = False
    for ln in lines:
        if ln.strip() == "## Counterexample Inputs":
            insec = True
            continue
        if insec and ln.startswith("## Branch"):
            break
        m = re.match(r"Input #(\d+)$", ln.strip())
        if insec and m:
            return int(m.group(1))
    return 0


def parse_accounting(txt):
    tag_only = "TAG-ONLY: True" in txt
    spec_only = []
    m = re.search(r"CE spec-only sets\s*:\s*\[([^\]]*)\]", txt)
    if m and m.group(1).strip():
        spec_only = [int(x) for x in m.group(1).split(",")]
    # per-set classification lines: "  set 32 -> prefetch of SPEC demand 31 (+1)   (base x4)"
    spec_sets = []
    for mm in re.finditer(r"set\s+(\d+)\s*->\s*(.+?)\s*\(base (x\?|x\d+)\)", txt):
        s, src, base = int(mm.group(1)), mm.group(2), mm.group(3)
        if ("IS spec demand" in src) or ("prefetch of SPEC demand" in src):
            spec_sets.append((s, base))
    elig = []
    m = re.search(r"eligible tag-sealed regs:\s*\[([^\]]*)\]", txt)
    if m and m.group(1).strip():
        elig = [x.strip().strip("'\"") for x in m.group(1).split(",")]
    return tag_only, spec_only, spec_sets, elig


def parse_sweep(txt):
    """Return (genuine_delta, {delta: residency}, baseline) or None."""
    base = None
    mb = re.search(r"genuine baseline res=([0-9.]+)", txt)
    if mb:
        base = float(mb.group(1))
    gd = None
    mg = re.search(r"genuine delta=(\d+)", txt)
    if mg:
        gd = int(mg.group(1))
    sweep = {}
    ms = re.search(r"residency by delta 0\.\.15:\s*\n?\s*(.+)", txt)
    if ms:
        for pair in ms.group(1).split():
            if ":" in pair:
                d, v = pair.split(":")
                try:
                    sweep[int(d)] = float(v)
                except ValueError:
                    pass
    if not sweep:
        return None
    return gd, sweep, base


def classify_sweep(gd, sweep):
    """Return (verdict, detail). TCMA-unchecked absolute tags are 0 and 15; here the sweep is over the
    retag DELTA, so we treat deltas 0 and 15 as the candidate TCMA-unchecked ones and require the match
    to dominate the remaining CHECKED mismatches."""
    if gd is None or gd not in sweep:
        gd = max(sweep, key=sweep.get)
    match = sweep[gd]
    checked_mm = [v for d, v in sweep.items() if d != gd and d not in (0, 15)]
    hi_mm = max(checked_mm) if checked_mm else 0.0
    if match >= MATCH_MIN and (match - hi_mm) >= SEP:
        return "GENUINE", f"match delta {gd}={match:.2f} dominates checked mismatches (max {hi_mm:.2f})"
    # tag-dependent but not a clean match-shrink?
    spread = max(sweep.values()) - min(sweep.values())
    if match >= MATCH_MIN and (match - hi_mm) > 0.08:
        return "QUESTIONABLE", f"match delta {gd}={match:.2f} only marginally above mismatches ({hi_mm:.2f})"
    if spread > 0.15 and match < hi_mm:
        return "QUESTIONABLE", f"tag-dependent but match delta {gd}={match:.2f} below a checked mismatch ({hi_mm:.2f})"
    return "NOISE", f"no match separation (match delta {gd}={match:.2f}, checked-mismatch max {hi_mm:.2f}, spread {spread:.2f})"


def main():
    cfg, vd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    ledger = sys.argv[sys.argv.index("--ledger") + 1] if "--ledger" in sys.argv else None
    sus = suspect_of(vd)

    acc = _run([ACCOUNTING, cfg, vd, str(sus)])
    tag_only, spec_only, spec_sets, elig = parse_accounting(acc)

    report = [f"# auto_triage: {os.path.basename(vd)}", "",
              f"config: {cfg}", f"suspect input: #{sus}", f"reps: {reps}", "",
              "## Step 1+2 (accounting)", "```", acc.strip(), "```", ""]

    verdict, reason, sweeps = "NOISE", "", []
    if not tag_only:
        verdict, reason = "NOISE", "code diff is not tag-only (not an MTE tag violation)"
    elif not spec_sets:
        verdict = "NOISE"
        reason = ("no speculative tag-sealed diverging set — every HW diverging set is an architectural "
                  f"demand or its prefetch stream (CE spec-only sets={spec_only}); "
                  "prefetcher/batch false positive")
    else:
        report.append("## Step 4 (device tag-sweep on spec-relevant diverging sets)")
        best = "NOISE"
        for s, base in spec_sets:
            sw = _run([TRIAGE_SWEEP, cfg, vd, str(sus), str(s), "--reps", str(reps)])
            report += [f"### diverging set {s} (base {base})", "```", sw.strip(), "```"]
            parsed = parse_sweep(sw)
            if not parsed:
                continue
            gd, sweep, bl = parsed
            v, d = classify_sweep(gd, sweep)
            sweeps.append((s, v, d, bl))
            order = {"GENUINE": 3, "QUESTIONABLE": 2, "NOISE": 1}
            if order[v] > order[best]:
                best, verdict, reason = v, v, f"set {s} (base {base}): {d}"
        if not sweeps:
            verdict, reason = "NOISE", "spec-relevant sets found but sweep produced no signal"

    report += ["", f"## VERDICT: {verdict}", reason,
               f"\n(spec-only CE sets={spec_only}; spec-relevant diverging sets={[s for s,_ in spec_sets]}; "
               f"eligible MTE regs={elig})"]
    open(f"{vd}/AUTO_TRIAGE.md", "w").write("\n".join(report) + "\n")
    print(f"{os.path.basename(vd)}  VERDICT={verdict}  :: {reason}")
    if ledger:
        with open(ledger, "a") as fh:
            fh.write(f"{os.path.basename(vd)}\t{verdict}\t{reason}\n")
    # non-zero exit for GENUINE so a driver can react
    sys.exit(0 if verdict != "GENUINE" else 42)


if __name__ == "__main__":
    main()
