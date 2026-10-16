#!/usr/bin/env python3
"""A6 localization cost (Table 7.3): measured mean HW-execution (Measure-call) count of the REAL
binary-search localizer (exponential_search, galloping+bisection) vs the analytic ceil(log2(k+1)),
for batch sizes k = 12, 64, 256. Device-independent: the localizer sees the chain only through
r_key(t); its cost is the number of DISTINCT memoized probes (each = one HW batch measurement),
which is fully determined by k and the boundary position. We average over all boundary positions
(uniform) — the faithful algorithmic cost; multiply by reps x 2 directions for the full HW cost.
Imports the ACTUAL localizers from src/aarch64/cross_input.py (no reimplementation)."""
import math, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aarch64.cross_input import exponential_search, linear_scan  # the real localizers

def counted_rkey(bnd):
    """Memoized r_key for a single boundary at position bnd: all-base ref (0) for t>bnd, differs (1) for
    t<=bnd. Counts DISTINCT probes (each distinct t = one HW batch measurement; repeats are memoized)."""
    cache = {}; calls = [0]
    def r_key(t):
        if t not in cache:
            calls[0] += 1
            cache[t] = 0 if t > bnd else 1
        return cache[t]
    return r_key, calls

def measure_localizer(localizer, k):
    """Run `localizer` over the search space [0, k+1) (k+1 classes; detecting = class k) for every
    boundary position bnd in [0, k); return the list of distinct-probe counts."""
    counts = []
    lo, hi = 0, k + 1
    for bnd in range(0, k):
        r_key, calls = counted_rkey(bnd)
        assert r_key(lo) != r_key(hi), "boundary must be bracketed"  # localizer precondition
        got = localizer(r_key, lo, hi)
        assert got == bnd, f"localizer returned {got}, expected boundary {bnd} (k={k})"
        counts.append(calls[0])
    return counts

if __name__ == "__main__":
    print(f"{'k':>5} {'analytic ceil(log2(k+1))':>26} {'binary(exp-search) mean+-sd [min-max]':>42} {'linear mean':>12}")
    for k in (12, 64, 256):
        analytic = math.ceil(math.log2(k + 1))
        b = measure_localizer(exponential_search, k)
        l = measure_localizer(linear_scan, k)
        bm, bsd = statistics.mean(b), (statistics.pstdev(b) if len(b) > 1 else 0.0)
        print(f"{k:>5} {analytic:>26} {bm:>18.2f} +- {bsd:4.2f} [{min(b)}-{max(b)}]    {statistics.mean(l):>8.2f}")
    print("\nNote: counts are DISTINCT memoized probes (= HW batch measurements) for the localization search.")
    print("Full HW cost per boundary = (probes) x reps x 2 swap-directions, + 2 reverify probes (verify_reps).")
