"""
File: boosted-lane construction for the regular-fuzzing cross-input priming detector.

Boosting (Fuzzer._boost_inputs) lays inputs out as R = inputs_per_class lanes of n inputs each,
order-preserving:

    [ I0, I1, ..., I(n-1),   I0', I1', ..., I(n-1)',   I0'', ..., I(n-1)'' ]
      \\------ lane 0 ------/  \\------ lane 1 -------/    \\--- lane 2 --- /

Lane r is boosted[r*n : (r+1)*n]. The members at position j across lanes (boosted[r*n + j] for each r)
belong to the SAME input class, so they are ct-equal by boosting. The cross-input priming detector's toggle at
position j is therefore "Ij vs its own boostings" -- cross-lane at the SAME position only, never
cross-position.

These helpers are the pure DATA seam: they turn the flat boosted list into lanes and enumerate which
lane pairs to compare. The search itself (the bisection) lives in cross_input.py and never sees a lane.
"""
from typing import List, Optional, Sequence, Tuple


def lanes_of(boosted: Sequence, n_orig: int) -> List[list]:
    """Split the flat boosted list into R = len(boosted) // n_orig lanes of n_orig inputs each
    (lane r = boosted[r*n : (r+1)*n]). Requires an exact multiple -- boosting appends whole lanes."""
    assert n_orig >= 1, "need at least one input class"
    assert len(boosted) % n_orig == 0, "boosted length must be a whole number of lanes"
    num_lanes = len(boosted) // n_orig
    return [list(boosted[r * n_orig:(r + 1) * n_orig]) for r in range(num_lanes)]


def reference_lane_pairs(num_lanes: int) -> List[Tuple[int, int]]:
    """The lane pairs to compare: lane 0 (the originals) against each boosted lane -- (0, 1), (0, 2),
    ... -- so the toggle at every position swaps an original for one of its boostings. O(R) pairs; the
    detector already searches each pair from both bases, so the reverse (r, 0) is redundant."""
    return [(0, r) for r in range(1, num_lanes)]


def detecting_position_and_lanes(htrace_groups: Sequence[Sequence],
                                 n_orig: int) -> Optional[Tuple[int, int, int]]:
    """Map a boosted-mode violation to what the focused search needs: (j, base_lane, toggle_lane).

    `htrace_groups` is the violation's members (one input class -- one ctrace -- across lanes) partitioned
    by htrace; two different groups are two lanes whose position-j readout diverged. Each member has an
    `.input_id` into the flat boosted list, so j = input_id % n_orig and lane = input_id // n_orig. Returns
    the class position j and two lanes drawn from two DIFFERENT groups (base from the first, toggle from a
    differing one at the same j), or None if fewer than two groups make it localizable."""
    groups = [g for g in htrace_groups if g]
    if len(groups) < 2:
        return None
    base = groups[0][0]
    j = base.input_id % n_orig
    for other in groups[1:]:
        for member in other:
            if member.input_id % n_orig == j:                 # a differing lane at the same class position
                return j, base.input_id // n_orig, member.input_id // n_orig
    return None
