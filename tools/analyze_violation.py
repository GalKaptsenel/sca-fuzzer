"""Analyze one saved non-interference violation under a chosen priming strategy, or root-cause it.

Faithful reproduction: parse the saved program (generated.asm), regenerate the inputs from the config's
input_gen_seed, and run the real fuzzing_round (NI boosting + the configured priming). The device must be
used exclusively -- the caller stops the campaign and holds the device lock. One mode per process so
config/seed state never bleeds between modes.

Usage:
  analyze_violation.py <violation_dir> mode <regular|new_both|new_xinput>
  analyze_violation.py <violation_dir> rootcause
"""
import io
import os
import sys
import contextlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.config import CONF


def _load_repro_config(vdir):
    CONF.load(os.path.join(vdir, "reproduce.yaml"))


def _n_inputs(vdir):
    # The saved sequence length / inputs_per_class = number of architectural inputs to regenerate.
    reif = [f for f in os.listdir(vdir) if f.startswith("input_") and f.endswith(".reif")]
    return max(1, len(reif) // CONF.inputs_per_class)


def _reproduce(vdir):
    """Parse the saved program, regenerate inputs from the config seed, run one fuzzing_round.
    Returns (fuzzer, violation)."""
    from src import factory
    fz = factory.get_fuzzer(os.path.join(_ROOT, "base.json"), vdir, "", [])
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(os.path.join(vdir, "generated.asm"))
    fz.input_gen.n_actors = len(tc.actors)
    inputs = fz.input_gen.generate(_n_inputs(vdir))
    violation = None
    for _ in range(max(1, CONF.minimizer_retries)):
        violation = fz.fuzzing_round(tc, inputs)
        if violation:
            break
    return fz, violation


_MODES = {
    "regular":    {"enable_cross_input_priming": False},
    "new_both":   {"enable_cross_input_priming": True, "cross_input_leaks_only": False},
    "new_xinput": {"enable_cross_input_priming": True, "cross_input_leaks_only": True},
}


def _one(vdir, name):
    """Run one reproduction under mode `name`; return (hit, pair) where pair=(leaking,detecting) or None."""
    _load_repro_config(vdir)
    for k, v in _MODES[name].items():
        setattr(CONF, k, v)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _fz, violation = _reproduce(vdir)
    if violation is None:
        return False, None
    cif = getattr(violation, "cross_input_finding", None)
    if cif is None:
        return True, None                         # regular priming confirmed (no cross-input finding)
    f = cif[0]
    return True, (f.leaking_pair, f.detecting_pair)


def mode(vdir, name, reps=1):
    """Run mode `name` `reps` times. A real leak reproduces the SAME pair consistently; an unstable or
    pair-inconsistent hit is hardware noise. STABLE requires >=80% of runs to hit the same pair."""
    from collections import Counter
    reps = max(1, int(reps))
    hits, pairs = 0, Counter()
    for _ in range(reps):
        hit, pair = _one(vdir, name)
        hits += int(hit)
        if hit:
            pairs[pair] += 1
    if hits == 0:
        print(f"MODE {name}: no-violation in {reps}/{reps} (rejected / filtered)")
        return
    top, top_n = pairs.most_common(1)[0]
    stable = reps > 1 and top is not None and top_n >= -(-reps * 4 // 5)   # ceil(0.8*reps)
    kind = "" if top is None else (" self-dependent" if top[0] == top[1] else " cross-input")
    tag = "STABLE" if stable else ("INTERMITTENT/noise" if reps > 1 else "single-shot")
    desc = "regular-confirmed" if top is None else f"leaking_pair={top[0]} detecting_pair={top[1]}{kind}"
    print(f"MODE {name}: VIOLATION {hits}/{reps} [{tag}] {desc}  (pairs={dict(pairs)})")


def rootcause(vdir):
    """Locate the leaking instruction: CE-trace the detecting input's genuine vs decoy variant and diff
    their speculative L1D cache sets. For a PAC leak the divergent set is touched by a speculative access
    behind a forged AUT* (the decoy), absent on the genuine path."""
    _load_repro_config(vdir)
    CONF.enable_cross_input_priming = True
    CONF.cross_input_leaks_only = False
    from src import factory
    from src.aarch64.aarch64_trace import show_context
    fz = factory.get_fuzzer(os.path.join(_ROOT, "base.json"), vdir, "", [])
    fz.initialize_modules()
    ex = fz.executor
    tc = fz.asm_parser.parse_file(os.path.join(vdir, "generated.asm"))
    ex.load_test_case(tc)
    inputs = fz.input_gen.generate(_n_inputs(vdir))

    import re
    rep = open(os.path.join(vdir, "report.txt")).read()
    ids = [int(m) for m in re.findall(r"Input #(\d+)\n\* Hardware trace", rep)]
    print(f"detecting/counterexample inputs from report: {ids}")
    if not ids:
        print("could not parse counterexample input ids"); return
    det = ids[0] % len(inputs)
    inp = inputs[det]
    resolved = ex._resolve(inp)
    if not getattr(ex._sealed, "_pac", None):
        print("no PAC slots in this test case -> leak source is not a PAC seal"); return

    from src.aarch64.aarch64_trace import _SANDBOX_BASE_GPR
    def sets(code_reloc):
        from src.aarch64.aarch64_relocations import apply_relocations
        cer = list(ex._ce_trace(apply_relocations(resolved.object_code, list(code_reloc)), inp))
        touched = {}
        pc0 = cer[0].cpu.pc
        for ite in cer:
            sb = ite.cpu.gpr[_SANDBOX_BASE_GPR]
            nest = ite.metadata.speculation_nesting
            for ma in ite.metadata.accesses():
                for b in range(ma.element_size):
                    cs = ((ma.effective_address + b - sb) // 64) % 64
                    touched.setdefault(cs, []).append((ite.cpu.pc - pc0, nest, ma.is_write))
        return touched, cer

    import random as _r
    print(f"input #{det}: PAC slots={len(ex._sealed._pac)}  decoy_eligible={ex.has_decoy(inp)}")
    g_sets, _ = sets(resolved.genuine())
    # forced_noncanon perturbs EVERY eligible speculative PAC slot with a guaranteed non-canonical sig --
    # the strongest possible decoy. If even it matches genuine, there is no PAC cache channel at all.
    variants = [("random-decoy", resolved.decoy(_r.Random(0)))]
    if ex.has_decoy(inp):
        variants.append(("forced-noncanon", resolved.forced_noncanon()))
    print(f"input #{det}: genuine L1D sets={sorted(g_sets)}")
    found = False
    for name, plan in variants:
        d_sets, _ = sets(plan)
        only_decoy = sorted(set(d_sets) - set(g_sets))
        print(f"  {name}: sets ONLY under decoy (the PAC leak channel) = {only_decoy or 'NONE'}")
        for cs in only_decoy:
            found = True
            for off, nest, is_w in sorted(set(d_sets[cs])):
                tag = "ARCH" if nest == 0 else f"SPEC(nest={nest})"
                print(f"      set {cs}: {'WRITE' if is_w else 'READ'} at +{off:#x} {tag}")
    print("VERDICT:", "PAC channel present (decoy-only speculative set)" if found
          else "NO PAC channel in the model -> not a PAC leak (noise/other)")


if __name__ == "__main__":
    vdir = sys.argv[1]
    if sys.argv[2] == "mode":
        mode(vdir, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else 1)
    elif sys.argv[2] == "rootcause":
        rootcause(vdir)
    else:
        sys.exit(f"unknown action {sys.argv[2]}")
