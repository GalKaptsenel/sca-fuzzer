"""Diagnostic: boost the BTB template's inputs through the real NI executor, trace on HW, and dump
per-input/per-lane merged-bitmap cache sets so we can SEE whether lanes diverge (candidate) and which
targets' sets fire. Usage: python -I tools/btb_diag.py <config.yml> <template.py> <n_inputs> <n_reps>"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONF

cfg, tmpl, n_inputs, n_reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
CONF.load(cfg)
from src.factory import get_fuzzer
from src.aarch64.template.runner import load_template, build_test_case
from src.analyser import MergedBitmapAnalyser

fuzzer = get_fuzzer("base.json", "eval_out/btb", "", "")
fuzzer.initialize_modules()
template = load_template(tmpl)
tc = build_test_case(fuzzer.generator, template)
fuzzer.input_gen.n_actors = len(tc.actors)
inputs = fuzzer.input_gen.generate(n_inputs)

fuzzer.executor.load_test_case(tc)
boosted, ctraces = fuzzer._boost_inputs(inputs, CONF.model_max_nesting)
n_orig = len(boosted) // CONF.inputs_per_class
print(f"n_orig(classes)={n_orig}  lanes(ipc)={CONF.inputs_per_class}  total_boosted={len(boosted)}")

def _dump_sysfs(tag):
    vals = {}
    for f in ("enable_view_rotation", "enable_phr_flush", "enable_pre_run_flush", "enable_ssbs",
              "pin_to_core", "measurement_mode"):
        try:
            with open(f"/sys/executor/{f}") as fh:
                vals[f] = fh.read().strip()
        except OSError as e:
            vals[f] = f"<{e}>"
    print(f"[SYSFS {tag}] " + "  ".join(f"{k}={v}" for k, v in vals.items()))


_dump_sysfs("before-trace")
htraces, _ = fuzzer.executor.trace_test_case(boosted, n_reps)
_dump_sysfs("after-trace")
outlier = CONF.analyser_outliers_threshold


def sets_of(bm):
    return sorted(i for i in range(64) if (bm >> i) & 1)


# round-major layout: lane r = boosted[r*n_orig : (r+1)*n_orig]; position j across lanes = same class
diverge = 0
for j in range(n_orig):
    bms = []
    for lane in range(CONF.inputs_per_class):
        h = htraces[lane * n_orig + j]
        bms.append(MergedBitmapAnalyser.merged_bitmap(h, outlier))
    uniq = set(bms)
    mark = ""
    if len(uniq) > 1:
        diverge += 1
        mark = "  <-- LANES DIVERGE"
    # only print diverging or first 8
    if mark or j < 6:
        cols = "  ".join(f"L{l}:{sets_of(b)}" for l, b in enumerate(bms))
        print(f"class {j:3d}: {cols}{mark}")
print(f"\nDIVERGING CLASSES: {diverge} / {n_orig}")
# which sets ever appear anywhere
allsets = set()
for h in htraces:
    allsets |= set(sets_of(MergedBitmapAnalyser.merged_bitmap(h, outlier)))
print(f"all cache sets observed across batch: {sorted(allsets)}")
