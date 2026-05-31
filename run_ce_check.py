"""
Re-run CE on the two counterexample inputs from violation-260514-100711.
Both inputs are patched to start with NZCV = nZcv (N=0 Z=1 C=0 V=0 = 0x40000000)
in the per-flag encoding before reconstruction, then compared.
"""
import sys
import struct

VDIR = "/home/gal_k_1_1998/revizor/sca-fuzzer/violation-260514-100711"

# Import as top-level package (src is the package root — its parent dir must be on the path)
sys.path.insert(0, "/home/gal_k_1_1998/revizor/sca-fuzzer")

from src.aarch64.aarch64_contract_executor import (
    ContractExecutorService, ContractExecution, SimArch, ContractExecutionResult
)
from src.aarch64.aarch64_target_desc import NZCVScheme
from src.generator import ConfigurableGenerator

CE_BIN = "/home/gal_k_1_1998/revizor/sca-fuzzer/src/aarch64/contract_executor/contract_executor"

# nZcv = N=0 Z=1 C=0 V=0  =>  per-flag slot value
# Build nZcv manually: Z=1 at byte_off=1, all others 0
_v = 1 << (1 * 8)   # Z flag in byte 49 (offset 1 within slot)
NZCV_nZcv_per_flag = _v | (_v << 32)
# Verify to_pstate gives 0x40000000
assert NZCVScheme.to_pstate(NZCV_nZcv_per_flag) == 0x40000000, \
    f"to_pstate gave {NZCVScheme.to_pstate(NZCV_nZcv_per_flag):#010x}, expected 0x40000000"
print(f"nZcv per-flag slot value : {NZCV_nZcv_per_flag:#018x}")
print(f"nZcv PSTATE after recon  : {NZCVScheme.to_pstate(NZCV_nZcv_per_flag):#010x}")


def patch_and_prepare(bin_path: str) -> tuple[bytes, bytes, int]:
    """Read a violation input bin, patch slot 6 to nZcv, reconstruct PSTATE.
    Returns (tc_memory, tc_regs, original_slot6_value)."""
    raw = bytearray(open(bin_path, "rb").read())
    # slot 6 starts at 0x2000 + 6*8 = 0x2030 (little-endian u64)
    slot_off = 0x2000 + NZCVScheme.SLOT_IDX * 8
    original = struct.unpack_from("<Q", raw, slot_off)[0]

    # Patch to nZcv per-flag
    struct.pack_into("<Q", raw, slot_off, NZCV_nZcv_per_flag)

    tc_memory = bytes(raw[:0x2000])
    tc_regs = bytearray(raw[0x2000:])

    # Reconstruct PSTATE (same as _reconstruct_pstate in aarch64_executor.py)
    view = memoryview(tc_regs).cast('Q')
    view[NZCVScheme.SLOT_IDX] = NZCVScheme.to_pstate(int(view[NZCVScheme.SLOT_IDX]))

    return tc_memory, bytes(tc_regs), original


def compute_ctrace(cer: ContractExecutionResult) -> list:
    """Reproduce compute_ctrace from aarch64_executor.py."""
    line_size, num_sets = 64, 64
    cache_sets = set()
    for ite in cer:
        if not ite.metadata.has_memory_access:
            continue
        ma = ite.metadata.memory_access
        for byte_idx in range(ma.element_size):
            addr = ma.effective_address + byte_idx - ite.cpu.gpr[29]
            cache_sets.add((addr // line_size) % num_sets)
    return sorted(cache_sets)


# Assemble the test case
with open(f"{VDIR}/generated.asm") as f:
    asm_text = f.read()
tc_bytes = ConfigurableGenerator.in_memory_assemble(asm_text)
print(f"Test case assembled: {len(tc_bytes)} bytes")

# Prepare both inputs
input_files = {
    "input_0000": f"{VDIR}/input_0000.bin",
    "input_0010": f"{VDIR}/input_0010.bin",
}

ce = ContractExecutorService(CE_BIN)

results = {}
for name, path in input_files.items():
    mem, regs, orig_slot6 = patch_and_prepare(path)
    pstate_in_regs = struct.unpack_from("<Q", regs, NZCVScheme.SLOT_IDX * 8)[0]
    print(f"\n{name}:")
    print(f"  original slot6    : {orig_slot6:#018x}")
    print(f"  patched per-flag  : {NZCV_nZcv_per_flag:#018x}")
    print(f"  PSTATE sent to CE : {pstate_in_regs:#010x}  (expect 0x40000000)")
    assert pstate_in_regs == 0x40000000, f"PSTATE mismatch: {pstate_in_regs:#010x}"

    execution = ContractExecution(
        machine_code=tc_bytes,
        memory=mem,
        registers=regs,
        arch=SimArch.RVZR_ARCH_AARCH64,
        max_misspred_branch_nesting=5,
        max_misspred_instructions=10,
    )
    cer = ce.run(execution)
    ctrace = compute_ctrace(cer)
    results[name] = ctrace
    print(f"  cache sets        : {ctrace}")

ce.stop()

print("\n--- Comparison ---")
if results["input_0000"] == results["input_0010"]:
    print("PASS: CE traces are IDENTICAL (same cache sets)")
else:
    print("FAIL: CE traces DIFFER")
    print(f"  input_0000: {results['input_0000']}")
    print(f"  input_0010: {results['input_0010']}")
    diff_only_in_0 = set(results['input_0000']) - set(results['input_0010'])
    diff_only_in_10 = set(results['input_0010']) - set(results['input_0000'])
    if diff_only_in_0:
        print(f"  extra sets in input_0000 (not in input_0010): {sorted(diff_only_in_0)}")
    if diff_only_in_10:
        print(f"  extra sets in input_0010 (not in input_0000): {sorted(diff_only_in_10)}")
