"""
File: AArch64-specific Configuration Options
"""
from typing import List

def try_get_cpu_vendor():
    return "aarch64"

_option_values = {
    'executor': [
        'aarch64',
    ],
    'executor_mode': [
        'P+P',
        'F+R',
    ],
    # AArch64 adds the non-interference fuzzer to the common set (x86 has no NI).
    'fuzzer': [
        'basic',
        'architectural',
        'archdiff',
        'non-iterfearence',
    ],
    # AArch64 adds the NZCV-aware input generator to the common set.
    'input_generator': [
        'random',
        'aarch64-nzcv',
    ],
    # UNUSED by AArch64 Revizor: fault injection is not implemented (the generator
    # reads this allowlist into flags that are never acted upon).
    'generator_faults_allowlist': [
    ],
    # Valid actor keys. UNUSED by AArch64 Revizor: 'observer' (multi-actor
    # non-interference), 'data_ept_properties' (stage-2), and 'fault_blocklist'
    # (no fault support) — AArch64 runs a single host/kernel actor.
    'actor': [
        'name',
        'mode',
        'privilege_level',
        'data_properties',
        'data_ept_properties',
        'observer',
        'instruction_blocklist',
        'fault_blocklist',
    ],
    # UNUSED by AArch64 Revizor: only a single 'host'/'kernel' actor is supported;
    # 'guest' and 'user' are accepted by the schema but never exercised.
    "actor_mode": [
        'host',
        'guest',
    ],
    "actor_privilege_level": [
        'kernel',
        'user',
    ],
    # UNUSED by AArch64 Revizor: these page-table property bits are NOT programmed
    # into the AArch64 page tables.
    "actor_data_properties": [
        "present",
        "is_table",
        "attribute-index-in-mair-register-0",
        "attribute-index-in-mair-register-1",
        "attribute-index-in-mair-register-2",
        "non-secure",
        "unprivileged-has-access",
        "privileged-read-only-access or dirty-bit",
        "shareable",
        "share_type",
        "accessed",
        "global",
        "dirty-bit-modifier",
        "Contiguous",
        "privileged-execute-never",
        "user-execute-never",
        "randomized",
    ],
    # UNUSED by AArch64 Revizor: stage-2 (EPT) properties; no guest/stage-2 support.
    "actor_data_ept_properties": [
        "present",
        "is_table",
        "attribute-index-in-mair-register-0",
        "attribute-index-in-mair-register-1",
        "attribute-index-in-mair-register-2",
        "non-secure",
        "unprivileged-has-access",
        "privileged-read-only-access or dirty-bit",
        "shareable",
        "share_type",
        "accessed",
        "global",
        "dirty-bit-modifier",
        "Contiguous",
        "privileged-execute-never",
        "user-execute-never",
        "randomized",
    ],
    'instruction_categories': [
            # functional
            "BASE-ARITH", "BASE-LOGICAL", "BASE-SHIFT", "BASE-BITFIELD", "BASE-BITCOUNT",
            "BASE-BITBYTE", "BASE-CONDSEL", "BASE-FLAGOP", "BASE-CRC", "BASE-PAC", "BASE-MTE",
            "BASE-MOVE", "BASE-NOP", "BASE-HINT", "BASE-SYSTEM", "BASE-BARRIER", "BASE-EXCEPTION",
            "BASE-FPSIMD",
            # legacy v1 tag names (TEMP, bisect: lets current code load an old-tag v1 base.json)
            "BASE-MEMORY-LOAD", "BASE-MEMORY-STORE", "BASE-COND-BRANCH", "BASE-UNCOND-BRANCH",
            # memory (coarse + direction + kind) and prefetch
            "BASE-MEM", "BASE-MEM-LOAD", "BASE-MEM-STORE", "BASE-MEM-ATOMIC", "BASE-MEM-EXCLUSIVE",
            "BASE-MEM-ACQREL", "BASE-MEM-COPY", "BASE-MEM-SET", "BASE-PREFETCH",
            # control flow
            "BASE-BRANCH", "BASE-BRANCH-COND", "BASE-BRANCH-UNCOND", "BASE-BRANCH-CALL",
            "BASE-BRANCH-RET", "BASE-BRANCH-INDIRECT",
            # flags
            "BASE-FLAGS", "BASE-FLAGS-WRITE", "BASE-FLAGS-READ",
            # SVE / SVE2 / SME
            "SVE", "SVE-ARITH", "SVE-LOGICAL", "SVE-COMPARE", "SVE-MOVE", "SVE-PERMUTE",
            "SVE-PREDICATE", "SVE-REDUCE", "SVE-BITCOUNT", "SVE-MEM", "SVE-MEM-LOAD",
            "SVE-MEM-STORE", "SVE-PREFETCH", "SVE-FLAGS", "SVE-FLAGS-WRITE", "SVE-FLAGS-READ",
            "SVE2-ARITH", "SVE2-CRYPTO", "SVE2-BITMANIP", "SVE2-HISTCNT", "SVE2-MATCH", "SVE2-MMLA",
            "SME", "SME-MEM", "SME-MEM-LOAD", "SME-MEM-STORE",
    ],
}

# Options present in the config schema but not (yet) implemented for AArch64.
# Reading or setting any of them raises ConfigException (see Conf.__getattr__).
_unsupported_options: List[str] = [
    "aarch64_executor_enable_prefetcher",  # prefetcher control not wired to the executor
    "aarch64_executor_enable_ssbp_patch",  # SSB mitigation toggle not implemented
    "aarch64_disable_div64",               # 64-bit division filtering not implemented
]

executor: str = try_get_cpu_vendor()
""" executor: the default executor depending on the current platform """

in_memory_assembler: bool = True
""" in_memory_assembler: AArch64 assembles test cases in memory, so skip per-test-case disk writes
during a fuzzing run (see the base config option of the same name). """

instruction_categories: List[str] = ["BASE-ARITH", "BASE-LOGICAL", "BASE-BRANCH-COND"]
""" instruction_categories: a default list of tested instruction categories """

# Instructions known to misbehave under the executor; none identified for AArch64 yet.
_buggy_instructions: List[str] = []

supported_instructions: List[str] = ["adds", "subs", "b.", "cbz", "b", "str", "ldr", "orr", "ands", "and", "eor", "cbnz", "tbz", "tbnz",
                                     "csel", "csinc", "csinv", "csneg", "ccmn", "ccmp",
                                     "sdiv", "udiv", "xpacd", "xpaci", "autda", "autdza", "autdb", "autdzb", "pacda", "pacdza", "pacdb", "pacdzb", "pacga",
                                     "bics", "rmif", "setf8", "setf16",
                                     "cls", "clz",
                                     "crc32b", "crc32h", "crc32w", "crc32x", "crc32cb", "crc32ch", "crc32cw", "crc32cx",
                                     "rbit", "rev", "rev16", "rev32",
                                     ]

# AArch64-only: relative weights for the PAC non-interference instrumentation
# (stage 1). AUTIA and XPAC strips are inserted with these weights, normalized
# against each other; signing is the unweighted baseline.
pac_auth_weight: float = 0.2
pac_xpac_weight: float = 0.2

instruction_blocklist: List[str] = [
    # Crash/stall hazards: must never be generated regardless of enabled categories.
    "eretaa", "eretab", "udf",   # exception return / undefined -> trap or EL change
    "wfe", "wfi", "wfet", "wfit",  # wait-for-event/interrupt -> stalls the harness
    # FEAT_CSSC GPR forms absent on the target HW (Neoverse N3) -> would SIGILL.
    "cnt", "ctz",
    # Not supported: these require a specific consecutive ordering of instructions.
    "setgp", "setgm", "setge",
    "setgpn", "setgmn", "setgen",
    "setgpt", "setgmt", "setget",
    "setgptn", "setgmtn", "setgetn",
    "setp", "setm", "sete",
    "setpn", "setmn", "seten",
    "setpt", "setmt", "setet",
    "setptn", "setmtn", "setetn",
    *["cpy" + first + second + third for first in ["f", ""] for second in ["p", "m", "e"] for third in ["", "n", "rn", "rt", "rtn", "rtrn", "rtwn", "t", "tn", "trn", "twn", "wn", "wt", "wtn", "wtrn", "wtwn"]],
    "caspa", "caspal", "casp", "caspl",
    # For some reason it is not recognized by our assembler
    "ldtaddal",
    "rcwcas",
    "rcwsclrpl",
    "addpt",
    "sttxr",
    "maddpt",
    "ldtaddl",
    "caspat",
    "rcwcasl",
    "swpta",
    "autibsppcr",
    "ldtsetal",
    "rcwsclrp",
    "caspalt",
    "stltxr",
    "ldatxr",
    "gcsstr",
    "msubpt",
    "rcwscasal",
    "rcwsswppa",
    "rcwcasal",
    "ldclrpal",
    "subpt",
    "rcwswppa",
    "rcwcaspa",
    "rcwsswpl",
    "rcwswppl",
    "gcssttr",
    "rcwsswp",
    "ldtp",
    "swptl",
    "autia171615",
    "autib171615",
    "rcwswpa",
    "rcwssetp",
    "caslt",
    "rcwclrpal",
    "rcwclr",
    "ldtclra",
    "ldtset",
    "pacia171615",
    "ldsetpal",
    "pacnbiasppc",
    "rcwswpal",
    "rcwsetpl",
    "rcwsswpal",
    "rcwscasl",
    "ldtadd",
    "swpp",
    "ldtsetl",
    "rcwscaspal",
    "ldiapp",
    "rcwssetpal",
    "rcwsclrpal",
    "rcwssetpa",
    "sttp",
    "ctz",
    "ldclrpa",
    "rcwsclrpa",
    "autiasppcr",
    "pacibsppc",
    "paciasppc",
    "rcwsswpa",
    "rcwswpl",
    "rcwsetpal",
    "rcwset",
    "ldtclr",
    "ldsetpl",
    "rcwsset",
    "rcwswp",
    "rcwscasa",
    "ldtadda",
    "pacnbibsppc",
    "swptal",
    "ldsetp",
    "ldtseta",
    "ldtclrl",
    "casplt",
    "rcwsclra",
    "swppal",
    "cast",
    "rcwsswppl",
    "ldclrpl",
    "rcwscaspl",
    "rcwsetal",
    "ldtxr",
    "rcwcasp",
    "stilp",
    "ldtclral",
    "pacib171615",
    "caspt",
    "ldclrp",
    "rcwclrpa",
    "rcwcaspl",
    "rcwclrpl",
    "rprfm",
    "rcwseta",
    "swppa",
    "rcwclra",
    "rcwsclrl",
    "casat",
    "rcwscasp",
    "rcwsetp",
    "rcwsetpa",
    "rcwcasa",
    "rcwscas",
    "ldsetpa",
    "rcwsetl",
    "swpt",
    "rcwssetpl",
    "rcwclrp",
    "rcwsseta",
    "rcwsswpp",
    "rcwsclral",
    "rcwswpp",
    "swppl",
    "umin",
    "umax",
    "rcwssetl",
    "smax",
    "smin",
    "rcwclral",
    "rcwssetal",
    "rcwscaspa",
    "rcwsswppal",
    "rcwclrl",
    "st64b",
    "rcwsclr",
    "rcwcaspal",
    "ret",
    "retaa",
    "retab",
    "retaasppcr",
    "retabsppcr",
    "eretaa",
    "eretab",
    "casalt",
    "abs",
    "cnt",
    "bc.",
    "blraa",
    "blraaz",
    "blrab",
    "blrabz",
    "blr",
    "br",
    "braa",
    "braaz",
    "brab",
    "brabz",
    "st64bv0",
    "rcwswppal",
    "stlp",
    "ldap",
    "ldapp",
]  # yapf: disable
instruction_blocklist.extend(_buggy_instructions)

register_allowlist: List[str] = []

# Usable by the generator: x0-x5 (plus NZCV/SP via their input slots).
# Reserved (blocked): x6-x30 and sp — used internally by the executor sandbox/instrumentation.
register_blocklist: List[str] = [
    *[f'x{number}' for number in range(6, 31)], 'sp',
    *[f'w{number}' for number in range(6, 31)], 'wsp',
    *[f'q{number}' for number in range(32)],
    *[f'v{number}' for number in range(32)],
    *[f'b{number}' for number in range(32)],
    *[f'h{number}' for number in range(32)],
    *[f's{number}' for number in range(32)],
    *[f'd{number}' for number in range(32)],
]  # yapf: disable

_actor_default = {
    'name': "main",
    'mode': "host",
    'privilege_level': "kernel",
    'observer': False,
    'data_properties': {
        "present": True,
        "is_table": False,
        "attribute-index-in-mair-register-0": False,
        "attribute-index-in-mair-register-1": False,
        "attribute-index-in-mair-register-2": False,
        "non-secure": True,
        "unprivileged-has-access": False,
        "privileged-read-only-access or dirty-bit": False,
        "shareable": False,
        "share_type": False,
        "accessed": False,
        "global": False,
        "dirty-bit-modifier": True,
        "Contiguous": True,
        "privileged-execute-never": True,
        "user-execute-never": True,
        "randomized": False,
    },
    'data_ept_properties': {
        "present": True,
        "is_table": False,
        "attribute-index-in-mair-register-0": False,
        "attribute-index-in-mair-register-1": False,
        "attribute-index-in-mair-register-2": False,
        "non-secure": True,
        "unprivileged-has-access": False,
        "privileged-read-only-access or dirty-bit": False,
        "shareable": False,
        "share_type": False,
        "accessed": False,
        "global": False,
        "dirty-bit-modifier": True,
        "Contiguous": True,
        "privileged-execute-never": True,
        "user-execute-never": True,
        "randomized": False,
    },
    'instruction_blocklist': set(),
    'fault_blocklist': set(),
}
