"""
File: x86-specific Configuration Options

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
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
    'generator_faults_allowlist': [
    ],
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
    "actor_mode": [
        'host',
        'guest',
    ],
    "actor_privilege_level": [
        'kernel',
        'user',
    ],
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
    "actor_data_ept_properties": [
        # What is the different from above?
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
        "general",
        "BASE-BINARY",
        "BASE-BITBYTE",
        "BASE-CMOV",
        "BASE-COND_BR",
        "BASE-CONVERT",
        "BASE-DATAXFER",
        "BASE-FLAGOP",
        "BASE-LOGICAL",
        "BASE-MISC",
        "BASE-NOP",
        "BASE-POP",
        "BASE-PUSH",
        "BASE-SEMAPHORE",
        "BASE-SETCC",
        "BASE-STRINGOP",
        "BASE-WIDENOP",

        # Base x86 - system instructions
        "BASE-INTERRUPT",
        # "BASE-ROTATE",      # Unknown bug in Unicorn - emulated incorrectly
        # "BASE-SHIFT",       # Unknown bug in Unicorn - emulated incorrectly
        # "BASE-UNCOND_BR",   # Not supported: Complex control flow
        # "BASE-CALL",        # Not supported: Complex control flow
        # "BASE-RET",         # Not supported: Complex control flow
        # "BASE-SEGOP",       # Not supported: System instructions
        # "BASE-IO",          # Not supported: System instructions
        # "BASE-IOSTRINGOP",  # Not supported: System instructions
        # "BASE-SYSCALL",     # Not supported: System instructions
        # "BASE-SYSRET",      # Not supported: System instructions
        "BASE-SYSTEM",
        "LONGMODE-CONVERT",
        "LONGMODE-DATAXFER",
        "LONGMODE-SEMAPHORE",
        "LONGMODE-SYSCALL",
        "LONGMODE-SYSRET",

        # SIMD extensions
        "SSE-SSE",
        "SSE-DATAXFER",
        "SSE-MISC",
        "SSE-LOGICAL_FP",
        # "SSE-CONVERT",  # require MMX
        # "SSE-PREFETCH",  # prefetch does not trigger a mem access in unicorn
        "SSE2-SSE",
        "SSE2-DATAXFER",
        "SSE2-MISC",
        "SSE2-LOGICAL_FP",
        "SSE2-LOGICAL",
        # "SSE2-CONVERT",  # require MMX
        # "SSE2-MMX",   # require MMX
        "SSE3-SSE",
        "SSE3-DATAXFER",
        # "SSE4-SSE",  # not tested yet
        "SSE4-LOGICAL",
        "SSE4a-BITBYTE",
        "SSE4a-DATAXFER",

        # Misc
        "CLFLUSHOPT-CLFLUSHOPT",
        "CLFSH-MISC",
        "MPX-MPX",
        "SMX-SYSTEM",
        "VTX-VTX",
        "XSAVE-XSAVE",
    ],
}

# by default, we always handle page faults
# _handled_faults: List[str] = ["PF"]

aarch64_executor_enable_prefetcher: bool = False
""" x86_executor_enable_prefetcher: enable all prefetchers"""
aarch64_executor_enable_ssbp_patch: bool = True
""" x86_executor_enable_ssbp_patch: enable a patch against Speculative Store Bypass"""
# x86_enable_hpa_gpa_collisions: bool = False
# """ x86_enable_hpa_gpa_collisions: enable collisions between HPA and GPA;
# useful for testing Foreshadow-like leaks"""
aarch64_disable_div64: bool = True
""" x86_disable_div64: do not generate 64-bit division instructions """
# x86_generator_align_locks: bool = True
# """ x86_generator_align_locks: align all generated locks to 8 bytes """

# Overwrite executor
executor: str = try_get_cpu_vendor()
""" executor: the default executor depending on the current platform """

instruction_categories: List[str] = ["BASE-BINARY", "BASE-BITBYTE", "BASE-COND_BR"]
""" instruction_categories: a default list of tested instruction categories """

_buggy_instructions: List[str] = [
    # "sti",  # enables interrupts
    # "cli",  # disables interrupts; blocked just in case
    # "xlat",  # requires support of segment registers
    # "xlatb",  # requires support of segment registers
    # "cmpxchg8b",  # known bug: doesn't execute the mem. access hook
    # "lock cmpxchg8b",  # https://github.com/unicorn-engine/unicorn/issues/990
    # "cmpxchg16b",  # known bug: doesn't execute the mem. access hook
    # "lock cmpxchg16b",  # https://github.com/unicorn-engine/unicorn/issues/990
    # "cpuid",  # causes false positives: the model and the CPU will likely have different values
    # "cmpps",  # causes crash
    # "cmpss",  # causes crash
    # 'cmppd',  # causes crash
    # 'cmpsd',  # causes crash
    # "movq2dq",  # requires MMX
    # 'movdq2q',  # requires MMX
    # "rcpps",  # incorrect emulation
    # "rcpss",  # incorrect emulation
    # "maskmovdqu",  # incorrect emulation
]

supported_instructions: List[str] = ["add", "sub", "b.", "cbz", "b", "str", "ldr", "orr", "and", "eor", "cbnz"]

instruction_blocklist: List[str] = [
    # Currently don't support them - they require very specific order of instruction (they must appear one after the other, what happens otherwise? I don't know)
    "setgp", "setgm", "setge",
    "setgpn", "setgmn", "setgen",
    "setgpt", "setgmt", "setget",
    "setgptn", "setgmtn", "setgetn",
    "setp", "setm", "sete",
    "setpn", "setmn", "seten",
    "setpt", "setmt", "setet",
    "setptn", "setmtn", "setetn",
    *["cpy" + first + second + third for first in ["f", ""] for second in ["p", "m", "e"] for third in ["", "n", "rn", "rt", "rtn", "rtrn", "rtwn", "t", "tn", "trn", "twn", "wn", "wt", "wtn", "wtrn", "wtwn"]],
    "caspa", "caspal", "casp", "caspl", "casp",
    # For some reason it does not recognized by our assembler
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
    "pacib171615i",
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
    "retabsppcr",
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
    "retaasppcr",
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

    # Assembel says the cpu does not cupport those instructions



    # # Hard to fix:
    # # - Requires complex instrumentation
    # "enterw", "enter", "leavew", "leave",
    # # - requires support of all possible interrupts
    # "int",
    # # - system management instruction
    # "encls", "vmxon", "stgi", "skinit", "ldmxcsr", "stmxcsr",
    #
    # # - not supported
    # "lfence", "mfence", "sfence", "clflush", "clflushopt",
    #
    # # - under construction
    # # -- trigger FPVI (we have neither a contract nor an instrumentation for it yet)
    # "divps", "divss", 'divpd', 'divsd',
    # "mulss", "mulps", 'mulpd', 'mulsd',
    # "rsqrtps", "rsqrtss", "sqrtps", "sqrtss", 'sqrtpd', 'sqrtsd',
    # 'addps', 'addss', 'addpd', 'addsd',
    # 'subps', 'subss', 'subpd', 'subsd',
    # 'addsubpd', 'addsubps', 'haddpd', 'haddps', 'hsubpd', 'hsubps',
]  # yapf: disable
instruction_blocklist.extend(_buggy_instructions)

register_allowlist: List[str] = [
    ]
# aarch64 executor internally uses x15...x22, x30, SP, thus, they are excluded
register_blocklist: List[str] = [
    # free - x0 - x14, x23-x29
    *[f'x{number}' for number in range(15, 23)], 'x30', 'sp',
    *[f'x{number}' for number in range(23, 30)],
    *[f'x{number}' for number in range(6, 15)],
    *[f'w{number}' for number in range(15, 23)], 'w30', 'wsp',
    *[f'w{number}' for number in range(23, 30)],
    *[f'w{number}' for number in range(6, 15)],
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

