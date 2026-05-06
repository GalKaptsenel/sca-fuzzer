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
            "BASE-ARITH",
            "BASE-LOGICAL",
            "BASE-SHIFT",
            "BASE-BITFIELD",
            "BASE-BITCOUNT",
            "BASE-CONDSEL",
            "BASE-BRANCH",
            "BASE-MEMORY-LOAD",
            "BASE-MEMORY-STORE",
            "BASE-ATOMIC",
            "BASE-EXCLUSIVE",
            "BASE-ACQUIRE",
            "BASE-CRC",
            "BASE-MTE",
            "BASE-PAC",
            "BASE-COPY",
            "BASE-SYSTEM",
            "BASE-FLAG",
            "SVE-ARITH",
            "SVE-LOGICAL",
            "SVE-MEMORY-LOAD",
            "SVE-MEMORY-STORE",
            "SVE-PREDICATE",
            "SVE-MOVE",
            "SVE-REDUCE",
            "SVE-BITCOUNT",
            "SVE-PERMUTE",
            "SVE-COMPARE",
            "SVE-MISC",
            "SVE2-ARITH",
            "SVE2-CRYPTO",
            "SVE2-BITMANIP",
            "SVE2-HISTCNT",
            "SVE2-MATCH",
            "SVE2-MMLA",
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

supported_instructions: List[str] = ["adds", "subs", "b.", "cbz", "b", "str", "ldr", "orr", "ands", "eor", "cbnz",
                                     "csel", "csinc", "csinv", "csneg", "ccmn", "ccmp",
                                     "sdiv", "udiv"
                                     ]
#                                     "ldp", "stp"]

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
    "stlp",
    "ldap",
    "ldapp",

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
# aarch64 executor internally uses x15...x22, x29, SP, thus, they are excluded
register_blocklist: List[str] = [
    # free - x0 - x14, x23-x29
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

        "reserved_bit": False # TODO: Does Aarch64 has reserved bits? For what reason do we need them?
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

        "reserved_bit": False # TODO: Does Aarch64 has reserved bits? For what reason do we need them?
    },
    'instruction_blocklist': set(),
    'fault_blocklist': set(),
}

