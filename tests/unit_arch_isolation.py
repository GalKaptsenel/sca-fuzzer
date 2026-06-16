"""
Architecture import-isolation.

Guards the lazy-loading contract:
  - importing the common layer loads NO architecture package (.x86 / .aarch64) and no Unicorn;
  - running architecture A never imports architecture B's modules (nor Unicorn, for AArch64).

Each check runs in a fresh subprocess so sys.modules starts clean.
"""
import os
import sys
import subprocess
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _loaded_modules(body: str) -> set:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {REPO!r})\n"
        f"{body}\n"
        "print('MODS=' + ','.join(sorted(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, f"subprocess failed:\n{out.stderr}"
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("MODS=")][-1]
    return set(line[len("MODS="):].split(","))


def _foreign(mods: set, prefixes) -> list:
    return sorted(m for m in mods if any(m == p or m.startswith(p + ".") for p in prefixes))


class ArchIsolationTest(unittest.TestCase):
    def test_common_import_loads_no_arch_or_unicorn(self):
        mods = _loaded_modules("import src")
        self.assertEqual(_foreign(mods, ["src.x86", "src.aarch64", "unicorn"]), [],
                         "importing the common package must not load any arch package or Unicorn")

    def test_aarch64_run_loads_no_x86_or_unicorn(self):
        body = (
            "from src.config import CONF\n"
            "CONF.load('config.yml')\n"
            "from src import factory\n"
            "from src.isa_loader import InstructionSet\n"
            "isa = InstructionSet('base.json', CONF.instruction_categories)\n"
            "factory.get_program_generator(isa, 0)\n"
            "factory.get_asm_parser(factory.get_program_generator(isa, 0))\n"
            "factory.get_analyser()\n"
        )
        mods = _loaded_modules(body)
        self.assertEqual(_foreign(mods, ["src.x86", "unicorn", "src.model"]), [],
                         "the AArch64 path must not load x86, the Unicorn model, or Unicorn")

    def test_x86_run_loads_no_aarch64(self):
        body = (
            "from src.config import CONF\n"
            "CONF.instruction_set = 'x86-64'\n"
            "CONF.set_to_arch_defaults()\n"
            "from src import factory\n"
            "from src.isa_loader import InstructionSet\n"
            "isa = InstructionSet('tests/x86_tests/min_x86.json', CONF.instruction_categories)\n"
            "try:\n"
            "    factory.get_program_generator(isa, 0)\n"
            "except Exception:\n"
            "    pass\n"  # construction may need a richer spec; the import side-effect is what we test
            "factory.get_analyser()\n"
        )
        mods = _loaded_modules(body)
        self.assertEqual(_foreign(mods, ["src.aarch64"]), [],
                         "the x86 path must not load any aarch64 module")


if __name__ == "__main__":
    unittest.main()
