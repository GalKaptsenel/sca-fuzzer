# Revizor — AArch64 (ARM64) port

Microarchitectural side-channel (Spectre-style) contract fuzzer for AArch64.
Runs on the ARM64 host under test: a Python fuzzer drives a kernel module
(`/dev/executor`) that executes generated test cases on real hardware, and a
contract executor models the architectural/speculative contract to compare
against.

> ⚠️ **Under active development.** The AArch64 port is a work in progress —
> interfaces, configs, and behavior may change, and some features are
> incomplete or experimental. Expect rough edges.

## 1. Install

Run the installer on the AArch64 host (it must be run *on* an ARM64 machine):

```bash
cd src/aarch64
./install_revizor_env.sh            # deps + toolchain + Python venv (run-only)
./install_revizor_env.sh --build    # also compile the kernel module + utilities
```

Optional flags (combinable):

- `--dev-environment` — also set up git for development.
- `--test <group>` — run tests then exit; `<group>` is one of `aarch64-ce`,
  `aarch64-ko`, `aarch64-python`, `aarch64-all`.

After building, load the kernel module and open the device:

```bash
sudo insmod src/aarch64/executor/revizor-executor.ko
sudo chmod 777 /dev/executor
```

### Download the AArch64 instruction set

The fuzzer needs an instruction-set spec (`base.json`); it is **not** shipped and
is generated from ARM's machine-readable A64 ISA. Download and build it once:

```bash
python revizor.py download_spec -a aarch64 -o base.json
```

This fetches the ARM A64 ISA XML release from `developer.arm.com` (cached locally
after the first run) and parses it into `base.json`, which the `fuzz`/`tfuzz`
commands consume via `-s base.json`. Restrict to specific instruction-class
categories with `--extensions <cat> ...` (default: all).

## 2. Documentation

Build the full manual to a standalone HTML file:

```bash
cd src/aarch64
./install_revizor_env.sh --doc      # prints the path to open in a browser
```

The source for the manual lives in [`docs/aarch64/index.md`](../../docs/aarch64/index.md)
(install / running / detecting Spectre-v1 / testing / internals).

## 3. Status

This port is **still under development**. It is usable for the documented
workflows (e.g. the Spectre-v1 detection configs in `configs/`), but is not yet
feature-complete or hardened — see the "What we do not do (yet)" section of the
documentation for known limitations.
