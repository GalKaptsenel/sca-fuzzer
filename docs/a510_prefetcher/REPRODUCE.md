# Reproducing the A510 prefetcher experiments (physical Pixel 8)

Everything needed to re-run the 2026-07-17 experiments from scratch. Results live in
`RESULTS.md`. **Read that file's RETRACTED section first** — the earlier
"violation is an F+R harness artifact" conclusion was withdrawn.

## 0. Topology

    GCP aarch64 VM (A)  --ssh tunnel :2222-->  WSL2 x86_64 (B)  --interop-->  Windows adb server  --USB-->  Pixel 8

- **A** = this repo. Generates test cases/inputs, runs the fuzzer + contract executor.
- **B** = WSL. The ONLY box that can build the GKI module (needs x86_64 + Android clang: CFI/LTO/MODVERSIONS).
- Phone = `shiba` (Pixel 8, Tensor G3), GKI `5.15.137-android14-11`, Magisk root.
- Core 0 = **Cortex-A510** (MIDR 0x411fd461, in-order little). The executor pins core 0.

## 1. Bring up the tunnels

On the **WSL/Windows** side, hold this open (`-N` blocks; that means success):

    ssh -i ~/.ssh/wsl_to_gcp -N -R 2222:localhost:22 gal_k_1_1998@<GCP_IP>

**Gotcha — the key disappears.** Google's guest agent rewrites `~/.ssh/authorized_keys` from instance
metadata, so a hand-appended key is wiped on every reboot. Durable fix: put the WSL pubkey in the
instance's SSH-keys **metadata**, not the file. Hand-append (temporary):

    echo 'ssh-ed25519 AAAA...DL4EBV gal@lap1092' >> ~/.ssh/authorized_keys

**Gotcha — the GCP IP is ephemeral** and changes on every stop/start. Reserve a static IP to stop this.

Then on **A**, expose the adb server. The Windows adb server binds 127.0.0.1 only, so WSL cannot reach it
over TCP (interop works because `adb.exe` is a Windows process). Rebind it to all interfaces:

    ssh -i ~/.ssh/revizor_remote -p 2222 gal@localhost \
      '/mnt/c/Users/gal/AppData/Local/Android/Sdk/platform-tools/adb.exe kill-server;
       nohup /mnt/c/.../adb.exe -a -P 5037 nodaemon server >/tmp/adbserver.log 2>&1 &'

    # forward A:5037 -> WSL -> Windows adb (172.21.144.1 = `ip route show default` inside WSL)
    ssh -f -N -L 5037:172.21.144.1:5037 -i ~/.ssh/revizor_remote -p 2222 \
        -o ExitOnForwardFailure=yes gal@localhost

Verify from A (no `adb` binary exists on A — the driver speaks the adb protocol directly via `ppadb`):

    python3 -c "import socket; s=socket.create_connection(('127.0.0.1',5037),timeout=5); \
                s.sendall(b'000chost:version'); print(s.recv(4), s.recv(4), s.recv(4))"   # -> OKAY 0004 0029

## 2. Build + flash the kernel module (on B)

    tar cf - -C src/aarch64/executor templates_jit.c | ssh ... 'tar xf - -C ~/sca-executor'

**Gotcha — `-C` must come BEFORE the filenames**, or tar silently syncs nothing and you build stale
source. Always verify the sync landed (`grep` a marker on B) before building.

    export PATH=$HOME/toolchains/llvm-bin:$PATH
    make -C ~/pixel-kernel/out M=~/sca-executor ARCH=arm64 LLVM=1 CC=clang CROSS_COMPILE= \
         HW_SUPPORTS_MTE=y HW_SUPPORTS_PAC=y KCFLAGS="-Wno-error -std=gnu11" modules

**Gotchas:**
- `CC=clang CROSS_COMPILE=` is REQUIRED — the Makefile hardcodes `CC := aarch64-linux-gnu-gcc`, which
  overrides `LLVM=1` and fails with "not found".
- `HW_SUPPORTS_MTE=y HW_SUPPORTS_PAC=y` is REQUIRED — the Makefile autodetects by grepping
  `/proc/cpuinfo`, which on B is **x86_64**, so it silently builds a no-MTE/no-PAC module.
- `KCFLAGS="-Wno-error -std=gnu11"` — else `-Wdeclaration-after-statement` is a WERROR.

Flash:

    adb push revizor-executor.ko /data/local/tmp/
    adb shell su -c "rmmod revizor_executor; insmod /data/local/tmp/revizor-executor.ko; chmod 666 /dev/executor"

## 3. Run the experiments (on A)

    VENV=/home/gal_k_1_1998/revizor/revizor-venv/bin/python   # /usr/bin/python3 lacks xxhash/numpy

    $VENV fast_sweep.py 100                  # EXP1: input-driven demand sweep (faulty page)
    VALUES=0x0000,0x0040,0x0080,0x00c0,0x0100,0x0001,0x0020 NORUN=1 $VENV compare_flush.py   # main page
    NORUN=1 $VENV compare_flush.py           # EXP1 with-vs-without demand-line flush (faulty)
    NORUN=1 $VENV knockout.py                # EXP2: targeted knockout of prefetched lines
    $VENV four_region.py                     # #9/#28 under the 4-region harness

## 4. How the probes work

`dc civac` is NOT in the fuzzer ISA, so the asm parser cannot emit it. `fast_sweep.py` hand-assembles the
victim with native `as`/`objcopy` and injects the bytes through the **`ex._sandboxed_cache` seam** in
`src/aarch64/aarch64_executor.py:283` — which keeps the fast batched transport.

Victim (x1 comes ONLY from the `.reif` input, so the demand address is input-driven):

    and x1, x1, #0x1fff ; add x0, x29, x1 ; ldr x3, [x0] ; dc civac, x0 ; dsb ish ; isb

The TC flushes its own demand line, so every remaining lit set is prefetcher-filled. `mkreif.py` builds the
inputs: load `input_0009.reif`, set `inp[0]['gpr'][1] = V`, `ExecutorInput(input_=inp).save(...)`.

**PERFORMANCE — do not repeat my mistake.** Use the batched executor path
(`ex.trace_test_case(exec_inputs, REPS)` = ONE super-batch = ONE round-trip): 7 points x 100 reps in
**~19 s**. `drive.py` (kept only as a cautionary example) drives `executor_userland` over `adb shell`
per rep — 5 sequential round-trips x N reps, **~100x slower** (7+ minutes for the same data).

**Other gotchas:**
- A killed driver leaves `/dev/executor` wedged mid-transaction -> `TRACE` returns `EINVAL`. Fix: reload
  the module. Kill drivers by PID (`pgrep -af ... | grep -v /bin/bash`); `pkill -f <script>.py` matches
  and kills your own shell.
- Detach long runs with `setsid`, else they die with the parent task.
- `os.chdir('/home/gal_k_1_1998/sca-fuzzer')` before `InstructionSet('base.json', ...)`.
- htrace: `HTrace.raw` bit b == set b (identity). The `report.txt` strings are bit-REVERSED (set = 63-pos).
- main and faulty **share the same 64 htrace sets** (OR'd), so a lit set alone cannot tell you which page.
  Use the knockout (flush one page's specific address) to disambiguate.
- Sets **15/23/31 are harness background** — lit even with a TC that has no matching load.

## 5. SAFETY — things that hard-reboot the phone

- **Writing `CPUECTLR_EL1`** (the prefetch control) — firmware/TF-A-trapped on this locked part. The
  `cpuectlr` sysfs knob can READ it; **never write**. TF-A reflash is not feasible.
- **Flipping PTE `AP` bit 6** (or `AF`/valid/`AttrIdx` bit 4) — PAN violation on kernel access -> reboot.
- **Modifying `x29` without restoring it** before `measurement_end` — the harness reuses it post-body.

## 6. Temporary state to revert

- `templates_jit.c` — `TEMP(lmfu)` 4-region flush/reload (flush lower->main->faulty->upper; reload the
  same, recording htrace bits only for main+faulty). Revert to the 2-page main+faulty walk for normal fuzzing.
- Debug sysfs added for this investigation: `cpuectlr`, `pte`, `pte_flip` (+ `pte_flip_region`/`pte_flip_xor`
  and `prefetch_override`/`prefetch_cpuectlr` in `executor.h`, applied in `measurement.c:execute()`).

## 7. Open question

An identical single load streams 5 lines in **faulty** but prefetches **nothing** mid-page in **main**,
despite byte-identical PTEs and identical page-relative geometry. Main's set-0 prefetch is unstable
(fires for `x1=0x0000`, not for `0x0001`, inconsistent for `0x0020`). Untested lead:
`get_stack_base_address()` = `main_region + 4096`, so TC stack traffic writes into main's high offsets and
never into faulty — asymmetric traffic that hits only main.
