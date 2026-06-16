#!/bin/bash
set -e

# ============================================================
# Revizor AArch64 environment installer
# ============================================================
# Installs and builds Revizor directly on an AArch64 (ARM64) host — the machine
# that runs the fuzzer and loads the kernel module. By default installs ONLY
# what is needed to run the fuzzer and sets up the Python venv; building the
# binaries and developer tooling are opt-in.
#
# Usage:
#   ./install_revizor_env.sh                 deps + toolchain + Python venv
#   ./install_revizor_env.sh --build         + compile the utilities
#   ./install_revizor_env.sh --dev-environment [--git-name N --git-email E [--git-token T]]
#                                            + git (developer setup)
#   ./install_revizor_env.sh --test <group>  run tests, then exit. <group> is one of:
#                                              aarch64-ce      CE C unit tests
#                                              aarch64-ko      build+load module, run /dev/executor tests
#                                              aarch64-python  all Python unit tests
#                                              aarch64-all     all of the above
#   ./install_revizor_env.sh --doc           build the docs to a standalone HTML file, then exit
#                                            (prints the path to open in a browser)
#
# Flags can be combined, e.g. --build --dev-environment.
# ------------------------------------------------------------

BUILD=false
DEV=false
TEST_GROUP=""
DOC=false
GIT_NAME=""
GIT_EMAIL=""
GIT_TOKEN=""

usage() { sed -n '6,26p' "$0"; }

if [ "$(uname -m)" != "aarch64" ]; then
    echo "[!] This installer targets AArch64 (ARM64) hosts; found '$(uname -m)'. Aborting."
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --build) BUILD=true; shift ;;
        --test) TEST_GROUP="$2"; shift 2 ;;
        --doc) DOC=true; shift ;;
        --dev-environment) DEV=true; shift ;;
        --git-name) GIT_NAME="$2"; DEV=true; shift 2 ;;
        --git-email) GIT_EMAIL="$2"; DEV=true; shift 2 ;;
        --git-token) GIT_TOKEN="$2"; DEV=true; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[!] Unknown option: $1"; usage; exit 1 ;;
    esac
done

# -------------------------
# Paths (operate in place — repo root is two levels up from src/aarch64/)
# -------------------------
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR=~/revizor
VENV_DIR=$BASE_DIR/revizor-venv
mkdir -p "$BASE_DIR"

# Package sets — kept as small as possible. (python3 pulls libncursesw6 for the
# curses dashboard; the venv's ensurepip provides pip, so no python3-pip.)
APT_RUNTIME=(python3 python3-venv)
APT_BUILD=(gcc make libc6-dev libelf-dev python3-dev)
APT_DEV=(git)

install_if_missing() {
    if dpkg -s "$1" &>/dev/null; then
        echo "[*] $1 already installed"
    else
        echo "[+] Installing $1 ..."
        sudo apt-get install -y "$1"
    fi
}

# -------------------------
# Locate the kernel build dir (some kernels lack the /lib/modules/.../build symlink)
# -------------------------
resolve_kdir() {
    local kr; kr=$(uname -r)
    if [ -d "/lib/modules/$kr/build" ]; then
        echo "/lib/modules/$kr/build"
    elif [ -d "/usr/src/linux-headers-$kr" ]; then
        echo "/usr/src/linux-headers-$kr"
    fi
}

# -------------------------
# Pick the native compiler. Prefer the gcc version the running kernel was built
# with (kernel modules are sensitive to a compiler-version mismatch).
# -------------------------
detect_compiler() {
    local ver=""
    if grep -qE 'gcc-[0-9]+' /proc/version; then
        ver=$(grep -oE 'gcc-[0-9]+' /proc/version | head -n1 | cut -d- -f2)
    elif grep -qE 'gcc version [0-9]+' /proc/version; then
        ver=$(grep -oE 'gcc version [0-9]+' /proc/version | awk '{print $3}')
    fi

    CC="gcc"
    if [ -n "$ver" ]; then
        if ! command -v "gcc-$ver" &>/dev/null; then
            sudo apt-get install -y "gcc-$ver" || true
        fi
        if command -v "gcc-$ver" &>/dev/null; then
            CC="gcc-$ver"
        fi
    fi
    echo "[*] Using compiler: $CC"
}

# -------------------------
# Build the kernel module (returns non-zero if no kernel headers are available)
# -------------------------
build_module() {
    if [ -z "${CC:-}" ]; then
        detect_compiler
    fi
    local kdir; kdir=$(resolve_kdir)
    if [ -z "$kdir" ]; then
        echo "[!] No kernel headers found; cannot build the module."
        return 1
    fi
    echo "[*] Building kernel module (revizor-executor)..."
    make -C "$DEST_DIR/src/aarch64/executor" CC="$CC" KDIR="$kdir"
}

# -------------------------
# Build the utilities
# -------------------------
build_executables() {
    detect_compiler

    echo "[*] Building asm_to_bytes..."
    make -C "$DEST_DIR/src/aarch64/asm_to_bytes" CC="$CC"

    echo "[*] Building contract_executor..."
    make -C "$DEST_DIR/src/aarch64/contract_executor" CC="$CC"

    echo "[*] Building executor_userland..."
    make -C "$DEST_DIR/src/executor_userland" CC="$CC"
    cp "$DEST_DIR/src/executor_userland/executor_userland" "$BASE_DIR/"

    if build_module; then
        cp "$DEST_DIR/src/aarch64/executor/revizor-executor.ko" "$BASE_DIR/"
    fi

    echo "[*] Build complete (executor_userland + revizor-executor.ko copied to $BASE_DIR)."
}

# -------------------------
# Tests (--test <group>)
# -------------------------
# Python tests that require the loaded kernel module (/dev/executor).
HW_PY_TESTS="tests.aarch64_tests.unit_pacga tests.aarch64_tests.unit_pac_generation \
tests.aarch64_tests.unit_pac_mistraining tests.aarch64_tests.unit_mte_random \
tests.aarch64_tests.unit_code_base tests.aarch64_tests.unit_pac_generator \
tests.aarch64_tests.unit_kernel_module"

activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "[!] venv not found at $VENV_DIR — run the installer first."
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
}

test_ce() {
    detect_compiler
    local ce_dir="$DEST_DIR/src/aarch64/contract_executor"
    echo "[*] Building contract_executor and its unit tests..."
    # contract_executor is built too: the integration test fork+execs ./contract_executor.
    make -C "$ce_dir" CC="$CC" contract_executor test_ce test_ce_integration
    echo "[*] Running contract_executor tests..."
    ( cd "$ce_dir" && ./test_ce && ./test_ce_integration )
}

test_ko() {
    build_module
    echo "[*] Reloading kernel module..."
    sudo rmmod revizor_executor 2>/dev/null || true
    sudo insmod "$DEST_DIR/src/aarch64/executor/revizor-executor.ko"
    sudo chmod 777 /dev/executor
    activate_venv
    echo "[*] Running /dev/executor tests..."
    ( cd "$DEST_DIR" && python3 -m unittest $HW_PY_TESTS )
}

test_python() {
    activate_venv
    echo "[*] Running all Python unit tests (common + aarch64)..."
    # discover does not recurse into the test subdirs, so run them separately
    # (this also keeps the x86 suite, which needs a separate env, out of scope).
    # -s and -t point at the same dir: the test dirs are namespace packages
    # (no __init__.py), which unittest discovery only handles when top == start.
    ( cd "$DEST_DIR" \
      && python3 -m unittest discover -s tests -p 'unit_*.py' -t tests \
      && python3 -m unittest discover -s tests/aarch64_tests -p 'unit_*.py' -t tests/aarch64_tests )
}

# Kernel-log patterns that mean the module faulted while a test ran.
DMESG_PATTERNS='Oops|Internal error|BUG:|Call [Tt]race|FPAC|synchronous exception|hung task|unable to handle|kernel NULL pointer'

# Run a test function and fail if the kernel logged an exception during it.
# A unique marker is written to the kernel log first, so only NEW lines are scanned.
run_with_dmesg_guard() {
    local mark="revizor-test-$(date +%s%N)"
    echo "$mark" | sudo tee /dev/kmsg >/dev/null 2>&1 || true
    "$@"; local rc=$?
    local since
    since=$(sudo dmesg 2>/dev/null | awk -v m="$mark" 'index($0,m){seen=1; next} seen')
    if printf '%s\n' "$since" | grep -qiE "$DMESG_PATTERNS"; then
        echo "[!] KERNEL EXCEPTION detected during '$*' — the module faulted:"
        printf '%s\n' "$since" | grep -iE "$DMESG_PATTERNS" | head -20
        echo "[!] The device is likely wedged; reload the module or reboot to recover. (test exit=$rc)"
        return 1
    fi
    echo "[*] dmesg clean — no kernel exceptions during '$*'."
    return $rc
}

run_tests() {
    case "$1" in
        aarch64-ce)     run_with_dmesg_guard test_ce ;;
        aarch64-ko)     run_with_dmesg_guard test_ko ;;
        aarch64-python) run_with_dmesg_guard test_python ;;
        aarch64-all)    run_with_dmesg_guard test_ce
                        run_with_dmesg_guard test_ko
                        run_with_dmesg_guard test_python ;;
        *) echo "[!] Unknown test group: '$1' (expected aarch64-all|aarch64-ce|aarch64-ko|aarch64-python)"; exit 1 ;;
    esac
}

# -------------------------
# Build the AArch64 docs to a standalone HTML file
# -------------------------
build_docs_html() {
    activate_venv
    local docs_dir="$DEST_DIR/docs/aarch64"
    ( cd "$docs_dir" && python3 build_docs.py )
    echo "[*] Docs built — open in a browser: $docs_dir/index.html"
}

# -------------------------
# Developer setup (git)
# -------------------------
setup_dev() {
    echo "[*] Developer environment: installing git..."
    for pkg in "${APT_DEV[@]}"; do install_if_missing "$pkg"; done

    if [ -n "$GIT_NAME" ]; then
        git -C "$DEST_DIR" config --local user.name "$GIT_NAME"
    fi
    if [ -n "$GIT_EMAIL" ]; then
        git -C "$DEST_DIR" config --local user.email "$GIT_EMAIL"
    fi
    if [ -n "$GIT_TOKEN" ]; then
        echo "[!] Embedding token in origin URL (stored in plaintext in .git/config)."
        git -C "$DEST_DIR" remote set-url origin \
            "https://$GIT_TOKEN@github.com/GalKaptsenel/sca-fuzzer.git" || true
    fi
}

# -------------------------
# Doc mode: build the docs HTML and exit (assumes the venv is set up)
# -------------------------
if [ "$DOC" = true ]; then
    build_docs_html
    exit 0
fi

# -------------------------
# Test mode: run the requested group and exit (assumes the env is set up)
# -------------------------
if [ -n "$TEST_GROUP" ]; then
    run_tests "$TEST_GROUP"
    exit 0
fi

# -------------------------
# Runtime dependencies + Python venv (always)
# -------------------------
echo "[*] Updating package lists..."
sudo apt-get update

echo "[*] Installing runtime dependencies and build toolchain..."
for pkg in "${APT_RUNTIME[@]}" "${APT_BUILD[@]}"; do install_if_missing "$pkg"; done

if apt-cache show "linux-headers-$(uname -r)" >/dev/null 2>&1; then
    install_if_missing "linux-headers-$(uname -r)"
else
    echo "[!] No prebuilt headers for $(uname -r) (expected on a custom kernel)."
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
echo "[*] Installing Python dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel
if [ -f "$DEST_DIR/requirements.txt" ]; then
    pip install -r "$DEST_DIR/requirements.txt"
else
    echo "[!] requirements.txt not found — skipping Python dependencies."
fi

# -------------------------
# Opt-in steps
# -------------------------
if [ "$DEV" = true ]; then
    setup_dev
fi
if [ "$BUILD" = true ]; then
    build_executables
fi

# -------------------------
# Done
# -------------------------
echo
echo "[*] Setup complete!"
echo "    Activate the venv with: source $VENV_DIR/bin/activate"
echo "    Repository: $DEST_DIR"
if [ "$BUILD" = true ]; then
    echo "    Binaries 'executor_userland' and 'revizor-executor.ko' are in $BASE_DIR"
fi
