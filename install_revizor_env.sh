#!/bin/bash
set -e

# ====================================================
# Minimal Revizor Dev Environment for AArch64
# ====================================================

# -------------------------
# Parse command-line arguments
# -------------------------
BUILD_EXECUTABLES=false
BUILD_ONLY=false
GIT_NAME=""
GIT_EMAIL=""
GIT_TOKEN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build) BUILD_EXECUTABLES=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --git-name) GIT_NAME="$2"; shift 2 ;;
        --git-email) GIT_EMAIL="$2"; shift 2 ;;
        --git-token) GIT_TOKEN="$2"; shift 2 ;;
        *) echo "[!] Unknown option: $1"; exit 1 ;;
    esac
done

# -------------------------
# Paths
# -------------------------
BASE_DIR=~/revizor
DEST_DIR=$BASE_DIR/sca-fuzzer
VENV_DIR=$BASE_DIR/revizor-venv

mkdir -p "$BASE_DIR"

# -------------------------
# Detect kernel GCC version
# -------------------------

detect_compiler() {
    echo "[*] Detecting kernel compiler version..."
    GCC_VERSION_STR=""

    if grep -qE 'gcc-[0-9]+' /proc/version; then
        GCC_VERSION_STR=$(grep -oE 'gcc-[0-9]+' /proc/version | head -n1 | cut -d- -f2)
    elif grep -qE 'gcc version [0-9]+' /proc/version; then
        GCC_VERSION_STR=$(grep -oE 'gcc version [0-9]+' /proc/version | awk '{print $3}')
    fi

    if [ -z "$GCC_VERSION_STR" ]; then
        HEADER_MAKEFILE="/usr/src/linux-headers-$(uname -r)/Makefile"
        if [ -f "$HEADER_MAKEFILE" ]; then
            GCC_VERSION_STR=$(grep -Eo 'GCC_VERSION\s*=\s*[0-9]+' "$HEADER_MAKEFILE" | grep -Eo '[0-9]+' | head -n1)
        fi
    fi

    if [ -z "$GCC_VERSION_STR" ] && command -v gcc &>/dev/null; then
        echo "[!] Falling back to system GCC version (may differ from kernel build)."
        GCC_VERSION_STR=$(gcc -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)
    fi

    if [ -z "$GCC_VERSION_STR" ]; then
        echo "[!] Could not detect kernel GCC version. Using default aarch64-linux-gnu-gcc."
        CROSS_GCC="aarch64-linux-gnu-gcc"
    else
        echo "[*] Kernel built with GCC version $GCC_VERSION_STR"
        CROSS_GCC_VER="aarch64-linux-gnu-gcc-$GCC_VERSION_STR"

        if command -v "$CROSS_GCC_VER" &>/dev/null; then
            CROSS_GCC="$CROSS_GCC_VER"
        else
            echo "[!] $CROSS_GCC_VER not found. Installing..."
            sudo apt install -y "gcc-$GCC_VERSION_STR-aarch64-linux-gnu" "g++-$GCC_VERSION_STR-aarch64-linux-gnu" || true
            if command -v "$CROSS_GCC_VER" &>/dev/null; then
                CROSS_GCC="$CROSS_GCC_VER"
            else
                echo "[!] Still not found. Falling back to aarch64-linux-gnu-gcc."
                CROSS_GCC="aarch64-linux-gnu-gcc"
            fi
        fi
    fi

    echo "[*] Using cross-compiler: $CROSS_GCC"
}


# -------------------------
# Build executables function
# -------------------------
build_executables() {
    detect_compiler
    echo "[*] Building executor_userland..."
    pushd "$DEST_DIR/src/executor_userland"
    make CC="$CROSS_GCC"
    cp executor_userland "$BASE_DIR/"
    popd

    echo "[*] Building kernel module (revizor-executor)..."
    pushd "$DEST_DIR/src/aarch64/executor"
    make CC="$CROSS_GCC"
    cp revizor-executor.ko "$BASE_DIR/"
    popd

    echo "[*] Binaries 'executor_userland' and 'revizor-executor.ko' are in $BASE_DIR"
}


# -------------------------
# Build-only mode
# -------------------------
if [ "$BUILD_ONLY" = true ]; then
    echo "[*] Build-only mode: skipping package installation."
    cd "$DEST_DIR"
    build_executables
    echo "[*] Build-only mode complete."
    exit 0
fi

# -------------------------
# Git installation & configuration
# -------------------------
install_git() {
    sudo apt install -y git gitk
}

configure_git() {
    # Prompt only if not given via command line
    if [ -z "$GIT_NAME" ]; then
        read -rp "[?] Enter Git user.name: " GIT_NAME
    fi
    if [ -z "$GIT_EMAIL" ]; then
        read -rp "[?] Enter Git user.email: " GIT_EMAIL
    fi
    if [ -z "$GIT_TOKEN" ]; then
        read -rsp "[?] Enter GitHub token (optional for HTTPS remote): " GIT_TOKEN
        echo
    fi

    git config --local user.name "$GIT_NAME"
    git config --local user.email "$GIT_EMAIL"

    # Set remote URL with token if provided
    if [ -n "$GIT_TOKEN" ]; then
        git remote set-url origin "https://$GIT_TOKEN@github.com/GalKaptsenel/sca-fuzzer.git" || true
    fi
}

if command -v git &>/dev/null; then
    echo "[*] Git is already installed."
    read -rp "[?] Do you want to reinstall and reconfigure Git? (y/N) " REINSTALL_GIT
    if [[ "$REINSTALL_GIT" =~ ^[Yy]$ ]]; then
        install_git
        configure_git
        echo "[*] Git reinstalled and configured."
    else
        echo "[*] Reusing existing Git installation."
    fi
else
    install_git
    configure_git
    echo "[*] Git installed and configured."
fi

# -------------------------
# System update & essential packages
# -------------------------
echo "[*] Updating system..."
sudo apt update && sudo apt upgrade -y

echo "[*] Installing essential build tools, libraries, and kernel headers..."
sudo apt install -y \
    build-essential \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    make cmake autoconf automake flex bison libtool pkg-config \
    python3 python3-pip python3-venv python-is-python3 python3-dev \
    libcapstone-dev libelf-dev libdw-dev libffi-dev libssl-dev \
    zlib1g-dev libbz2-1.0 liblzma5 libzstd1 libncursesw6 libtinfo6 libxxhash0 \
    gdb gdb-multiarch lldb python3-lldb strace vbindiff \
    wget curl tar unzip screen tmux \
    linux-headers-$(uname -r)

# -------------------------
# Clone or update repository
# -------------------------
if [ ! -d "$DEST_DIR/.git" ]; then
    echo "[*] Cloning repository into $DEST_DIR..."
    git clone "https://github.com/GalKaptsenel/sca-fuzzer.git" "$DEST_DIR"
else
    echo "[*] Repository already exists at $DEST_DIR. Pulling latest changes..."
    cd "$DEST_DIR"
    git pull
fi
cd "$DEST_DIR"

# -------------------------
# Python virtual environment
# -------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# -------------------------
# Python dependencies
# -------------------------
REQ_FILE="$DEST_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "[*] Installing Python dependencies from requirements.txt..."
    pip install -r "$REQ_FILE"
else
    echo "[*] WARNING: requirements.txt not found. Generate it with pipreqs if needed."
fi

# -------------------------
# Optional build executables
# -------------------------
if [ "$BUILD_EXECUTABLES" = true ]; then
    build_executables
fi

# -------------------------
# Done
# -------------------------
echo "[*] Setup complete!"
echo "Activate the Python environment with: source $VENV_DIR/bin/activate"
echo "Your Revizor repository is located at: $DEST_DIR"
if [ "$BUILD_EXECUTABLES" = true ]; then
    echo "Binaries 'executor_userland' and 'revizor-executor.ko' are in $BASE_DIR"
fi

