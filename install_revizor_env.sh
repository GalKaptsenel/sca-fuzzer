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
# Build executables function
# -------------------------
build_executables() {
    echo "[*] Building executor_userland..."
    pushd "$DEST_DIR/src/executor_userland" >/dev/null
    make
    cp executor_userland "$BASE_DIR/"
    popd >/dev/null

    echo "[*] Building kernel module (executor)..."
    pushd "$DEST_DIR/src/aarch64/executor" >/dev/null
    make
    cp executor "$BASE_DIR/"
    popd >/dev/null

    echo "[*] Binaries 'executor_userland' and 'executor' are in $BASE_DIR"
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
    echo "Binaries 'executor_userland' and 'executor' are in $BASE_DIR"
fi

