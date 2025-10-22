#!/bin/bash
set -e

# ====================================================
# Minimal Revizor Dev Environment for AArch64
# ====================================================

# -------------------------
# Parse command-line arguments
# -------------------------
BUILD_EXECUTABLES=false
GIT_NAME=""
GIT_EMAIL=""
GIT_TOKEN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_EXECUTABLES=true
            shift
            ;;
        --git-name)
            GIT_NAME="$2"
            shift 2
            ;;
        --git-email)
            GIT_EMAIL="$2"
            shift 2
            ;;
        --git-token)
            GIT_TOKEN="$2"
            shift 2
            ;;
        *)
            echo "[!] Unknown option: $1"
            exit 1
            ;;
    esac
done

# -------------------------
# 1️⃣ Update and upgrade system
# -------------------------
echo "[*] Updating system..."
sudo apt update && sudo apt upgrade -y

# -------------------------
# 2️⃣ Install essential apt packages, including kernel headers
# -------------------------
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
    git wget curl tar unzip screen tmux \
    linux-headers-$(uname -r)

# Warn if headers not found
if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
    echo "[!] WARNING: Kernel headers not found for $(uname -r). Module compilation may fail."
fi

# -------------------------
# 3️⃣ Clone your Revizor repository
# -------------------------
REPO_URL="https://github.com/GalKaptsenel/sca-fuzzer.git"
BASE_DIR=~/revizor
DEST_DIR=$BASE_DIR/sca-fuzzer

mkdir -p "$BASE_DIR"

if [ ! -d "$DEST_DIR/.git" ]; then
    echo "[*] Cloning repository into $DEST_DIR..."
    git clone "$REPO_URL" "$DEST_DIR"
else
    echo "[*] Repository already exists at $DEST_DIR. Pulling latest changes..."
    cd "$DEST_DIR"
    git pull
fi

cd "$DEST_DIR"

# -------------------------
# 4️⃣ Configure Git user identity and token
# -------------------------
if [ -z "$GIT_NAME" ]; then
    read -rp "[?] Enter Git user.name: " GIT_NAME
fi

if [ -z "$GIT_EMAIL" ]; then
    read -rp "[?] Enter Git user.email: " GIT_EMAIL
fi

git config --local user.name "$GIT_NAME"
git config --local user.email "$GIT_EMAIL"
echo "[*] Git configured: $GIT_NAME <$GIT_EMAIL>"

if [ -z "$GIT_TOKEN" ]; then
    read -rsp "[?] Enter GitHub personal access token (leave empty if not using HTTPS push): " GIT_TOKEN
    echo
fi

if [ -n "$GIT_TOKEN" ]; then
    REMOTE_URL="https://$GIT_TOKEN@github.com/GalKaptsenel/sca-fuzzer.git"
    git remote set-url origin "$REMOTE_URL"
    echo "[*] Remote URL updated with token for HTTPS authentication"
fi

# -------------------------
# 6️⃣ Set up Python virtual environment
# -------------------------
VENV_DIR=~/revizor-venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# -------------------------
# 7️⃣ Install Python dependencies
# -------------------------
REQ_FILE="$DEST_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "[*] Installing Python dependencies from requirements.txt..."
    pip install -r "$REQ_FILE"
else
    echo "[*] WARNING: requirements.txt not found. You can generate it using pipreqs."
fi

# -------------------------
# 8️⃣ Optional: build executor_userland and kernel module
# -------------------------
if [ "$BUILD_EXECUTABLES" = true ]; then
    echo "[*] Building executor_userland..."
    pushd "$DEST_DIR/src/executor_userland"
    make
    cp executor_userland "$BASE_DIR/"
    popd

    echo "[*] Building kernel module (executor)..."
    pushd "$DEST_DIR/src/aarch64/executor"
    make
    cp executor "$BASE_DIR/"
    popd
fi

# -------------------------
# 9️⃣ Done
# -------------------------
echo "[*] Setup complete!"
echo "Activate the Python environment with: source $VENV_DIR/bin/activate"
echo "Your Revizor repository is located at: $DEST_DIR"
if [ "$BUILD_EXECUTABLES" = true ]; then
    echo "Binaries 'executor_userland' and 'executor' are in $BASE_DIR"
fi

