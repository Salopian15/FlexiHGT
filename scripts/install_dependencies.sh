#!/usr/bin/env bash
#
# Install DIAMOND and MMseqs2 for FlexiHGT.
#
# Usage: bash scripts/install_dependencies.sh [install_dir]
#
# The preferred route is conda, which handles both tools and their runtime
# dependencies:
#
#   conda create -n flexihgt -c bioconda -c conda-forge python=3.11 diamond mmseqs2
#
# This script exists for systems without conda. It installs into a directory
# you own (default ~/.local/bin), so it needs no sudo.

set -euo pipefail

INSTALL_DIR="${1:-$HOME/.local/bin}"
DIAMOND_VERSION="${DIAMOND_VERSION:-v2.1.9}"

mkdir -p "$INSTALL_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

fetch() {
    # curl is present on macOS by default; wget is more common on Linux images.
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$1" -O "$2"
    else
        echo "Error: neither curl nor wget is available." >&2
        exit 1
    fi
}

case "$(uname -s)" in
    Linux)  DIAMOND_ASSET="diamond-linux64.tar.gz"; MMSEQS_ASSET="mmseqs-linux-avx2.tar.gz" ;;
    Darwin) DIAMOND_ASSET=""                      ; MMSEQS_ASSET="mmseqs-osx-universal.tar.gz" ;;
    *)      echo "Unsupported platform: $(uname -s). Please use conda." >&2; exit 1 ;;
esac

echo "Installing into $INSTALL_DIR"

if [ -n "$DIAMOND_ASSET" ]; then
    echo "Installing DIAMOND $DIAMOND_VERSION"
    fetch "https://github.com/bbuchfink/diamond/releases/download/${DIAMOND_VERSION}/${DIAMOND_ASSET}" \
          "$WORK_DIR/diamond.tar.gz"
    tar xzf "$WORK_DIR/diamond.tar.gz" -C "$WORK_DIR"
    install -m 0755 "$WORK_DIR/diamond" "$INSTALL_DIR/diamond"
else
    # Upstream publishes no macOS binary; bioconda/homebrew do.
    echo "No prebuilt DIAMOND for macOS. Install it with:"
    echo "  conda install -c bioconda diamond   # or: brew install diamond"
fi

echo "Installing MMseqs2"
fetch "https://mmseqs.com/latest/${MMSEQS_ASSET}" "$WORK_DIR/mmseqs.tar.gz"
tar xzf "$WORK_DIR/mmseqs.tar.gz" -C "$WORK_DIR"
install -m 0755 "$WORK_DIR/mmseqs/bin/mmseqs" "$INSTALL_DIR/mmseqs"

echo
echo "Done. Ensure $INSTALL_DIR is on your PATH, e.g.:"
echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
