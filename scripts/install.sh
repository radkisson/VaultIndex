#!/usr/bin/env bash
# Install (or re-install) the `vault` entry point and record where things live.
#
#   scripts/install.sh                      # -> ~/bin/vault, config template
#   VAULT_BIN=~/bin/vault scripts/install.sh
#
# The wrapper is COPIED, never symlinked: a symlink makes any write to ~/bin/vault
# a write to the tracked file, and a shim that execs itself is indistinguishable
# from a working install until it hangs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/vault/config"
BIN="${VAULT_BIN:-$HOME/bin/vault}"

mkdir -p "$(dirname "$BIN")"
if [ -L "$BIN" ]; then
    rm -f "$BIN"
    echo "removed symlink at $BIN"
fi
cp "$HERE/bin/vault" "$BIN"
chmod +x "$BIN"
echo "installed $BIN"

if [ ! -f "$CONFIG" ]; then
    mkdir -p "$(dirname "$CONFIG")"
    cat > "$CONFIG" <<EOF
# Machine layout for \`vault\`. Sourced by bin/vault; never committed.
# VAULTINDEX_HOME is required for a copy install: a copy in ~/bin cannot find
# the repo from its own path.
VAULTINDEX_HOME=$HERE
# VAULT_TOOLS=/path/to/vault-tools       # dir holding .venv (or the venv itself)
# VAULT_DEFAULT=/path/to/a/vault         # fallback when the cwd is not in a vault
# DB_DIR=/path/to/lancedb                # overrides the vault's .vaultindex.json
# DOMAINS_PATH=/path/to/domains.json     # taxonomy for this instance
EOF
    echo "wrote $CONFIG (sets VAULTINDEX_HOME; fill in VAULT_TOOLS)"
else
    echo "kept existing $CONFIG"
    if ! grep -q '^VAULTINDEX_HOME=' "$CONFIG"; then
        echo "VAULTINDEX_HOME=$HERE" >> "$CONFIG"
        echo "appended VAULTINDEX_HOME to $CONFIG"
    fi
fi

echo
echo "verify with:  $BIN --help && scripts/check.sh"
