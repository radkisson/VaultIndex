#!/usr/bin/env bash
# Verify the engine end to end. One command, no arguments required.
#
#   VAULT_TOOLS=/path/to/vault-tools  where the venv lives (else ~/.config/vault/config)
#   VAULT_PATH=/path/to/vault         optional: also smoke-test a real vault
#
# Fails loudly on: a failing fixture test, an installed wrapper that differs from
# the tracked one, or a smoke test that cannot plan.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/vault/config"
# shellcheck disable=SC1090
[ -f "$CONFIG" ] && . "$CONFIG"

VT="${VAULT_TOOLS:-${XDG_DATA_HOME:-$HOME/.local/share}/vault-tools-venv}"
PY="$VT/.venv/bin/python"
[ -x "$PY" ] || PY="$VT/bin/python"

if [ ! -x "$PY" ]; then
    echo "check: no python at $PY" >&2
    echo "       set VAULT_TOOLS in $CONFIG (see docs/REPRODUCE.md)" >&2
    exit 1
fi

fail=0

echo "== fixture tests (synthetic vectors, known ground truth) =="
"$PY" "$HERE/tests/test_organize.py" || fail=1

# The installed wrapper is a copy, so it does not follow repo edits. A stale copy
# behaves exactly like a working one until the behaviour changes, so compare.
INSTALLED="${VAULT_BIN:-$HOME/bin/vault}"
if [ -e "$INSTALLED" ]; then
    if [ -L "$INSTALLED" ]; then
        echo "== installed wrapper =="
        echo "   FAIL: $INSTALLED is a symlink; a write to it edits the tracked file"
        echo "         (and a self-exec shim looks identical to a working install)"
        fail=1
    elif cmp -s "$HERE/bin/vault" "$INSTALLED"; then
        echo "== installed wrapper matches the repo =="
    else
        echo "== installed wrapper =="
        echo "   FAIL: $INSTALLED differs from bin/vault — run scripts/install.sh"
        fail=1
    fi
else
    echo "== installed wrapper: none at $INSTALLED (skipped) =="
fi

if [ -n "${VAULT_PATH:-}" ] && [ -d "${VAULT_PATH}" ]; then
    echo "== plan smoke on $VAULT_PATH =="
    if VAULT_PATH="$VAULT_PATH" "$PY" "$HERE/organize.py" plan --limit 1 --json > /dev/null; then
        echo "plan ok"
    else
        echo "   FAIL: plan smoke failed"
        fail=1
    fi
else
    echo "== plan smoke skipped (VAULT_PATH unset) =="
fi

if [ "$fail" -ne 0 ]; then
    echo "== FAILED =="
    exit 1
fi
echo "== done =="
