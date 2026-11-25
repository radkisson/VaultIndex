#!/usr/bin/env bash
# Install the acceptance gate as a git hook.
#
#   scripts/install-hooks.sh
#
# Hooks are not tracked by git, so this is how a clone gets the gate. The hook
# refuses commits that touch the engine when check.sh fails; --no-verify bypasses.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS="$(git -C "$HERE" rev-parse --git-path hooks)"

mkdir -p "$HOOKS"
cp "$HERE/scripts/hooks/pre-commit" "$HOOKS/pre-commit"
chmod +x "$HOOKS/pre-commit"
echo "installed $HOOKS/pre-commit"
