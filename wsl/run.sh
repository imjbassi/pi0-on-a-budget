#!/usr/bin/env bash
# run.sh — run a command inside openpi's environment with project paths set.
#
#   bash wsl/run.sh python -m gello_pi0.convert_to_lerobot --episodes ... --name gello_fake
#   bash wsl/run.sh python -m gello_pi0.memory_probe --batch-sizes 1 2 4
#
# From Windows PowerShell:
#   wsl -d Ubuntu -- bash /mnt/c/Users/jaive.DESKTOP-3TNM9JL/Desktop/pi0-on-a-budget/wsl/run.sh python -m ...
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
exec uv run "$@"
