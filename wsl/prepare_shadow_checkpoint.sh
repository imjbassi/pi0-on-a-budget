#!/usr/bin/env bash
# Make a lightweight inference checkpoint: base params + existing synthetic norm stats.
# The params are symlinked, not copied. This is for SHADOW inference only.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"

base_params="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_fast_base/params"
norm_stats="$GELLO_RUNS_DIR/assets/gello_fake_lora/local/gello_fake/norm_stats.json"
shadow_dir="$GELLO_RUNS_DIR/shadow/pi0_fast_gello_fake"
shadow_assets="$shadow_dir/assets/local/gello_fake"

test -d "$base_params" || { echo "[FAIL] missing base params: $base_params" >&2; exit 1; }
test -f "$norm_stats" || { echo "[FAIL] missing norm stats: $norm_stats" >&2; exit 1; }

mkdir -p "$shadow_assets"
ln -sfn "$base_params" "$shadow_dir/params"
cp "$norm_stats" "$shadow_assets/norm_stats.json"

echo "[OK] shadow checkpoint: $shadow_dir"
echo "[!!] Base policy + synthetic stats. Display predictions only; never execute them."
