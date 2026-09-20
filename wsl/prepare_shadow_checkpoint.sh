#!/usr/bin/env bash
# Make an inference checkpoint: WSL-local base params + synthetic norm stats.
# TensorStore reads of the 10.85 GB checkpoint have failed with ENOMEM when the
# params live on /mnt/d, so the safe default is a one-time copy to WSL's ext4 disk.
# This is for SHADOW inference only.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"

copy_to_wsl=true
if [[ "${1:-}" == "--link-source" ]]; then
  copy_to_wsl=false
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--link-source]" >&2
  exit 2
fi

base_params="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_fast_base/params"
norm_stats="$GELLO_RUNS_DIR/assets/gello_fake_lora/local/gello_fake/norm_stats.json"
shadow_dir="$GELLO_RUNS_DIR/shadow/pi0_fast_gello_fake"
shadow_assets="$shadow_dir/assets/local/gello_fake"
local_params="${SHADOW_PARAMS_DIR:-$HOME/pi0-cache/pi0_fast_base/params}"

test -d "$base_params" || { echo "[FAIL] missing base params: $base_params" >&2; exit 1; }
test -f "$norm_stats" || { echo "[FAIL] missing norm stats: $norm_stats" >&2; exit 1; }

params_target="$base_params"
if $copy_to_wsl; then
  mkdir -p "$local_params"
  echo "[INFO] syncing base params to WSL ext4: $local_params"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$base_params/" "$local_params/"
  else
    cp -a "$base_params/." "$local_params/"
  fi
  params_target="$local_params"
fi

mkdir -p "$shadow_assets"
if [[ -e "$shadow_dir/params" && ! -L "$shadow_dir/params" ]]; then
  echo "[FAIL] refusing to replace non-symlink: $shadow_dir/params" >&2
  exit 1
fi
ln -sfn "$params_target" "$shadow_dir/params"
cp "$norm_stats" "$shadow_assets/norm_stats.json"

echo "[OK] shadow checkpoint: $shadow_dir"
echo "[OK] params -> $(readlink -f "$shadow_dir/params")"
echo "[OK] serve with policy config: gello_fake_base"
echo "[!!] Base policy + synthetic stats. Display predictions only; never execute them."
