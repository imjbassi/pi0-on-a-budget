#!/usr/bin/env bash
# setup_openpi.sh — install openpi inside WSL2 (Ubuntu), per openpi's README.
#
# Usage (from WSL):  bash wsl/setup_openpi.sh
#
# Pins openpi to a specific commit so results are reproducible. Change
# OPENPI_REV deliberately, not by re-running against a moving main branch.
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$HOME/projects/openpi}"
OPENPI_REV="${OPENPI_REV:-215abfb217dbac7d5f1273282331b9b1866c0479}"   # tested commit

log() { echo "[setup] $*"; }

# ---------------------------------------------------------------- GPU check
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
    echo "[FAIL] nvidia-smi does not see a GPU inside WSL. Install/update the Windows NVIDIA driver;"
    echo "       do NOT install a Linux NVIDIA driver inside WSL."
    exit 1
fi

# --------------------------------------------------------------------- uv
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ----------------------------------------------------------------- openpi
if [ ! -d "$OPENPI_DIR/.git" ]; then
    log "cloning openpi into $OPENPI_DIR"
    mkdir -p "$(dirname "$OPENPI_DIR")"
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi
cd "$OPENPI_DIR"
if [ -n "$OPENPI_REV" ]; then
    git fetch origin "$OPENPI_REV"
    git checkout "$OPENPI_REV"
fi
git submodule update --init --recursive
log "openpi at commit $(git rev-parse HEAD)"

log "uv sync (large download: JAX CUDA, PyTorch, LeRobot)"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# -------------------------------------------------------- JAX sees the GPU?
log "checking JAX CUDA"
uv run python - <<'EOF'
import jax
devices = jax.devices()
print("jax", jax.__version__, "devices:", devices)
assert any(d.platform == "gpu" for d in devices), "JAX did not find a GPU"
import jax.numpy as jnp
x = jnp.ones((2048, 2048))
print("GPU matmul ok:", float((x @ x).sum()))
EOF
log "done — openpi commit $(git rev-parse HEAD)"
