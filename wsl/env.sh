# Source from WSL before any openpi command:  source wsl/env.sh
#
# Keeps large files (model weights, datasets, checkpoints) on D: — the Ubuntu
# virtual disk lives on C:, which has ~40 GB free.
export GELLO_REPO="/mnt/c/Users/jaive.DESKTOP-3TNM9JL/Desktop/pi0-on-a-budget"
export GELLO_RUNS_DIR="${GELLO_RUNS_DIR:-/mnt/d/pi0-on-a-budget-runs}"
export OPENPI_DIR="${OPENPI_DIR:-$HOME/projects/openpi}"

export OPENPI_DATA_HOME="$GELLO_RUNS_DIR/openpi_cache"
export HF_LEROBOT_HOME="$GELLO_RUNS_DIR/lerobot"
export HF_HOME="$GELLO_RUNS_DIR/hf_home"
export PYTHONPATH="$GELLO_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled

mkdir -p "$OPENPI_DATA_HOME" "$HF_LEROBOT_HOME" "$HF_HOME"
cd "$OPENPI_DIR"
