#!/usr/bin/env python3
"""
run.py — run openpi's own scripts with the gello-lite configs registered.

    cd ~/projects/openpi
    uv run python -m gello_pi0.run norm-stats --config-name gello_fake_lora
    uv run python -m gello_pi0.run train gello_fake_lora --exp-name first --batch-size 1
    uv run python -m gello_pi0.run serve --policy.config gello_lora --policy.dir <checkpoint>/<step>

Everything after the subcommand is passed to openpi's script unchanged.
Set GELLO_SKIP_CHECKPOINTS=1 to turn checkpoint saving into a no-op (memory tests).
"""

import os
import pathlib
import runpy
import sys

OPENPI_DIR = pathlib.Path(os.environ.get("OPENPI_DIR", "~/projects/openpi")).expanduser()
SCRIPTS = {
    "train": "scripts/train.py",
    "norm-stats": "scripts/compute_norm_stats.py",
    "serve": "scripts/serve_policy.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        raise SystemExit(f"usage: python -m gello_pi0.run {{{','.join(SCRIPTS)}}} [openpi args...]")
    command = sys.argv[1]

    from gello_pi0 import openpi_configs
    openpi_configs.register()

    if os.environ.get("GELLO_SKIP_CHECKPOINTS") == "1":
        import openpi.training.checkpoints as checkpoints
        checkpoints.save_state = lambda *args, **kwargs: None
        print("[gello] checkpoint saving disabled (GELLO_SKIP_CHECKPOINTS=1)", flush=True)

    script = OPENPI_DIR / SCRIPTS[command]
    sys.argv = [str(script)] + sys.argv[2:]
    sys.path.insert(0, str(OPENPI_DIR))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
