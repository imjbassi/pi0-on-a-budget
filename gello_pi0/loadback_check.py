#!/usr/bin/env python3
"""
loadback_check.py — load a converted dataset back through openpi's own data pipeline.

Checks, in order:
  1. openpi's create_torch_dataset returns the declared shapes, and the
     action chunk for (episode, step) equals the resampled recorder commands
     for steps k+1 .. k+H exactly — catches any off-by-one in chunking.
  2. Repack + GelloInputs + DeltaActions: joints become (target - state), gripper stays absolute.
  3. openpi's full training data loader (normalization + FAST tokenization) yields a batch
     with the shapes the model expects.

    bash wsl/run.sh python -m gello_pi0.loadback_check --config gello_fake_lora
"""

import argparse
import json
import pathlib

import numpy as np

from gello_pi0 import conventions
from gello_pi0 import episodes as eps


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  [ok] {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gello_fake_lora")
    parser.add_argument("--episodes-to-check", type=int, default=3)
    args = parser.parse_args()

    from gello_pi0 import openpi_configs
    openpi_configs.register()
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    horizon = config.model.action_horizon
    report_path = HF_LEROBOT_HOME / f"{data_config.repo_id}_conversion.json"
    report = json.loads(report_path.read_text())
    robot = conventions.RobotConvention(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in report["robot_convention"].items()
    })

    print(f"[1] raw dataset via openpi create_torch_dataset ({data_config.repo_id}, horizon {horizon})")
    dataset = _data_loader.create_torch_dataset(data_config, horizon, config.model)
    item = dataset[0]
    check(tuple(item["state"].shape) == (4,), f"state shape {tuple(item['state'].shape)}")
    check(tuple(item["actions"].shape) == (horizon, 4), f"actions shape {tuple(item['actions'].shape)}")
    check(tuple(item["image"].shape) == (3, 224, 224), f"image shape {tuple(item['image'].shape)} (CHW float)")
    check(isinstance(item["task"], str) and item["task"], f"task = {item['task']!r}")

    # Map dataset indices back to episodes: LeRobot stores episodes in save order (sorted train stems).
    train_stems = report["split"]["train"]
    offsets = np.cumsum([0] + [report["episodes"][s]["steps"] for s in train_stems])
    check(offsets[-1] == len(dataset), f"frame count {len(dataset)} matches conversion report")

    for e, stem in enumerate(train_stems[: args.episodes_to_check]):
        raw = eps.load_episode(str(pathlib.Path(report["source_episodes"]) / f"{stem}_meta.json"))
        ep = eps.resample_episode(raw, report["fps"], (report["camera_latency_ms"] or 0.0) / 1000.0)
        expected_actions = robot.deg_to_model(ep.action_deg)
        expected_state = robot.deg_to_model(ep.state_deg)
        worst = 0.0
        for k in (0, 1, len(ep) // 2, len(ep) - horizon, len(ep) - 1):
            got = dataset[int(offsets[e] + k)]
            n = min(horizon, len(ep) - k)
            worst = max(worst,
                        float(np.abs(got["actions"].numpy()[:n] - expected_actions[k:k + n]).max()),
                        float(np.abs(got["state"].numpy() - expected_state[k]).max()))
            if n < horizon:   # past episode end LeRobot repeats the last action and flags padding
                pad = got["actions_is_pad"].numpy()
                check(pad[n:].all() and not pad[:n].any(), f"{stem} step {k}: padding flags beyond episode end")
        check(worst < 1e-5, f"{stem}: state/action chunks match resampled commands (max abs diff {worst:.2e})")

    print("[2] repack + GelloInputs + DeltaActions")
    transformed = _transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    raw_item = dataset[int(offsets[0] + 5)]
    raw_np = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in raw_item.items()}
    raw_np["prompt"] = raw_np["task"]
    original_actions = raw_np["actions"].copy()          # DeltaActions modifies arrays in place
    original_state = raw_np["state"].copy()
    out = transformed(raw_np)
    delta_expected = original_actions[:, :3] - original_state[:3]
    check(np.abs(delta_expected).max() > 0, "chosen sample has non-zero joint motion")
    check(np.allclose(out["actions"][:, :3], delta_expected, atol=1e-6), "joint actions are deltas from state")
    check(np.allclose(out["actions"][:, 3], original_actions[:, 3]), "gripper action stays absolute")
    check(out["image"]["base_0_rgb"].dtype == np.uint8 and out["image"]["base_0_rgb"].shape == (224, 224, 3),
          "base_0_rgb is uint8 HWC")
    check(not out["image"]["left_wrist_0_rgb"].any(), "wrist slots are zero images")

    print("[3] full openpi training data loader (normalize + FAST tokenize), batch_size 2")
    import dataclasses
    loader = _data_loader.create_data_loader(dataclasses.replace(config, batch_size=2, num_workers=0),
                                             shuffle=True, num_batches=1, skip_norm_stats=False)
    observation, actions = next(iter(loader))
    check(tuple(actions.shape) == (2, horizon, 4), f"batch actions shape {tuple(actions.shape)}")
    check(tuple(observation.images["base_0_rgb"].shape) == (2, 224, 224, 3),
          f"batch base image {tuple(observation.images['base_0_rgb'].shape)}")
    tokens = observation.tokenized_prompt_mask.sum(axis=1)
    check(int(tokens.max()) < config.model.max_token_len,
          f"prompt+state+action tokens {int(tokens.max())} < max_token_len {config.model.max_token_len}")
    print("[OK] load-back check passed")


if __name__ == "__main__":
    main()
