#!/usr/bin/env python3
"""
convert_to_lerobot.py — recorder episodes -> LeRobot datasets for openpi.

Runs inside openpi's uv environment in WSL (needs openpi's pinned LeRobot):

    cd ~/projects/openpi
    uv run python -m gello_pi0.convert_to_lerobot \
        --episodes /mnt/d/pi0-on-a-budget-runs/fake_episodes \
        --robot-config config/robot_fake.json --name gello_fake

Writes two datasets under $HF_LEROBOT_HOME:
    local/<name>          training episodes
    local/<name>_heldout  held-out episodes (never used for training or norm stats)
and a conversion report (split, per-episode resampling stats, settings) to
    $HF_LEROBOT_HOME/local/<name>_conversion.json

Schema follows openpi's LIBERO example (examples/libero/convert_libero_data_to_lerobot.py):
    image    uint8 (224, 224, 3)   ZV-1F frame, letterboxed
    state    float32 (4,)          commanded [base, shoulder, elbow] rad + gripper 0..1
    actions  float32 (4,)          next commanded target, same units (absolute;
                                   openpi's DeltaActions converts joints to deltas)
    task     str                   language prompt
"""

import argparse
import dataclasses
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

from gello_pi0 import conventions
from gello_pi0 import episodes as eps


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="Folder of recorder episodes")
    parser.add_argument("--robot-config", required=True, help="config/robot.json (or robot_fake.json)")
    parser.add_argument("--name", required=True, help="Dataset name; repo_id becomes local/<name>")
    parser.add_argument("--fps", type=float, default=30.0, help="Dataset rate (camera rate)")
    parser.add_argument("--camera-latency-ms", type=float, default=0.0,
                        help="Shift frames earlier by this much before pairing with joints. Keep 0: "
                             "closed_loop.py pairs the newest frame (by arrival time) with the current "
                             "command, so arrival-time pairing in training matches deployment.")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def check_inputs(raw_episodes, robot):
    synthetic = [bool(ep.meta.get("synthetic")) for ep in raw_episodes]
    if any(synthetic) and not all(synthetic):
        raise SystemExit("Refusing to mix synthetic and real episodes in one dataset.")
    is_synthetic = all(synthetic)

    robot.require_gripper()
    if not is_synthetic:
        if not robot.confirmed_on_hardware:
            raise SystemExit("Robot config is not confirmed_on_hardware; real data can't be converted yet.")
    return is_synthetic


def load_frame(path):
    with Image.open(path) as img:
        return conventions.preprocess_image(np.asarray(img.convert("RGB")), bgr=False)


def write_dataset(repo_id, resampled, robot, fps, root):
    # Imported here so the rest of this module (and its tests) work without LeRobot.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out = root / repo_id
    if out.exists():
        shutil.rmtree(out)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="gello_lite",
        fps=int(round(fps)),
        features={
            "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (4,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (4,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )
    for ep in resampled:
        state = robot.deg_to_model(ep.state_deg)
        actions = robot.deg_to_model(ep.action_deg)
        for k in range(len(ep)):
            dataset.add_frame({
                "image": load_frame(ep.raw.frame_paths[ep.frame_index[k]]),
                "state": state[k],
                "actions": actions[k],
                "task": ep.raw.task,
            })
        dataset.save_episode()
    return dataset


def main(argv=None):
    args = build_parser().parse_args(argv)
    if abs(args.fps - round(args.fps)) > 1e-9:
        raise SystemExit("--fps must be an integer (LeRobot stores fps as int)")

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    robot = conventions.RobotConvention.load(args.robot_config)
    meta_paths = eps.find_episodes(args.episodes)
    if not meta_paths:
        raise SystemExit(f"No episodes in {args.episodes}")
    raw = [eps.load_episode(p) for p in meta_paths]
    is_synthetic = check_inputs(raw, robot)
    latency_s = args.camera_latency_ms / 1000.0

    train_stems, heldout_stems = eps.choose_split(raw, args.heldout_fraction, args.seed)
    by_stem = {ep.stem: ep for ep in raw}

    repo_train = f"local/{args.name}"
    repo_heldout = f"local/{args.name}_heldout"
    report_path = HF_LEROBOT_HOME / f"local/{args.name}_conversion.json"
    if (HF_LEROBOT_HOME / repo_train).exists() and not args.overwrite:
        raise SystemExit(f"{HF_LEROBOT_HOME / repo_train} exists (use --overwrite)")

    report = {
        "repo_id_train": repo_train,
        "repo_id_heldout": repo_heldout,
        "source_episodes": os.path.abspath(args.episodes),
        "synthetic": is_synthetic,
        "fps": args.fps,
        "camera_latency_ms": args.camera_latency_ms,
        "frame_pairing": "nearest frame by arrival time minus camera_latency_ms; 0 matches closed_loop.py",
        "robot_convention": dataclasses.asdict(robot),
        "state_semantics": "commanded angle at t (no servo feedback)",
        "action_semantics": "commanded angle at t + 1/fps, absolute",
        "split": {"seed": args.seed, "heldout_fraction": args.heldout_fraction,
                  "train": train_stems, "heldout": heldout_stems},
        "episodes": {},
    }

    for repo_id, stems in ((repo_train, train_stems), (repo_heldout, heldout_stems)):
        resampled = []
        for stem in stems:
            ep = eps.resample_episode(by_stem[stem], args.fps, latency_s)
            report["episodes"][stem] = {
                "split": "heldout" if repo_id == repo_heldout else "train",
                "task": ep.raw.task, "condition": ep.raw.condition, **ep.stats(),
            }
            resampled.append(ep)
        if resampled:
            write_dataset(repo_id, resampled, robot, args.fps, HF_LEROBOT_HOME)
            print(f"[OK] {repo_id}: {len(resampled)} episodes, {sum(len(e) for e in resampled)} frames")
        else:
            print(f"[--] {repo_id}: no episodes")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[OK] report: {report_path}")
    gaps = sum(e["steps_in_camera_gaps"] for e in report["episodes"].values())
    if gaps:
        print(f"[!!] {gaps} timesteps fall in camera gaps (nearest frame > 1 period away) — see report")
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
