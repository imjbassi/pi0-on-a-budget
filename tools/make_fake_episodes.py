#!/usr/bin/env python3
"""
make_fake_episodes.py — synthetic episodes in the exact recorder format.

Written offline (no real-time waiting) with realistic imperfections:
  - joints at ~50 Hz on an Arduino millis() clock with slight drift, plus
    USB-serial arrival jitter on host_mono
  - camera at ~30 fps with 33–47 ms intervals and occasional dropped frames

The image is a drawing of the arm computed from the joint angles plus a block
whose position depends on the episode's condition, so images and actions are
genuinely related. That makes these usable for pipeline, load-back and memory
tests. They say nothing about how π0 will do on real data.

Usage:
    python tools/make_fake_episodes.py --outdir D:/pi0-on-a-budget-runs/fake_episodes --episodes 12
"""

import argparse
import csv
import json
import math
import os
import shutil

import cv2
import numpy as np

CONDITIONS = {"block_left": 50.0, "block_center": 90.0, "block_right": 130.0}
JOINT_MIN = np.array([0, 15, 0, 20])
JOINT_MAX = np.array([180, 165, 180, 160])


def pick_trajectory(t, duration, block_base_deg, rng_offsets):
    """Joint angles (deg) for a reach-grasp-lift at time t."""
    phase = t / duration
    home = np.array([90.0, 120.0, 60.0, 20.0])
    reach = np.array([block_base_deg, 70.0, 120.0, 20.0]) + rng_offsets
    grasp = reach.copy()
    grasp[3] = 160.0
    lift = grasp + np.array([0.0, 40.0, -30.0, 0.0])

    def blend(a, b, s):
        s = 0.5 - 0.5 * math.cos(math.pi * min(max(s, 0.0), 1.0))
        return a + (b - a) * s

    if phase < 0.35:
        q = blend(home, reach, phase / 0.35)
    elif phase < 0.5:
        q = blend(reach, grasp, (phase - 0.35) / 0.15)
    elif phase < 0.8:
        q = blend(grasp, lift, (phase - 0.5) / 0.3)
    else:
        q = lift
    return np.clip(q, JOINT_MIN, JOINT_MAX)


def draw_scene(q_deg, block_base_deg, grasped, width=1280, height=720):
    img = np.full((height, width, 3), (60, 70, 80), dtype=np.uint8)
    cv2.rectangle(img, (0, int(height * 0.8)), (width, height), (90, 110, 120), -1)
    origin = np.array([width * 0.5, height * 0.8])

    base = math.radians(q_deg[0])
    shoulder = math.radians(q_deg[1])
    elbow = math.radians(q_deg[2])
    horizontal = math.cos(base)                    # fake perspective for base rotation
    l1, l2 = 220.0, 180.0
    p1 = origin + np.array([l1 * math.cos(shoulder) * horizontal, -l1 * math.sin(shoulder)])
    a2 = shoulder - (math.pi - elbow)
    p2 = p1 + np.array([l2 * math.cos(a2) * horizontal, -l2 * math.sin(a2)])

    block_x = int(width * 0.5 + 350 * math.cos(math.radians(block_base_deg)))
    block_center = (int(p2[0]), int(p2[1]) + 30) if grasped else (block_x, int(height * 0.8) - 25)
    cv2.rectangle(img, (block_center[0] - 25, block_center[1] - 25),
                  (block_center[0] + 25, block_center[1] + 25), (40, 40, 220), -1)

    cv2.line(img, tuple(origin.astype(int)), tuple(p1.astype(int)), (230, 230, 230), 18)
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (200, 200, 200), 14)
    opening = int(40 * (1 - (q_deg[3] - 20) / 140))
    for side in (-1, 1):
        tip = p2 + np.array([side * (8 + opening), 45])
        cv2.line(img, tuple(p2.astype(int)), tuple(tip.astype(int)), (0, 200, 255), 8)
    return img


def write_episode(outdir, index, condition, rng, task, duration):
    stem = f"episode_{index:04d}"
    frames_dir = os.path.join(outdir, f"{stem}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    block = CONDITIONS[condition] + rng.uniform(-8, 8)
    offsets = np.array([rng.uniform(-3, 3), rng.uniform(-5, 5), rng.uniform(-5, 5), 0.0])
    t0_mono = 1000.0 + index * 100.0
    drift = 1.0 + rng.uniform(-50e-6, 50e-6)

    joint_rows = []
    grasp_time = duration * 0.5
    n_joint = int(duration * 50)
    for i in range(n_joint):
        t = i * 0.020
        q = np.rint(pick_trajectory(t, duration, block, offsets)).astype(int)
        host_mono = t0_mono + t * drift + abs(rng.normal(0.004, 0.003))
        joint_rows.append({"host_time": 1.7e9 + host_mono, "device_ms": 5000 + int(round(t * 1000)),
                           "base": q[0], "shoulder": q[1], "elbow": q[2], "gripper": q[3],
                           "host_mono": host_mono})

    frame_rows = []
    t = 0.01
    frame_index = 0
    while t < duration - 0.02:
        if rng.random() > 0.01:                      # ~1% dropped frames
            q = pick_trajectory(t, duration, block, offsets)
            img = draw_scene(q, block, grasped=t > grasp_time)
            name = f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_rows.append({"frame_index": frame_index, "host_time": 1.7e9 + t0_mono + t,
                               "host_mono": t0_mono + t, "file": name})
            frame_index += 1
        t += 1 / 30 + rng.choice([0.0, 0.0, 0.0, 0.009])

    with open(os.path.join(outdir, f"{stem}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["host_time", "device_ms", "base", "shoulder", "elbow",
                                          "gripper", "host_mono"])
        w.writeheader()
        w.writerows(joint_rows)
    with open(os.path.join(outdir, f"{stem}_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "host_time", "host_mono", "file"])
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "schema_version": 2, "episode": index, "task": task, "condition": condition,
        "synthetic": True, "duration_sec": duration, "samples": len(joint_rows),
        "joints": ["base", "shoulder", "elbow", "gripper"], "joint_units": "degrees",
        "timing": {"camera_latency_corrected": False},
        "camera": {"source": "synthetic", "width": 1280, "height": 720,
                   "fps_reported_by_driver": 30.0, "frames_saved": len(frame_rows),
                   "frames_dropped_writer_behind": 0,
                   "frames_dir": f"{stem}_frames", "frames_csv": f"{stem}_frames.csv"},
    }
    with open(os.path.join(outdir, f"{stem}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.outdir) and os.listdir(args.outdir):
        if not args.overwrite:
            parser.error(f"{args.outdir} is not empty (use --overwrite)")
        for name in os.listdir(args.outdir):
            if name.endswith("_meta.json"):
                with open(os.path.join(args.outdir, name)) as f:
                    if not json.load(f).get("synthetic"):
                        parser.error(f"{args.outdir} contains real episodes ({name}); refusing to overwrite")
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    names = sorted(CONDITIONS)
    for i in range(args.episodes):
        write_episode(args.outdir, i, names[i % len(names)], rng, args.task, args.duration)
    print(f"Wrote {args.episodes} synthetic episodes to {args.outdir}")


if __name__ == "__main__":
    main()
