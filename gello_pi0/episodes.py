"""
episodes.py — load recorder episodes and resample them onto a fixed-rate timeline.

LeRobot assumes frame k of an episode happens at k / fps, and openpi builds
action chunks by looking up the next `action_horizon` frames. So the dataset
must be uniformly sampled. The recorder's streams are not:
  - joints arrive at ~50 Hz (Arduino millis() clock, jittery USB arrival)
  - camera frames arrive at ~30 fps with 33–47 ms intervals

Resampling, per episode:
  1. joint sample times = least-squares fit of host_mono against device_ms
     (regular Arduino ticks, mapped onto the PC clock)
  2. frame times = host_mono - camera_latency_s (when the pictured scene happened)
  3. timeline t_k = start + k / fps over the overlap of both streams
  4. image[k]  = nearest camera frame to t_k   (offset recorded, never hidden)
     state[k]  = joint command interpolated at t_k
     action[k] = joint command interpolated at t_{k+1}  (the next target)

"state" is the commanded angle, not a measured position — the SG90/MG90S servos
give no feedback.
"""

import csv
import dataclasses
import glob
import json
import math
import os
import random

import numpy as np


@dataclasses.dataclass
class RawEpisode:
    stem: str
    folder: str
    meta: dict
    joint_times: np.ndarray        # (N,) seconds, host_mono clock
    joints_deg: np.ndarray         # (N, 4)
    frame_times: np.ndarray        # (M,) seconds, host_mono clock (latency uncorrected)
    frame_paths: list

    @property
    def task(self):
        return self.meta.get("task", "unspecified")

    @property
    def condition(self):
        return self.meta.get("condition")


@dataclasses.dataclass
class ResampledEpisode:
    raw: RawEpisode
    fps: float
    times: np.ndarray              # (K,)
    frame_index: np.ndarray        # (K,) index into raw.frame_paths
    frame_offset_s: np.ndarray     # (K,) |t_k - chosen frame time|
    state_deg: np.ndarray          # (K, 4)
    action_deg: np.ndarray         # (K, 4)

    def __len__(self):
        return len(self.times)

    def stats(self):
        offsets_ms = self.frame_offset_s * 1000.0
        return {
            "steps": int(len(self)),
            "fps": self.fps,
            "frame_offset_ms_median": round(float(np.median(offsets_ms)), 2),
            "frame_offset_ms_max": round(float(offsets_ms.max()), 2),
            # A step whose nearest frame is more than one period away sits in a camera gap.
            "steps_in_camera_gaps": int(np.sum(self.frame_offset_s > 1.0 / self.fps)),
            "frames_reused": int(len(self) - len(np.unique(self.frame_index))),
        }


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_episode(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    folder = os.path.dirname(os.path.abspath(meta_path))
    stem = os.path.basename(meta_path).removesuffix("_meta.json")

    rows = _read_rows(os.path.join(folder, f"{stem}.csv"))
    if len(rows) < 2:
        raise ValueError(f"{stem}: fewer than 2 joint samples")
    device_s = np.array([float(r["device_ms"]) for r in rows]) / 1000.0
    host_mono = np.array([float(r["host_mono"]) for r in rows])
    if np.any(np.diff(device_s) <= 0):
        raise ValueError(f"{stem}: device_ms not strictly increasing (Arduino reset mid-episode?)")
    slope, intercept = np.polyfit(device_s, host_mono, 1)
    joint_times = slope * device_s + intercept
    joints_deg = np.array([[float(r[j]) for j in ("base", "shoulder", "elbow", "gripper")] for r in rows])

    frame_rows = _read_rows(os.path.join(folder, f"{stem}_frames.csv"))
    if len(frame_rows) < 2:
        raise ValueError(f"{stem}: fewer than 2 camera frames")
    frames_dir = os.path.join(folder, meta.get("camera", {}).get("frames_dir", f"{stem}_frames"))
    frame_times = np.array([float(r["host_mono"]) for r in frame_rows])
    frame_paths = [os.path.join(frames_dir, r["file"]) for r in frame_rows]

    return RawEpisode(stem, folder, meta, joint_times, joints_deg, frame_times, frame_paths)


def resample_episode(raw, fps, camera_latency_s=0.0):
    frame_times = raw.frame_times - camera_latency_s
    start = max(frame_times[0], raw.joint_times[0])
    end = min(frame_times[-1], raw.joint_times[-1])
    n_points = int(math.floor((end - start) * fps + 1e-9)) + 1
    if n_points < 3:
        raise ValueError(f"{raw.stem}: streams overlap for only {end - start:.3f}s")

    grid = start + np.arange(n_points) / fps
    commands = np.stack([np.interp(grid, raw.joint_times, raw.joints_deg[:, j]) for j in range(4)], axis=1)

    times = grid[:-1]                       # last point only serves as the final action target
    idx = np.clip(np.searchsorted(frame_times, times), 1, len(frame_times) - 1)
    left, right = frame_times[idx - 1], frame_times[idx]
    use_left = np.abs(times - left) <= np.abs(right - times)
    frame_index = np.where(use_left, idx - 1, idx)
    frame_offset = np.abs(times - frame_times[frame_index])

    return ResampledEpisode(
        raw=raw,
        fps=fps,
        times=times,
        frame_index=frame_index,
        frame_offset_s=frame_offset,
        state_deg=commands[:-1],
        action_deg=commands[1:],
    )


def find_episodes(folder):
    return sorted(glob.glob(os.path.join(folder, "episode_*_meta.json")))


def choose_split(raw_episodes, heldout_fraction=0.2, seed=0):
    """Deterministic held-out split, stratified by (task, condition).

    Every group with >= 2 episodes contributes ceil(fraction * n) held-out
    episodes (at least 1), so each condition can be evaluated open-loop.
    Groups with a single episode go entirely to training.
    """
    groups = {}
    for ep in raw_episodes:
        groups.setdefault((ep.task, ep.condition), []).append(ep.stem)

    train, heldout = [], []
    rng = random.Random(seed)
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        stems = sorted(groups[key])
        if len(stems) < 2 or heldout_fraction <= 0:
            train.extend(stems)
            continue
        n_heldout = min(len(stems) - 1, max(1, math.ceil(heldout_fraction * len(stems))))
        shuffled = stems[:]
        rng.shuffle(shuffled)
        heldout.extend(sorted(shuffled[:n_heldout]))
        train.extend(sorted(shuffled[n_heldout:]))
    return sorted(train), sorted(heldout)
