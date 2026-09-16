#!/usr/bin/env python3
"""
sync_report.py — timing and synchronization check for recorded episodes.

Answers the questions that decide how an episode can be converted:
  - What rate did the joint stream and the camera actually deliver?
  - Were there gaps (dropped frames, stalled serial)?
  - How far is each camera frame from the nearest joint sample?

Serial arrival times on the host are jittery (USB-serial adapters buffer
lines), but the Arduino's millis() ticks are regular. So joint sample times
are taken from a least-squares fit host_mono ≈ a·device_ms + b, and the
residual of that fit is reported as serial arrival jitter.

What this can NOT measure: the fixed camera latency (exposure + USB + decode)
between a physical event and the frame's timestamp. That needs the real arm —
see README.

Usage:
    python sync_report.py episodes/episode_0000_meta.json
    python sync_report.py episodes/            # every episode in the folder
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

GAP_FACTOR = 1.5    # an interval > 1.5x the median counts as a gap


def _read_csv_columns(path, columns):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) for r in rows]) for c in columns}


def _interval_stats(times_s):
    if len(times_s) < 2:
        return {"median_ms": None, "p95_ms": None, "max_ms": None, "gaps": 0}
    dt = np.diff(times_s) * 1000.0
    median = float(np.median(dt))
    return {
        "median_ms": round(median, 2),
        "p95_ms": round(float(np.percentile(dt, 95)), 2),
        "max_ms": round(float(dt.max()), 2),
        "gaps": int(np.sum(dt > GAP_FACTOR * median)),
    }


def analyze_episode(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    folder = os.path.dirname(meta_path)
    stem = os.path.basename(meta_path).removesuffix("_meta.json")
    warnings = []

    # ------------------------------------------------------------ joints
    joints = _read_csv_columns(os.path.join(folder, f"{stem}.csv"), ["device_ms", "host_mono"])
    device_s = joints["device_ms"] / 1000.0
    host_mono = joints["host_mono"]

    if np.any(np.diff(joints["device_ms"]) <= 0):
        warnings.append("device_ms is not strictly increasing (Arduino reset mid-episode?)")

    slope, intercept = np.polyfit(device_s, host_mono, 1)
    joint_times = slope * device_s + intercept
    residual_ms = (host_mono - joint_times) * 1000.0
    joint_span = device_s[-1] - device_s[0]

    joint_report = {
        "samples": int(len(device_s)),
        "rate_hz": round((len(device_s) - 1) / joint_span, 2) if joint_span > 0 else None,
        "device_interval": _interval_stats(device_s),
        "clock_drift_ppm": round((slope - 1.0) * 1e6, 1),
        "serial_arrival_jitter_ms": {
            "std": round(float(residual_ms.std()), 2),
            "max_abs": round(float(np.abs(residual_ms).max()), 2),
        },
    }
    if joint_report["device_interval"]["gaps"]:
        warnings.append(f"{joint_report['device_interval']['gaps']} gaps in the joint stream")

    # ------------------------------------------------------------ camera
    frames = _read_csv_columns(os.path.join(folder, f"{stem}_frames.csv"), ["host_mono"])
    frame_times = frames["host_mono"]
    frame_span = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0.0

    camera_meta = meta.get("camera", {})
    camera_report = {
        "frames": int(len(frame_times)),
        "fps_measured": round((len(frame_times) - 1) / frame_span, 2) if frame_span > 0 else None,
        "fps_reported_by_driver": camera_meta.get("fps_reported_by_driver"),
        "resolution": f"{camera_meta.get('width')}x{camera_meta.get('height')}",
        "interval": _interval_stats(frame_times),
        "dropped_writer_behind": camera_meta.get("frames_dropped_writer_behind", 0),
    }
    if camera_report["interval"]["gaps"]:
        warnings.append(f"{camera_report['interval']['gaps']} gaps in the camera stream "
                        "(frames the camera/driver never delivered)")
    if camera_report["dropped_writer_behind"]:
        warnings.append(f"{camera_report['dropped_writer_behind']} frames dropped by the disk writer")
    reported = camera_report["fps_reported_by_driver"]
    measured = camera_report["fps_measured"]
    if reported and measured and measured < 0.9 * reported:
        warnings.append(f"camera delivers {measured} fps but the driver reports {reported}")

    # --------------------------------------------------------- alignment
    idx = np.clip(np.searchsorted(joint_times, frame_times), 1, len(joint_times) - 1)
    nearest = np.minimum(np.abs(frame_times - joint_times[idx - 1]),
                         np.abs(frame_times - joint_times[idx])) * 1000.0
    inside = (frame_times >= joint_times[0]) & (frame_times <= joint_times[-1])

    alignment_report = {
        "frames_inside_joint_span": int(inside.sum()),
        "frames_outside_joint_span": int((~inside).sum()),
        "nearest_joint_sample_ms": {
            "median": round(float(np.median(nearest)), 2),
            "max": round(float(nearest.max()), 2),
        },
        "camera_latency_corrected": meta.get("timing", {}).get("camera_latency_corrected", False),
    }
    if not alignment_report["camera_latency_corrected"]:
        warnings.append("camera latency not yet measured — frame/joint alignment has an "
                        "unknown constant offset")

    return {
        "episode": stem,
        "task": meta.get("task"),
        "duration_sec": meta.get("duration_sec"),
        "joints": joint_report,
        "camera": camera_report,
        "alignment": alignment_report,
        "warnings": warnings,
    }


def find_meta_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "episode_*_meta.json")))
    return [path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="An episode_XXXX_meta.json, or a folder of episodes")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args()

    metas = find_meta_files(args.path)
    if not metas:
        print(f"No episodes found at {args.path}")
        sys.exit(1)

    for meta_path in metas:
        report = analyze_episode(meta_path)
        if args.json:
            print(json.dumps(report, indent=2))
            continue
        j, c, a = report["joints"], report["camera"], report["alignment"]
        print(f"\n== {report['episode']}  ({report['duration_sec']}s, task: {report['task']!r})")
        print(f"  joints : {j['samples']} samples @ {j['rate_hz']} Hz, "
              f"interval median {j['device_interval']['median_ms']} ms / max {j['device_interval']['max_ms']} ms, "
              f"serial jitter std {j['serial_arrival_jitter_ms']['std']} ms")
        print(f"  camera : {c['frames']} frames @ {c['fps_measured']} fps measured "
              f"({c['fps_reported_by_driver']} reported, {c['resolution']}), "
              f"interval median {c['interval']['median_ms']} ms / p95 {c['interval']['p95_ms']} ms / "
              f"max {c['interval']['max_ms']} ms")
        print(f"  align  : nearest joint sample median {a['nearest_joint_sample_ms']['median']} ms / "
              f"max {a['nearest_joint_sample_ms']['max']} ms; "
              f"{a['frames_outside_joint_span']} frames outside the joint span")
        for w in report["warnings"]:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
