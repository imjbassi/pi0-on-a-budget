"""Recorder tests — run entirely on fake serial + fake camera, no hardware."""

import csv
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recorder"))

import camera as cam  # noqa: E402
import record_episode as rec  # noqa: E402
import sync_report  # noqa: E402
from fake_serial import FakeSerial  # noqa: E402


# --------------------------------------------------------------- parsing

@pytest.mark.parametrize("line, expected", [
    ("1234,90,45,120,20", (1234, [90, 45, 120, 20])),
    ("millis,base,shoulder,elbow,gripper", None),
    ("", None),
    ("1234,90,45", None),                 # partial line
    ("12a4,90,45,120,20", None),          # corrupted
    ("1234,90,45,120,20,7", None),        # too many fields
])
def test_parse_line(line, expected):
    assert rec.parse_line(line) == expected


def test_fake_serial_matches_teleop_format():
    ser = FakeSerial(rate_hz=500)
    assert ser.readline().decode().strip() == "millis,base,shoulder,elbow,gripper"
    parsed = [rec.parse_line(ser.readline().decode().strip()) for _ in range(20)]
    assert all(p is not None for p in parsed)
    millis = [p[0] for p in parsed]
    assert millis == sorted(millis) and len(set(millis)) == 20
    for _, angles in parsed:
        for a, lo, hi in zip(angles, [0, 15, 0, 20], [180, 165, 180, 160]):
            assert lo <= a <= hi


# ------------------------------------------------------------- numbering

def test_episode_numbering_continues(tmp_path):
    assert rec.next_episode_index(tmp_path) == 0
    (tmp_path / "episode_0003_meta.json").write_text("{}")
    (tmp_path / "episode_0005_frames").mkdir()     # orphan from a crashed session
    assert rec.next_episode_index(tmp_path) == 6


# ------------------------------------------------------------ end to end

def make_recorder(outdir, fps=30.0):
    camera = cam.CameraRecorder(cam.FakeSource(160, 90, fps))
    return rec.EpisodeRecorder(FakeSerial(), camera, str(outdir), "test task",
                               {"port": "FAKE", "fake": True})


def test_record_episode_end_to_end(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.start_streams()
    try:
        meta_path = recorder.record_for(1.5)
    finally:
        recorder.stop_streams()

    assert meta_path == str(tmp_path / "episode_0000_meta.json")
    meta = json.loads(open(meta_path).read())
    assert meta["schema_version"] == 2
    assert meta["task"] == "test task"
    assert meta["joint_units"] == "degrees"

    # Joint CSV: gello-lite's columns first, host_mono appended.
    with open(tmp_path / "episode_0000.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["host_time", "device_ms", "base", "shoulder",
                                     "elbow", "gripper", "host_mono"]
        joint_rows = list(reader)
    assert len(joint_rows) == meta["samples"]
    assert 50 <= len(joint_rows) <= 90          # ~75 at 50 Hz, loose for loaded machines

    # Frames: every CSV row points at a real, decodable JPEG; times increase.
    with open(tmp_path / "episode_0000_frames.csv", newline="") as f:
        frame_rows = list(csv.DictReader(f))
    assert len(frame_rows) == meta["camera"]["frames_saved"]
    assert 30 <= len(frame_rows) <= 50          # ~45 at 30 fps
    times = [float(r["host_mono"]) for r in frame_rows]
    assert times == sorted(times)
    assert [int(r["frame_index"]) for r in frame_rows] == list(range(len(frame_rows)))
    for r in frame_rows:
        image = cv2.imread(str(tmp_path / "episode_0000_frames" / r["file"]))
        assert image is not None and image.shape == (90, 160, 3)
    assert len(os.listdir(tmp_path / "episode_0000_frames")) == len(frame_rows)

    # Both streams should sit inside the recording window.
    start, end = meta["timing"]["started_mono"], meta["timing"]["ended_mono"]
    assert all(start <= t <= end for t in times)
    assert all(start <= float(r["host_mono"]) <= end for r in joint_rows)

    report = sync_report.analyze_episode(meta_path)
    assert report["joints"]["rate_hz"] == pytest.approx(50, rel=0.05)
    assert report["camera"]["fps_measured"] == pytest.approx(30, rel=0.15)
    assert report["alignment"]["nearest_joint_sample_ms"]["max"] <= 15   # half a 20 ms tick + slack


def test_second_episode_does_not_touch_first(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.start_streams()
    try:
        recorder.record_for(0.5)
        first = sorted(os.listdir(tmp_path / "episode_0000_frames"))
        recorder.record_for(0.5)
    finally:
        recorder.stop_streams()
    assert sorted(os.listdir(tmp_path / "episode_0000_frames")) == first
    assert (tmp_path / "episode_0001_meta.json").exists()


def test_camera_drops_are_counted_not_hidden(tmp_path):
    """A writer that can't keep up must report drops rather than lose frames silently."""
    source = cam.FakeSource(160, 90, fps=200)
    camera = cam.CameraRecorder(source, max_queue=1)
    original_imwrite = cam.cv2.imwrite

    def slow_imwrite(*args, **kwargs):
        import time
        time.sleep(0.05)
        return original_imwrite(*args, **kwargs)

    cam.cv2.imwrite = slow_imwrite
    try:
        camera.start()
        camera.begin_episode(str(tmp_path / "frames"))
        import time
        time.sleep(0.5)
        stats = camera.end_episode()
        camera.stop()
    finally:
        cam.cv2.imwrite = original_imwrite

    assert stats["frames_dropped_writer_behind"] > 0
    assert stats["frames_saved"] == len(os.listdir(tmp_path / "frames"))


# ------------------------------------------------------------ sync report

def write_synthetic_episode(folder, frame_times, joint_hz=50, duration=2.0):
    folder.mkdir(exist_ok=True)
    n = int(duration * joint_hz)
    with open(folder / "episode_0000.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rec.CSV_FIELDS)
        w.writeheader()
        for i in range(n):
            t = i / joint_hz
            w.writerow({"host_time": 1e9 + t, "device_ms": int(t * 1000), "base": 90,
                        "shoulder": 90, "elbow": 90, "gripper": 20, "host_mono": 100.0 + t})
    cam.write_frames_csv(folder / "episode_0000_frames.csv", [
        {"frame_index": i, "host_time": 1e9 + t, "host_mono": 100.0 + t, "file": f"frame_{i:06d}.jpg"}
        for i, t in enumerate(frame_times)
    ])
    meta = {"task": "synthetic", "duration_sec": duration,
            "camera": {"fps_reported_by_driver": 30.0, "width": 1280, "height": 720,
                       "frames_dropped_writer_behind": 0},
            "timing": {"camera_latency_corrected": False}}
    (folder / "episode_0000_meta.json").write_text(json.dumps(meta))
    return str(folder / "episode_0000_meta.json")


def test_sync_report_clean_stream(tmp_path):
    frames = np.arange(0.01, 1.95, 1 / 30)
    report = sync_report.analyze_episode(write_synthetic_episode(tmp_path, frames))
    assert report["camera"]["interval"]["gaps"] == 0
    assert report["camera"]["fps_measured"] == pytest.approx(30, rel=0.01)
    assert report["joints"]["rate_hz"] == pytest.approx(50, rel=0.01)
    assert not any("gaps" in w for w in report["warnings"])


def test_sync_report_detects_missing_frames_and_slow_camera(tmp_path):
    frames = np.arange(0.01, 1.95, 1 / 30)
    frames = np.delete(frames, [10, 11, 12, 30])            # two gaps
    frames = frames[::2]                                    # and only ~15 fps actually delivered
    report = sync_report.analyze_episode(write_synthetic_episode(tmp_path, frames))
    assert report["camera"]["interval"]["gaps"] >= 1
    assert any("driver reports" in w for w in report["warnings"])
