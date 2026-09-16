#!/usr/bin/env python3
"""
record_episode.py — capture teleoperation episodes: joint stream + camera.

Extends gello-lite's record_episode.py (github.com/imjbassi/gello-lite) with
synchronized webcam capture, for building VLA fine-tuning data.

Per episode, in --outdir:
    episode_0000.csv          host_time, device_ms, base, shoulder, elbow, gripper, host_mono
    episode_0000_frames.csv   frame_index, host_time, host_mono, file
    episode_0000_frames/      frame_000000.jpg, ...
    episode_0000_meta.json    task, timing, camera + serial details

The first six CSV columns are identical to gello-lite's; host_mono is appended.
host_mono (time.perf_counter) is the clock to align joints and frames on —
host_time (time.time) is kept for wall-clock reference only.

Joint values are the smoothed leader angles teleop.ino *commands* to the
servos, in degrees. SG90s give no position feedback, so there is no measured
follower state.

Episode numbering continues from whatever is already in --outdir, so a second
recording session never overwrites the first.

Usage:
    python record_episode.py --list-cameras
    python record_episode.py --port COM3 --camera 0 --task "pick up the block"
    python record_episode.py --fake-serial --fake-camera --duration 5    # no hardware

Controls (interactive mode):
    ENTER   start / stop an episode
    q       quit
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import threading
import time

import camera as cam
from fake_serial import FakeSerial

JOINTS = ["base", "shoulder", "elbow", "gripper"]
CSV_FIELDS = ["host_time", "device_ms"] + JOINTS + ["host_mono"]
SCHEMA_VERSION = 2


def parse_line(line):
    """Parse one teleop.ino line -> (device_ms, [4 angles]), or None if invalid."""
    if not line or line.startswith("millis"):
        return None
    parts = line.split(",")
    if len(parts) != 1 + len(JOINTS):
        return None                     # partial line, or noise on connect
    try:
        return int(parts[0]), [int(p) for p in parts[1:]]
    except ValueError:
        return None


def next_episode_index(outdir):
    indices = [
        int(m.group(1))
        for path in glob.glob(os.path.join(outdir, "episode_*"))
        if (m := re.match(r"episode_(\d+)", os.path.basename(path)))
    ]
    return max(indices) + 1 if indices else 0


class EpisodeRecorder:
    def __init__(self, ser, camera, outdir, task, serial_info=None):
        self.ser = ser
        self.camera = camera
        self.outdir = outdir
        self.task = task
        self.serial_info = serial_info or {}

        self.recording = False
        self.running = True
        self.lock = threading.Lock()

        self.rows = []
        self.bad_lines = 0
        self.last_line_mono = None
        self.episode_start = None

        os.makedirs(outdir, exist_ok=True)
        self.episode_index = next_episode_index(outdir)

    # ---------------------------------------------------------- streams

    def start_streams(self):
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()
        self.camera.start()
        print(f"[OK] Camera streaming: {self.camera.source.actual_settings()}")

    def stop_streams(self):
        self.running = False
        self.camera.stop()
        self.ser.close()

    def read_loop(self):
        """Continuously drain the serial port. Runs in a background thread."""
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            host_mono = time.perf_counter()
            host_time = time.time()

            parsed = parse_line(line)
            if parsed is None:
                if line and not line.startswith("millis"):
                    self.bad_lines += 1
                continue
            self.last_line_mono = host_mono

            device_ms, angles = parsed
            with self.lock:
                if self.recording:
                    row = {"host_time": host_time, "device_ms": device_ms, "host_mono": host_mono}
                    row.update(zip(JOINTS, angles))
                    self.rows.append(row)

    # --------------------------------------------------------- episodes

    def stem(self):
        return f"episode_{self.episode_index:04d}"

    def start_episode(self):
        frames_dir = os.path.join(self.outdir, f"{self.stem()}_frames")
        self.camera.begin_episode(frames_dir)
        with self.lock:
            self.rows = []
            self.episode_start = (time.time(), time.perf_counter())
            self.recording = True
        print(f"\n  ● RECORDING {self.stem()}  (ENTER to stop)")

    def stop_episode(self):
        with self.lock:
            self.recording = False
            rows = self.rows
        ended_mono = time.perf_counter()
        cam_stats = self.camera.end_episode()

        started_at, started_mono = self.episode_start
        duration = ended_mono - started_mono
        stem = self.stem()
        frames_dir = os.path.join(self.outdir, f"{stem}_frames")

        if not rows or cam_stats["frames_saved"] == 0:
            if not rows:
                print("  ! No joint samples captured — is the sketch running and streaming?")
            if cam_stats["frames_saved"] == 0:
                print("  ! No camera frames saved — is the camera still streaming?")
            print(f"    {stem} discarded.")
            shutil.rmtree(frames_dir, ignore_errors=True)
            return None

        csv_path = os.path.join(self.outdir, f"{stem}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        frames_csv = os.path.join(self.outdir, f"{stem}_frames.csv")
        cam.write_frames_csv(frames_csv, cam_stats["rows"])

        n_frames = cam_stats["frames_saved"]
        meta = {
            "schema_version": SCHEMA_VERSION,
            "episode": self.episode_index,
            "task": self.task,
            "started_at": started_at,
            "duration_sec": round(duration, 3),
            "samples": len(rows),
            "sample_rate_hz": round(len(rows) / duration, 1) if duration > 0 else 0,
            "joints": JOINTS,
            "joint_units": "degrees",
            "joint_semantics": "smoothed leader angle commanded to the follower servo "
                               "(SG90s have no position feedback)",
            "timing": {
                "sync_clock": "host_mono = time.perf_counter() on the recording PC",
                "started_mono": started_mono,
                "ended_mono": ended_mono,
                "camera_latency_corrected": False,
            },
            "camera": {
                **self.camera.source.actual_settings(),
                "frames_saved": n_frames,
                "frames_dropped_writer_behind": cam_stats["frames_dropped_writer_behind"],
                "write_failures": cam_stats["write_failures"],
                "fps_measured": round(n_frames / duration, 2) if duration > 0 else 0,
                "jpeg_quality": self.camera.jpeg_quality,
                "frames_dir": os.path.basename(frames_dir),
                "frames_csv": os.path.basename(frames_csv),
            },
            "serial": {**self.serial_info, "bad_lines_this_session": self.bad_lines},
        }
        meta_path = os.path.join(self.outdir, f"{stem}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ○ Saved {stem}: {len(rows)} joint samples "
              f"({meta['sample_rate_hz']} Hz), {n_frames} frames "
              f"({meta['camera']['fps_measured']} fps), {duration:.1f}s")
        if cam_stats["frames_dropped_writer_behind"]:
            print(f"  ! {cam_stats['frames_dropped_writer_behind']} frames dropped "
                  "(disk writer fell behind)")

        self.episode_index += 1
        return meta_path

    def check_streams_alive(self, max_silence=1.0):
        now = time.perf_counter()
        problems = []
        if self.last_line_mono is None or now - self.last_line_mono > max_silence:
            problems.append("no joint data from serial")
        if self.camera.last_frame_mono is None or now - self.camera.last_frame_mono > max_silence:
            problems.append("no frames from camera")
        return problems

    # ------------------------------------------------------------- modes

    def record_for(self, seconds):
        """Non-interactive: record exactly one episode of the given length."""
        time.sleep(0.5)                 # let the serial stream settle
        start_problems = self.check_streams_alive()
        if start_problems:
            print(f"  ! Streams not healthy before recording: {', '.join(start_problems)}")
        self.start_episode()
        time.sleep(seconds)
        return self.stop_episode()

    def run_interactive(self):
        print("\n" + "=" * 58)
        print(f"  Task: {self.task}")
        print(f"  Output: {self.outdir}/   (next: {self.stem()})")
        print("=" * 58)
        print("  ENTER = start/stop episode      q + ENTER = quit\n")

        try:
            while True:
                cmd = input().strip().lower()
                if cmd == "q":
                    break
                if self.recording:
                    self.stop_episode()
                else:
                    problems = self.check_streams_alive()
                    if problems:
                        print(f"  ! Not starting: {', '.join(problems)}")
                        continue
                    self.start_episode()
        except KeyboardInterrupt:
            pass
        finally:
            if self.recording:
                self.stop_episode()


def open_serial(args):
    if args.fake_serial:
        return FakeSerial(), {"port": "FAKE", "baud": args.baud, "fake": True}
    try:
        import serial
    except ImportError:
        print("pyserial not installed. Run: pip install pyserial")
        sys.exit(1)
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"[FAIL] Could not open {args.port}: {e}")
        print("       Close the Arduino IDE Serial Monitor if it's open —")
        print("       only one program can hold a serial port at a time.")
        sys.exit(1)
    time.sleep(2)                       # Nano resets on serial connect
    ser.reset_input_buffer()
    print(f"[OK] Connected to {args.port} at {args.baud} baud")
    return ser, {"port": args.port, "baud": args.baud, "fake": False}


def open_camera(args):
    if args.fake_camera:
        source = cam.FakeSource(args.width, args.height, args.fps)
    else:
        try:
            source = cam.OpenCVSource(args.camera, args.width, args.height, args.fps, args.backend)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            sys.exit(1)
    return cam.CameraRecorder(source, jpeg_quality=args.jpeg_quality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Serial port (COM3, /dev/ttyUSB0, ...)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--outdir", default="episodes")
    parser.add_argument("--task", default="unspecified",
                        help="Task description; becomes the language prompt for training")

    parser.add_argument("--camera", type=int,
                        help="OpenCV camera index (required: index 0 is often a virtual camera, "
                             "e.g. NVIDIA Broadcast, not your real one)")
    parser.add_argument("--backend", choices=sorted(cam.BACKENDS), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--list-cameras", action="store_true")

    parser.add_argument("--fake-serial", action="store_true", help="Synthetic joint stream")
    parser.add_argument("--fake-camera", action="store_true", help="Synthetic frames")
    parser.add_argument("--duration", type=float,
                        help="Record one episode of this many seconds, then exit")
    args = parser.parse_args()

    if args.list_cameras:
        cams = cam.list_cameras(backend=args.backend)
        if not cams:
            print("No cameras found.")
        for c in cams:
            print(c)
        print("\nOpenCV cannot report device names. Unplug the camera and run this again:")
        print("the index that disappears is the real camera. Virtual cameras (NVIDIA")
        print("Broadcast, OBS, Imaging Edge Webcam) show up here too.")
        return

    if not args.port and not args.fake_serial:
        parser.error("--port is required unless --fake-serial is set")
    if args.camera is None and not args.fake_camera:
        parser.error("--camera is required unless --fake-camera is set (see --list-cameras)")

    ser, serial_info = open_serial(args)
    camera = open_camera(args)
    recorder = EpisodeRecorder(ser, camera, args.outdir, args.task, serial_info)
    try:
        recorder.start_streams()
    except RuntimeError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    try:
        if args.duration:
            recorder.record_for(args.duration)
        else:
            recorder.run_interactive()
    finally:
        recorder.stop_streams()
        print(f"\nDone. Episodes in {args.outdir}/")


if __name__ == "__main__":
    main()
