#!/usr/bin/env python3
"""
closed_loop.py — run the fine-tuned policy on the real arm and score each trial.

Runs on native Windows. The policy runs in WSL behind openpi's policy server:
    (WSL)      bash wsl/run.sh python -m gello_pi0.run serve policy:checkpoint \
                   --policy.config gello_lora --policy.dir <checkpoint>/<step>
    (Windows)  python closedloop/closed_loop.py --port COM3 --camera 0 \
                   --robot-config config/robot.json --task "pick up the red block" \
                   --condition block_left --checkpoint-label run1_step2999 --trials 10

Arduino must run arduino/policy_follower (not teleop.ino).

Per trial:
  1. Arm is in TELEOP mode: use the leader to put it in the start pose, reset the scene.
  2. ENTER starts the trial. The controller reads the follower's current angles and
     switches to POLICY mode from exactly that pose (no jump).
  3. Loop at --fps: when the action queue is empty, send (camera frame, commanded
     state, prompt) to the policy, queue the first --replan-steps actions of the chunk,
     then send one action per tick. The arm holds still while the policy thinks.
  4. Keys during a trial:  s = success   f = failure   SPACE = abort (e-stop: hold pose)
     A trial also ends at --max-trial-s (scored as timeout = failure).
  5. Outcome + optional failure note are saved; the arm returns to TELEOP mode.

Safety, layered:
  - PC side: model outputs clamped to joint limits, per-tick change capped (--max-step-deg)
  - Arduino side: limits, 6 deg / 20 ms rate cap, 500 ms watchdog hold
  - Physical: keep a hand near the power switch. The software cannot stop a brownout
    or a jammed gripper.

Per trial, in <outdir>/<checkpoint_label>/trial_NNNN/:
    trial.json       task, condition, outcome, timing, settings
    ticks.csv        every control tick: target, sent command, follower echo
    inferences.jsonl every policy call: latency, state, predicted chunk (degrees)
    frames/          the 224x224 image the policy saw at each inference
"""

import argparse
import csv
import json
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "recorder"))

from gello_pi0 import conventions  # noqa: E402
import camera as cam  # noqa: E402


class Follower:
    """Talks to policy_follower.ino. Tracks the latest echoed angles and mode."""

    def __init__(self, ser):
        self.ser = ser
        self.lock = threading.Lock()
        self.latest_deg = None
        self.latest_mono = None
        self.mode = None
        self.watchdog_events = 0
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            if line.startswith("#MODE,"):
                self.mode = line.split(",", 1)[1]
                continue
            if line == "#WATCHDOG":
                self.watchdog_events += 1
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            try:
                angles = [int(p) for p in parts[1:]]
            except ValueError:
                continue
            with self.lock:
                self.latest_deg = np.array(angles, dtype=float)
                self.latest_mono = time.perf_counter()

    def angles(self):
        with self.lock:
            return None if self.latest_deg is None else self.latest_deg.copy()

    def command(self, deg):
        d = np.rint(deg).astype(int)
        self.ser.write(f"C,{d[0]},{d[1]},{d[2]},{d[3]}\n".encode())
        return d

    def teleop(self):
        self.ser.write(b"T\n")

    def hold(self):
        self.ser.write(b"H\n")

    def close(self):
        self.running = False
        self.ser.close()


class FakeFollowerSerial:
    """Simulates policy_follower.ino for dry runs: echoes rate-limited targets at 50 Hz."""

    def __init__(self, start_deg=(90, 120, 60, 20)):
        self.current = np.array(start_deg, dtype=float)
        self.target = self.current.copy()
        self.mode = "TELEOP"
        self.pending = [b"#MODE,TELEOP\n"]
        self.start = time.perf_counter()
        self.tick = 0
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            for line in data.decode().strip().splitlines():
                if line.startswith("C,"):
                    if self.mode != "POLICY":
                        self.mode = "POLICY"
                        self.pending.append(b"#MODE,POLICY\n")
                    self.target = np.clip(np.array([float(x) for x in line.split(",")[1:]]),
                                          [0, 15, 0, 20], [180, 165, 180, 160])
                elif line == "T":
                    self.mode = "TELEOP"
                    self.pending.append(b"#MODE,TELEOP\n")
                elif line == "H" and self.mode != "POLICY":
                    self.mode = "POLICY"
                    self.target = self.current.copy()
                    self.pending.append(b"#MODE,POLICY\n")

    def readline(self):
        with self.lock:
            if self.pending:
                return self.pending.pop(0)
        due = self.start + self.tick * 0.02
        now = time.perf_counter()
        if now < due:
            time.sleep(due - now)
        self.tick += 1
        with self.lock:
            if self.mode == "POLICY":
                self.current += np.clip(self.target - self.current, -6, 6)
            angles = ",".join(str(int(round(a))) for a in self.current)
        return f"{int(self.tick * 20)},{angles}\n".encode()

    def close(self):
        pass


class HoldPolicy:
    """Stand-in policy for dry runs: always predicts 'stay where you are' in model units."""

    def __init__(self, horizon=15):
        self.horizon = horizon
        self.metadata = {"fake": True}

    def infer(self, obs):
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        return {"actions": np.repeat(state[None], self.horizon, axis=0)}

    def close(self):
        pass


def key_pressed():
    """Non-blocking single key read (Windows). Returns a lowercase char or None."""
    try:
        import msvcrt
    except ImportError:
        return None
    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        return ch.lower()
    return None


def monitor_hardware(args, follower, camera, dashboard):
    """Show real camera and firmware telemetry without sending motion commands."""
    settings = camera.source.actual_settings()
    dashboard.update(
        status="MONITOR", monitor_only=True, mode=follower.mode or "UNKNOWN",
        task="hardware monitor — policy commands disabled",
        condition=args.condition, trial=None, elapsed_s=0.0, max_trial_s=0.0,
        camera_fps=settings.get("fps_reported_by_driver", 0.0),
        control_hz=0.0, replan_steps=0, queue_remaining=0,
        latency_ms=None, inferences=0, events=[])
    print("[OK] monitor-only: reading camera + Arduino; no motion commands will be sent")
    print("     Ctrl+C exits. Dashboard STOP / HOLD sends one hold command, then exits.")
    started = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            latest = camera.latest
            if latest is not None:
                image, _ = latest
                dashboard.set_frame(image)
            angles = follower.angles()
            serial_age = None if follower.latest_mono is None else round((now - follower.latest_mono) * 1000)
            frame_age = None if camera.last_frame_mono is None else round((now - camera.last_frame_mono) * 1000)
            dashboard.update(
                elapsed_s=round(now - started, 3), frame=camera.frames_seen,
                frame_age_ms=frame_age, serial_age_ms=serial_age,
                firmware_deg=None if angles is None else angles.tolist(),
                commanded_deg=None, target_deg=None,
                watchdog_events=follower.watchdog_events,
                mode=follower.mode or "UNKNOWN")
            if "e_stop" in dashboard.controls():
                follower.hold()
                dashboard.update(status="STOPPED", mode="POLICY")
                return
            time.sleep(0.1)
    except KeyboardInterrupt:
        dashboard.update(status="STOPPED")
        print("\n[OK] monitor stopped")


def run_shadow_policy(args, follower, camera, policy, robot, dashboard):
    """Run real remote inference while leaving all model actions unexecuted."""
    settings = camera.source.actual_settings()
    dashboard.update(
        status="SHADOW", shadow_policy=True, monitor_only=False,
        mode=follower.mode or "TELEOP", task=args.task,
        condition=args.condition, trial=None, elapsed_s=0.0,
        max_trial_s=args.max_trial_s,
        camera_fps=settings.get("fps_reported_by_driver", 0.0),
        control_hz=0.0, replan_steps=args.replan_steps,
        queue_remaining=0, latency_ms=None, inferences=0,
        action_chunk_deg=[], events=[])
    print("[OK] RTX policy shadow mode: predictions are displayed but NEVER sent to the Arduino")
    print("     Move with TELEOP/pots. Ctrl+C exits; STOP / HOLD sends one hold command.")
    started = time.perf_counter()
    inference_count = 0
    paused = False
    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            latest = camera.latest
            angles = follower.angles()
            if latest is not None:
                image, frame_mono = latest
                dashboard.set_frame(image)
            else:
                frame_mono = None

            for control in dashboard.controls():
                if control == "e_stop":
                    follower.hold()
                    dashboard.update(status="STOPPED", mode="POLICY")
                    return
                if control in ("success", "failure"):
                    dashboard.update(status=control.upper(), outcome=control)
                    return
                if control == "toggle_pause":
                    paused = not paused
                    dashboard.update(paused=paused)

            serial_age = None if follower.latest_mono is None else round((now - follower.latest_mono) * 1000)
            frame_age = None if camera.last_frame_mono is None else round((now - camera.last_frame_mono) * 1000)
            dashboard.update(
                elapsed_s=round(elapsed, 3), frame=camera.frames_seen,
                frame_age_ms=frame_age, serial_age_ms=serial_age,
                firmware_deg=None if angles is None else angles.tolist(),
                commanded_deg=None, watchdog_events=follower.watchdog_events,
                mode=follower.mode or "TELEOP")

            if not paused and latest is not None and angles is not None:
                image, frame_mono = latest
                model_image = conventions.preprocess_image(image, bgr=True)
                obs = {
                    "observation/image": model_image,
                    "observation/state": robot.deg_to_model(angles),
                    "prompt": args.task,
                }
                t0 = time.perf_counter()
                chunk_model = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
                latency = time.perf_counter() - t0
                chunk_deg = robot.model_to_deg(chunk_model, clamp=True)
                inference_count += 1
                dashboard.update(
                    latency_ms=round(latency * 1000),
                    frame_age_ms=round((t0 - frame_mono) * 1000),
                    target_deg=np.round(chunk_deg[0], 2).tolist(),
                    action_chunk_deg=np.round(chunk_deg, 2).tolist(),
                    inferences=inference_count)
                dashboard.add_event(f"RTX {inference_count}", elapsed)

            if elapsed >= args.max_trial_s:
                dashboard.update(status="COMPLETE", outcome="shadow_complete")
                return
            time.sleep(0.01)
    except KeyboardInterrupt:
        dashboard.update(status="STOPPED")
        print("\n[OK] shadow mode stopped")


def run_trial(args, index, follower, camera, policy, robot, out_root, keys=key_pressed, dashboard=None):
    trial_dir = os.path.join(out_root, f"trial_{index:04d}")
    frames_dir = os.path.join(trial_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    start_deg = follower.angles()
    if start_deg is None:
        raise RuntimeError("No angles from the follower — is policy_follower.ino running?")
    state_deg = start_deg.copy()
    queue = []
    ticks, inference_log = [], []
    outcome, note = None, ""
    period = 1.0 / args.fps
    t_start = time.perf_counter()
    paused = False
    pause_started = None
    paused_total = 0.0
    dashboard_frame_due = 0.0
    next_tick = t_start
    follower.hold()                                  # enter POLICY mode at the current pose
    if dashboard:
        settings = camera.source.actual_settings()
        dashboard.update(
            status="RUNNING", mode="POLICY", task=args.task,
            condition=getattr(args, "condition", ""), trial=index,
            elapsed_s=0.0, max_trial_s=args.max_trial_s,
            camera_fps=settings.get("fps_reported_by_driver", 0.0),
            control_hz=args.fps, replan_steps=args.replan_steps,
            firmware_deg=start_deg.tolist(), commanded_deg=start_deg.tolist(),
            target_deg=start_deg.tolist(), action_chunk_deg=[], queue_remaining=0,
            latency_ms=None, inferences=0, watchdog_events=follower.watchdog_events,
            paused=False, outcome=None, events=[])
        dashboard.add_event("start", 0.0)

    while True:
        now = time.perf_counter()
        elapsed = now - t_start - paused_total - ((now - pause_started) if paused else 0.0)
        if dashboard:
            for control in dashboard.controls():
                if control == "e_stop":
                    outcome = "aborted"
                    follower.hold()
                    dashboard.add_event("stop / hold", elapsed, "pink")
                elif control in ("success", "failure"):
                    outcome = control
                    follower.hold()
                    dashboard.add_event(control, elapsed, "pink" if control == "failure" else "blue")
                elif control == "toggle_pause":
                    paused = not paused
                    if paused:
                        pause_started = now
                        follower.hold()
                        dashboard.add_event("pause", elapsed, "orange")
                    else:
                        paused_total += now - pause_started
                        pause_started = None
                        next_tick = time.perf_counter()
                        dashboard.add_event("resume", elapsed)
                dashboard.update(paused=paused)
            if outcome:
                break

            latest = camera.latest
            if latest is not None and now >= dashboard_frame_due:
                image, frame_mono = latest
                dashboard.set_frame(image)
                dashboard_frame_due = now + 0.1
            echo_now = follower.angles()
            serial_age = None if follower.latest_mono is None else round((now - follower.latest_mono) * 1000)
            frame_age = None if camera.last_frame_mono is None else round((now - camera.last_frame_mono) * 1000)
            dashboard.update(
                elapsed_s=round(elapsed, 3), frame=camera.frames_seen,
                frame_age_ms=frame_age, serial_age_ms=serial_age,
                firmware_deg=None if echo_now is None else echo_now.tolist(),
                watchdog_events=follower.watchdog_events, mode=follower.mode or "POLICY")

        key = keys()
        if key in ("s", "f", " "):
            outcome = {"s": "success", "f": "failure", " ": "aborted"}[key]
            break
        if paused:
            time.sleep(0.05)
            continue
        if elapsed > args.max_trial_s:
            outcome = "timeout"
            break

        inferred = False
        if not queue:
            latest = camera.latest
            if latest is None:
                raise RuntimeError("camera stopped delivering frames")
            image, frame_mono = latest
            model_image = conventions.preprocess_image(image, bgr=True)
            obs = {
                "observation/image": model_image,
                "observation/state": robot.deg_to_model(state_deg),
                "prompt": args.task,
            }
            t0 = time.perf_counter()
            chunk_model = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
            latency = time.perf_counter() - t0
            chunk_deg = robot.model_to_deg(chunk_model, clamp=True)
            queue = list(chunk_deg[: args.replan_steps])
            n = len(inference_log)
            Image.fromarray(model_image).save(os.path.join(frames_dir, f"inference_{n:05d}.jpg"), quality=90)
            inference_log.append({
                "n": n, "t": round(elapsed, 4), "latency_s": round(latency, 4),
                "frame_age_s": round(t0 - frame_mono, 4),
                "state_deg": state_deg.tolist(), "chunk_deg": np.round(chunk_deg, 2).tolist(),
            })
            inferred = True
            if dashboard:
                dashboard.update(
                    latency_ms=round(latency * 1000),
                    action_chunk_deg=np.round(chunk_deg, 2).tolist(),
                    queue_remaining=len(queue), inferences=len(inference_log))
                dashboard.add_event(f"replan {len(inference_log)}", elapsed)
            next_tick = time.perf_counter()          # don't try to "catch up" after thinking

        target = queue.pop(0)
        step = np.clip(target - state_deg, -args.max_step_deg, args.max_step_deg)
        state_deg = np.clip(state_deg + step, robot.joint_min_deg, robot.joint_max_deg)
        sent = follower.command(state_deg)
        echo = follower.angles()
        ticks.append({
            "t": round(time.perf_counter() - t_start, 4), "inferred": int(inferred),
            **{f"target_{j}": round(float(v), 2) for j, v in zip(conventions.JOINTS, target)},
            **{f"sent_{j}": int(v) for j, v in zip(conventions.JOINTS, sent)},
            **{f"echo_{j}": (int(v) if echo is not None else "") for j, v in
               zip(conventions.JOINTS, echo if echo is not None else [None] * 4)},
        })
        if dashboard:
            dashboard.update(
                target_deg=np.round(target, 2).tolist(),
                commanded_deg=sent.astype(float).tolist(),
                firmware_deg=None if echo is None else echo.tolist(),
                queue_remaining=len(queue))

        next_tick += period
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

    follower.hold()
    now = time.perf_counter()
    duration = now - t_start - paused_total - ((now - pause_started) if paused else 0.0)
    if dashboard:
        dashboard.update(
            status=outcome.upper(), paused=False, outcome=outcome,
            elapsed_s=round(duration, 3), queue_remaining=0)
        dashboard.add_event(outcome, duration, "pink" if outcome in ("failure", "aborted", "timeout") else "blue")
    return trial_dir, outcome, ticks, inference_log, duration, start_deg


def save_trial(trial_dir, info, ticks, inference_log):
    with open(os.path.join(trial_dir, "trial.json"), "w") as f:
        json.dump(info, f, indent=2)
    if ticks:
        with open(os.path.join(trial_dir, "ticks.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ticks[0].keys()))
            w.writeheader()
            w.writerows(ticks)
    with open(os.path.join(trial_dir, "inferences.jsonl"), "w") as f:
        for row in inference_log:
            f.write(json.dumps(row) + "\n")


def next_trial_index(out_root):
    existing = [int(n.split("_")[1]) for n in os.listdir(out_root) if n.startswith("trial_") and n[6:].isdigit()]
    return max(existing) + 1 if existing else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--camera", type=int)
    p.add_argument("--backend", default="dshow")
    p.add_argument("--robot-config", required=True)
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--task", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--checkpoint-label", required=True, help="e.g. run1_step2999; groups trials for analysis")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--fps", type=float, default=30.0, help="Control rate; match the dataset fps")
    p.add_argument("--replan-steps", type=int, default=5, help="Actions executed per chunk before re-querying")
    p.add_argument("--max-step-deg", type=float, default=4.0, help="Max command change per tick")
    p.add_argument("--max-trial-s", type=float, default=30.0)
    p.add_argument("--outdir", default="closed_loop_trials")
    p.add_argument("--dry-run", action="store_true",
                   help="Fake follower + fake camera + hold policy; no hardware, no server")
    p.add_argument("--fake-follower", action="store_true", help="Fake follower only")
    p.add_argument("--fake-camera", action="store_true")
    p.add_argument("--fake-policy", action="store_true")
    p.add_argument("--auto-outcome", choices=["success", "failure"],
                   help="Testing only: end each trial after --max-trial-s with this outcome, no keyboard")
    p.add_argument("--dashboard", action="store_true",
                   help="Serve the live browser dashboard with trial and motion controls")
    p.add_argument("--monitor-only", action="store_true",
                   help="Dashboard real camera + Arduino telemetry without loading or commanding a policy")
    p.add_argument("--shadow-policy", action="store_true",
                   help="Run real policy inference for the dashboard but never send its actions to the Arduino")
    p.add_argument("--dashboard-host", default="127.0.0.1")
    p.add_argument("--dashboard-port", type=int, default=8765)
    p.add_argument("--no-dashboard-browser", action="store_true",
                   help="Serve the dashboard without opening it automatically")
    args = p.parse_args()

    if args.dry_run:
        args.fake_follower = args.fake_camera = args.fake_policy = True

    robot = conventions.RobotConvention.load(args.robot_config)
    robot.require_gripper()
    if not robot.confirmed_on_hardware and not args.fake_follower and not args.monitor_only:
        p.error("robot config is not confirmed_on_hardware; refusing to drive the real arm")
    if args.monitor_only and not args.dashboard:
        p.error("--monitor-only requires --dashboard")
    if args.shadow_policy and not args.dashboard:
        p.error("--shadow-policy requires --dashboard")
    if args.monitor_only and args.shadow_policy:
        p.error("--monitor-only and --shadow-policy are mutually exclusive")
    if args.monitor_only and (args.fake_follower or args.fake_camera):
        p.error("--monitor-only expects a real --port and --camera; do not combine it with fake hardware")

    if args.fake_follower:
        ser = FakeFollowerSerial()
    else:
        if not args.port:
            p.error("--port is required")
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
    follower = Follower(ser)

    if args.fake_camera:
        source = cam.FakeSource(1280, 720, 30)
    else:
        if args.camera is None:
            p.error("--camera is required (see recorder/record_episode.py --list-cameras)")
        source = cam.OpenCVSource(args.camera, 1280, 720, 30, args.backend)
    camera = cam.CameraRecorder(source)
    camera.start()

    dashboard = None
    if args.dashboard:
        from dashboard import LiveDashboard
        dashboard = LiveDashboard(
            args.dashboard_host, args.dashboard_port,
            open_browser=not args.no_dashboard_browser).start()
        dashboard.update(
            status="READY", mode="TELEOP", task=args.task, condition=args.condition,
            checkpoint=args.checkpoint_label, max_trial_s=args.max_trial_s,
            camera_fps=source.actual_settings().get("fps_reported_by_driver", 0.0),
            control_hz=args.fps, replan_steps=args.replan_steps)

    if args.monitor_only:
        try:
            monitor_hardware(args, follower, camera, dashboard)
        finally:
            camera.stop()
            follower.close()
            dashboard.stop()
        return

    if args.fake_policy:
        policy = HoldPolicy()
    else:
        from gello_pi0.policy_client import PolicyClient
        policy = PolicyClient(args.server)
    print(f"[OK] policy: {policy.metadata}")

    if args.shadow_policy:
        try:
            run_shadow_policy(args, follower, camera, policy, robot, dashboard)
        finally:
            camera.stop()
            follower.close()
            policy.close()
            dashboard.stop()
        return

    out_root = os.path.join(args.outdir, args.checkpoint_label)
    os.makedirs(out_root, exist_ok=True)
    start_index = next_trial_index(out_root)

    try:
        for n in range(args.trials):
            index = start_index + n
            follower.teleop()
            if args.auto_outcome:
                keys = (lambda: None)
            else:
                input(f"\n[trial {index}] TELEOP: set start pose + scene ({args.condition}), ENTER to start ")
                keys = key_pressed
            print("  running... s = success, f = failure, SPACE = abort")
            trial_dir, outcome, ticks, inference_log, duration, start_deg = run_trial(
                args, index, follower, camera, policy, robot, out_root, keys, dashboard)
            if outcome == "timeout" and args.auto_outcome:
                outcome = args.auto_outcome
            note = "" if args.auto_outcome else input(f"  outcome={outcome}. Failure/observation note (ENTER to skip): ")
            latencies = [r["latency_s"] for r in inference_log]
            info = {
                "trial": index, "checkpoint_label": args.checkpoint_label, "task": args.task,
                "condition": args.condition, "outcome": outcome,
                "success": outcome == "success", "note": note,
                "duration_s": round(duration, 2), "start_deg": start_deg.tolist(),
                "inferences": len(inference_log), "ticks": len(ticks),
                "median_inference_s": float(np.median(latencies)) if latencies else None,
                "watchdog_events_total": follower.watchdog_events,
                "settings": {k: v for k, v in vars(args).items() if k not in ("port",)},
                "robot_convention_confirmed": robot.confirmed_on_hardware,
                "dry_run": bool(args.fake_follower or args.fake_camera or args.fake_policy),
            }
            save_trial(trial_dir, info, ticks, inference_log)
            print(f"  saved {trial_dir}: {outcome}, {len(inference_log)} inferences, "
                  f"median latency {info['median_inference_s']}")
    finally:
        follower.teleop()
        time.sleep(0.1)
        camera.stop()
        follower.close()
        policy.close()
        if dashboard:
            dashboard.stop()


if __name__ == "__main__":
    main()
