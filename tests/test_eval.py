"""Evaluation plumbing tests: analysis stats, policy wire protocol, closed-loop controller dry run."""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "recorder"))
sys.path.insert(0, os.path.join(ROOT, "closedloop"))

from gello_pi0 import analyze  # noqa: E402
from gello_pi0 import conventions  # noqa: E402


def test_spearman_known_values():
    assert analyze.spearman([1, 2, 3, 4], [10, 20, 30, 40])["rho"] == pytest.approx(1.0)
    assert analyze.spearman([1, 2, 3, 4], [4, 3, 2, 1])["rho"] == pytest.approx(-1.0)
    r = analyze.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert r["p"] == pytest.approx(2 / 120, abs=1e-4)   # exact: only 2 of 120 permutations as extreme
    assert analyze.spearman([1, 2], [1, 2])["rho"] is None
    assert analyze.spearman([1, 2, 3], [1, 1, 1])["rho"] is None


def test_wilson_interval():
    lo, hi = analyze.wilson(5, 10)
    assert lo < 0.5 < hi and lo == pytest.approx(0.237, abs=0.002)
    assert analyze.wilson(0, 0) == (None, None)


def test_policy_client_wire_format_round_trip():
    pytest.importorskip("msgpack")
    from gello_pi0 import policy_client
    obs = {"observation/image": np.zeros((224, 224, 3), np.uint8),
           "observation/state": np.array([0.1, -0.2, 0.3, 1.0], np.float32),
           "prompt": "pick up the red block"}
    back = policy_client.unpackb(policy_client.packb(obs))
    assert back["prompt"] == obs["prompt"]
    np.testing.assert_array_equal(back["observation/state"], obs["observation/state"])
    assert back["observation/image"].shape == (224, 224, 3) and back["observation/image"].dtype == np.uint8


def test_dashboard_state_and_frame_are_thread_safe():
    import dashboard

    state = dashboard.DashboardState()
    state.update(status="RUNNING", commanded_deg=[90, 80, 100, 40])
    state.add_event("start", 0.0)
    state.set_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    snap = state.snapshot()
    assert snap["status"] == "RUNNING"
    assert snap["commanded_deg"] == [90, 80, 100, 40]
    assert snap["events"] == [{"label": "start", "t": 0.0, "color": "blue"}]
    assert state.jpeg().startswith(b"\xff\xd8")


def test_closed_loop_follows_policy_and_respects_limits(tmp_path):
    import closed_loop
    import camera as cam
    from types import SimpleNamespace

    robot = conventions.RobotConvention(gripper_open_deg=20, gripper_closed_deg=160)

    class WildPolicy:
        """Asks for base far past its limit and gripper fully closed."""
        metadata = {}

        def infer(self, obs):
            return {"actions": np.tile(np.array([[4.0, 0.5, 0.5, 1.0]], np.float32), (15, 1))}

        def close(self):
            pass

    follower = closed_loop.Follower(closed_loop.FakeFollowerSerial(start_deg=(90, 120, 60, 20)))
    camera = cam.CameraRecorder(cam.FakeSource(320, 180, 30))
    camera.start()
    deadline = [None]

    def keys():
        import time
        if deadline[0] is None:
            deadline[0] = time.perf_counter() + 1.5
        return "s" if time.perf_counter() > deadline[0] else None

    args = SimpleNamespace(fps=30.0, replan_steps=5, max_step_deg=4.0, max_trial_s=10, task="t")
    import time
    time.sleep(0.2)
    try:
        trial_dir, outcome, ticks, inferences, duration, start = closed_loop.run_trial(
            args, 0, follower, camera, WildPolicy(), robot, str(tmp_path), keys)
    finally:
        camera.stop()
        follower.close()

    assert outcome == "success"
    sent_base = np.array([t["sent_base"] for t in ticks])
    sent_grip = np.array([t["sent_gripper"] for t in ticks])
    assert sent_base.max() <= 180                                  # clamped to limit
    assert np.abs(np.diff(sent_base)).max() <= 4 + 1                # rate limited (rounding)
    assert sent_grip[-1] > sent_grip[0]                             # moving toward closed
    assert len(inferences) >= 2 and len(os.listdir(os.path.join(trial_dir, "frames"))) == len(inferences)
    assert inferences[0]["state_deg"] == [90.0, 120.0, 60.0, 20.0]  # started from follower's pose
