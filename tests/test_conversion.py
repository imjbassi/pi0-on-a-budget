"""Conversion math tests: units, resampling, state/action shift, split. No LeRobot needed."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from gello_pi0 import conventions  # noqa: E402
from gello_pi0 import episodes as eps  # noqa: E402

FAKE_ROBOT = conventions.RobotConvention(gripper_open_deg=20, gripper_closed_deg=160)


# ------------------------------------------------------------- joints

def test_deg_to_model_units():
    out = FAKE_ROBOT.deg_to_model([[90, 180, 0, 20], [0, 90, 90, 160], [45, 90, 90, 90]])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [0, np.pi / 2, -np.pi / 2, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [-np.pi / 2, 0, 0, 1.0], atol=1e-6)
    assert out[2, 3] == pytest.approx(0.5)


def test_inverted_gripper_direction():
    robot = conventions.RobotConvention(gripper_open_deg=160, gripper_closed_deg=20)
    np.testing.assert_allclose(robot.deg_to_model([[90, 90, 90, 160], [90, 90, 90, 20]])[:, 3], [0, 1])


def test_model_to_deg_round_trip_and_clamp():
    deg = np.array([[10, 30, 170, 40], [175, 150, 5, 150]], dtype=float)
    np.testing.assert_allclose(FAKE_ROBOT.model_to_deg(FAKE_ROBOT.deg_to_model(deg)), deg, atol=1e-3)
    wild = FAKE_ROBOT.model_to_deg([[3.0, -3.0, 0.0, 2.0]])
    assert np.all(wild >= np.array(FAKE_ROBOT.joint_min_deg))
    assert np.all(wild <= np.array(FAKE_ROBOT.joint_max_deg))


def test_unconfirmed_gripper_refuses():
    with pytest.raises(ValueError, match="Gripper"):
        conventions.RobotConvention().deg_to_model([[90, 90, 90, 90]])


def test_repo_robot_config_matches_the_built_arm():
    """config/robot.json tracks the real arm: measured limits, gripper 40 closed / 180 open."""
    robot = conventions.RobotConvention.load(os.path.join(ROOT, "config", "robot.json"))
    robot.require_gripper()
    assert robot.joint_min_deg == (0, 60, 75, 40)
    assert robot.joint_max_deg == (180, 130, 120, 180)
    grip = robot.deg_to_model([[90, 90, 90, 40], [90, 90, 90, 180]])[:, 3]
    np.testing.assert_allclose(grip, [1.0, 0.0], atol=1e-6)   # 1.0 = closed, per openpi
    # Flipped to true only after the teleop check; the converter refuses real data until then.
    assert robot.confirmed_on_hardware is False


# ------------------------------------------------------------- images

def test_preprocess_image_letterbox_and_color_order():
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    bgr[..., 0] = 255                                   # pure blue in BGR
    out = conventions.preprocess_image(bgr, bgr=True)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    center = out[112, 112]
    assert center[2] == 255 and center[0] == 0          # blue lands in RGB channel 2
    assert out[0, 112].sum() == 0 and out[-1, 112].sum() == 0   # letterbox bars
    content_rows = np.where(out[:, 112, 2] > 0)[0]
    assert len(content_rows) == 126                     # 720 * 224/1280


# --------------------------------------------------------- resampling

def make_raw(joint_fn, joint_hz=50, fps=30, duration=2.0, frame_jitter=0.0, drop=(), seed=0):
    rng = np.random.default_rng(seed)
    jt = np.arange(0, duration, 1 / joint_hz)
    ft = np.arange(0.005, duration, 1 / fps) + rng.uniform(0, frame_jitter, size=len(np.arange(0.005, duration, 1 / fps)))
    ft = np.sort(np.delete(ft, list(drop)))
    joints = np.stack([joint_fn(t) for t in jt])
    return eps.RawEpisode("episode_0000", ".", {"task": "t", "condition": None},
                          jt, joints, ft, [f"f{i}.jpg" for i in range(len(ft))])


def test_resample_uniform_grid_and_action_is_next_state():
    raw = make_raw(lambda t: np.array([90 + 10 * t, 90, 90, 20]))
    ep = eps.resample_episode(raw, fps=30)
    np.testing.assert_allclose(np.diff(ep.times), 1 / 30, atol=1e-9)
    np.testing.assert_allclose(ep.action_deg[:-1], ep.state_deg[1:], atol=1e-9)
    np.testing.assert_allclose(ep.state_deg[:, 0], 90 + 10 * ep.times, atol=1e-6)   # linear interp exact
    assert ep.stats()["steps_in_camera_gaps"] == 0
    assert ep.times[0] >= raw.frame_times[0] and ep.times[-1] <= raw.joint_times[-1]


def test_resample_nearest_frame_and_gaps_are_reported():
    raw = make_raw(lambda t: np.array([90, 90, 90, 20]), drop=range(20, 26), frame_jitter=0.01)
    ep = eps.resample_episode(raw, fps=30)
    chosen = raw.frame_times[ep.frame_index]
    for k in range(len(ep)):
        assert abs(chosen[k] - ep.times[k]) == pytest.approx(np.min(np.abs(raw.frame_times - ep.times[k])))
    stats = ep.stats()
    assert stats["steps_in_camera_gaps"] > 0
    assert stats["frames_reused"] > 0


def test_camera_latency_shifts_frame_times():
    raw = make_raw(lambda t: np.array([90, 90, 90, 20]))
    ep0 = eps.resample_episode(raw, fps=30, camera_latency_s=0.0)
    ep1 = eps.resample_episode(raw, fps=30, camera_latency_s=0.1)
    # With 100 ms latency, frame i depicts time t_i - 0.1, so the same timeline
    # point maps to a frame ~3 indices later.
    k = 20
    t = ep0.times[k]
    k1 = int(np.argmin(np.abs(ep1.times - t)))
    assert ep1.frame_index[k1] - ep0.frame_index[k] in (2, 3, 4)


# --------------------------------------------------------------- split

def test_split_is_stratified_deterministic_and_disjoint():
    raws = []
    for i in range(10):
        cond = "left" if i < 6 else "right"
        raws.append(eps.RawEpisode(f"episode_{i:04d}", ".", {"task": "t", "condition": cond},
                                   None, None, None, []))
    raws.append(eps.RawEpisode("episode_0099", ".", {"task": "t", "condition": "solo"}, None, None, None, []))
    train, heldout = eps.choose_split(raws, 0.2, seed=1)
    assert set(train).isdisjoint(heldout)
    assert len(train) + len(heldout) == 11
    assert "episode_0099" in train                      # singleton condition stays in train
    conds = {r.stem: r.condition for r in raws}
    assert {conds[s] for s in heldout} == {"left", "right"}
    assert eps.choose_split(raws, 0.2, seed=1) == (train, heldout)


# ------------------------------------------------ fake episodes end to end

def test_fake_episodes_load_and_resample(tmp_path):
    out = tmp_path / "fake"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_fake_episodes.py"),
                    "--outdir", str(out), "--episodes", "3", "--duration", "2"], check=True)
    metas = eps.find_episodes(str(out))
    assert len(metas) == 3
    for m in metas:
        raw = eps.load_episode(m)
        ep = eps.resample_episode(raw, fps=30)
        assert 50 <= len(ep) <= 60
        assert all(os.path.exists(raw.frame_paths[i]) for i in set(ep.frame_index))
        model = FAKE_ROBOT.deg_to_model(ep.state_deg)
        assert np.all((model[:, 3] >= 0) & (model[:, 3] <= 1))

    # Overwrite guard: real episodes must never be deleted.
    meta = json.loads(open(metas[0]).read())
    meta.pop("synthetic")
    open(metas[0], "w").write(json.dumps(meta))
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_fake_episodes.py"),
                             "--outdir", str(out), "--overwrite"], capture_output=True, text=True)
    assert result.returncode != 0 and "real episodes" in result.stderr
    assert os.path.exists(metas[0])
