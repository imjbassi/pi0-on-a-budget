#!/usr/bin/env python3
"""
eval_open_loop.py — fine-tuned policy vs recorded expert actions on held-out episodes.

Runs in WSL (openpi env). For every `stride`-th timestep of each held-out
episode, feeds the recorded image + commanded state + task prompt to the
policy and compares its predicted action chunk against what the operator
actually commanded next.

Reported side by side with a HOLD baseline (predict "stay where you are" for
the whole chunk). At 30 fps consecutive commands are nearly identical, so a
low error alone proves little — the question is whether the policy beats hold,
and where.

Every row is tagged so errors can be broken down by situation, which is what
the open-loop vs closed-loop comparison needs:
    condition            episode's --condition label (e.g. block_left)
    gripper_transition   recorded gripper moves > 30% of its range within the chunk
    moving / static      recorded mean joint speed over the chunk above / below 15 deg/s
    near_limit           any recorded joint within 10 deg of its limit in the chunk

    cd ~/projects/openpi
    uv run python -m gello_pi0.eval_open_loop --config gello_lora \
        --checkpoint /mnt/d/pi0-on-a-budget-runs/checkpoints/gello_lora/run1/2999 \
        --conversion-report $HF_LEROBOT_HOME/local/gello_conversion.json
"""

import argparse
import csv
import json
import pathlib
import time

import numpy as np

from gello_pi0 import conventions
from gello_pi0 import episodes as eps
from gello_pi0.convert_to_lerobot import load_frame

MOVING_DEG_PER_S = 15.0
GRIPPER_TRANSITION = 0.30
NEAR_LIMIT_DEG = 10.0


def tag_chunk(target_deg, fps, robot):
    grip_model = robot.deg_to_model(target_deg)[:, 3]
    tags = []
    if len(target_deg) > 1:
        speed = np.abs(np.diff(target_deg[:, :3], axis=0)).mean() * fps
        tags.append("moving" if speed > MOVING_DEG_PER_S else "static")
    else:
        tags.append("static")
    if np.ptp(grip_model) > GRIPPER_TRANSITION:
        tags.append("gripper_transition")
    lo, hi = np.array(robot.joint_min_deg), np.array(robot.joint_max_deg)
    if np.any(target_deg - lo < NEAR_LIMIT_DEG) or np.any(hi - target_deg < NEAR_LIMIT_DEG):
        tags.append("near_limit")
    return tags


def chunk_errors(pred_deg, target_deg, pred_model, target_model):
    err = np.abs(pred_deg - target_deg)                       # (h, 4) degrees
    return {
        "err_deg_first": err[0].tolist(),                     # per joint, first action
        "err_deg_mean": float(err[:, :3].mean()),             # arm joints over chunk
        "err_gripper_deg_mean": float(err[:, 3].mean()),
        "mse_model_units": float(np.mean((pred_model - target_model) ** 2)),
    }


def summarize(rows, key_fn):
    groups = {}
    for r in rows:
        for key in key_fn(r):
            groups.setdefault(key, []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        out[key] = {
            "n": len(rs),
            "policy_mse": float(np.mean([r["policy_mse_model_units"] for r in rs])),
            "hold_mse": float(np.mean([r["hold_mse_model_units"] for r in rs])),
            "policy_err_deg": float(np.mean([r["policy_err_deg_mean"] for r in rs])),
            "hold_err_deg": float(np.mean([r["hold_err_deg_mean"] for r in rs])),
            "policy_beats_hold_frac": float(np.mean([r["policy_mse_model_units"] < r["hold_mse_model_units"]
                                                     for r in rs])),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conversion-report", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output folder (default: <checkpoint>/open_loop)")
    args = parser.parse_args()

    from gello_pi0 import openpi_configs
    openpi_configs.register()
    from openpi.policies import policy_config
    from openpi.training import config as _config

    report = json.loads(pathlib.Path(args.conversion_report).read_text())
    robot = conventions.RobotConvention(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in report["robot_convention"].items()
    })
    fps = report["fps"]
    latency_s = (report["camera_latency_ms"] or 0.0) / 1000.0
    heldout = report["split"]["heldout"]
    if not heldout:
        raise SystemExit("Conversion report has no held-out episodes.")

    train_config = _config.get_config(args.config)
    horizon = train_config.model.action_horizon
    policy = policy_config.create_trained_policy(train_config, args.checkpoint)

    out_dir = pathlib.Path(args.out or pathlib.Path(args.checkpoint) / "open_loop")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in heldout:
        raw = eps.load_episode(str(pathlib.Path(report["source_episodes"]) / f"{stem}_meta.json"))
        ep = eps.resample_episode(raw, fps, latency_s)
        state_model = robot.deg_to_model(ep.state_deg)
        action_model = robot.deg_to_model(ep.action_deg)
        steps = range(0, len(ep), args.stride)
        if args.max_steps_per_episode:
            steps = list(steps)[: args.max_steps_per_episode]
        for k in steps:
            h = min(horizon, len(ep) - k)
            obs = {
                "observation/image": load_frame(raw.frame_paths[ep.frame_index[k]]),
                "observation/state": state_model[k],
                "prompt": raw.task,
            }
            t0 = time.perf_counter()
            pred_model = np.asarray(policy.infer(obs)["actions"])[:h]
            latency = time.perf_counter() - t0
            pred_deg = robot.model_to_deg(pred_model, clamp=True)

            target_deg = ep.action_deg[k:k + h]
            target_model = action_model[k:k + h]
            hold_deg = np.repeat(ep.state_deg[k:k + 1], h, axis=0)
            hold_model = np.repeat(state_model[k:k + 1], h, axis=0)

            policy_e = chunk_errors(pred_deg, target_deg, pred_model, target_model)
            hold_e = chunk_errors(hold_deg, target_deg, hold_model, target_model)
            rows.append({
                "episode": stem, "condition": raw.condition, "step": k, "time_s": round(k / fps, 3),
                "horizon": h, "tags": "|".join(tag_chunk(target_deg, fps, robot)),
                "inference_s": round(latency, 4),
                **{f"policy_{key}": v for key, v in policy_e.items()},
                **{f"hold_{key}": v for key, v in hold_e.items()},
            })
        print(f"[open-loop] {stem}: {len([r for r in rows if r['episode'] == stem])} chunks", flush=True)

    with open(out_dir / "per_step.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "action_horizon": horizon,
        "stride": args.stride,
        "heldout_episodes": heldout,
        "synthetic_data": report.get("synthetic", False),
        "median_inference_s": float(np.median([r["inference_s"] for r in rows])),
        "overall": summarize(rows, lambda r: ["all"])["all"],
        "by_condition": summarize(rows, lambda r: [str(r["condition"])]),
        "by_tag": summarize(rows, lambda r: r["tags"].split("|")),
        "by_episode": summarize(rows, lambda r: [r["episode"]]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    o = summary["overall"]
    print(f"[open-loop] overall: policy MSE {o['policy_mse']:.5f} vs hold {o['hold_mse']:.5f}; "
          f"arm error {o['policy_err_deg']:.2f} deg vs hold {o['hold_err_deg']:.2f} deg; "
          f"beats hold on {100 * o['policy_beats_hold_frac']:.0f}% of chunks")
    print(f"[open-loop] wrote {out_dir}")


if __name__ == "__main__":
    main()
