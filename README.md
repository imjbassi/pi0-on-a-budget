# pi0-on-a-budget

Fine-tuning Physical Intelligence's [π0-FAST](https://github.com/Physical-Intelligence/openpi)
with LoRA on a consumer RTX 4070 (12 GB), using demonstrations from
[gello-lite](https://github.com/imjbassi/gello-lite) — a 4-DOF, SG90-servo,
potentiometer-leader teleoperation arm — then evaluating open-loop vs closed-loop.

Negative results are documented as results.

## Status

| Phase | State |
|---|---|
| 0. Recorder: joints + camera | **Built.** Real ZV-1F camera tested (720p, 30 fps, no gaps). Joint stream tested only with simulated serial; arm not yet tested. |
| 1. CSV → LeRobot converter | Not started |
| 1b. WSL2 GPU check + openpi load-back test | Not started |
| 2a. 12 GB memory test (few LoRA steps, image encoder frozen) | Not started — runs on fake episodes, **before** recording real data |
| 2b. Fine-tuning on real episodes | Blocked: arm disassembled, 0 episodes |
| 3. Open-loop / closed-loop evaluation | Not started |

### Known risks, stated up front
- **Memory.** openpi documents LoRA fine-tuning as needing **> 22.5 GB**. Its stock
  π0-FAST LoRA config freezes only the language model; the ~400M-param image encoder
  still trains in float32 with AdamW state (~6.4 GB). Plan: also freeze the image
  encoder and use a small batch — a change to openpi's recipe that has not been
  verified to fit in 12 GB.
- **Embodiment.** π0's pretraining normalization stats cover ALOHA, Franka, UR5e, ARX
  arms (6–7 DOF, radians, industrial actuators). Nothing resembles a 4-DOF hobby-servo
  arm, so fresh norm stats are required and transfer may be weak.
- **No measured state.** SG90s have no position feedback. The recorded joint values
  are the smoothed *commands* sent to the servos, so "state" means commanded
  angle, not the arm's actual position.

## Recorder (native Windows)

```
pip install -r recorder/requirements.txt
```

Test without any hardware:

```
python recorder/record_episode.py --fake-serial --fake-camera --duration 5 --outdir episodes_fake
python recorder/sync_report.py episodes_fake
python -m pytest tests
```

Real recording:

```
python recorder/record_episode.py --list-cameras
python recorder/record_episode.py --port COM3 --camera 0 --task "pick up the block" --outdir episodes
python recorder/sync_report.py episodes
```

Per episode it writes `episode_NNNN.csv` (gello-lite's columns + `host_mono`),
`episode_NNNN_frames.csv`, `episode_NNNN_frames/*.jpg`, and `episode_NNNN_meta.json`.
Episode numbers continue from what's already in `--outdir`.

`--camera` has no default on purpose: virtual cameras (NVIDIA Broadcast etc.)
show up as OpenCV indexes too, and OpenCV can't report device names. To find the
real one, run `--list-cameras` with the camera plugged in and again with it unplugged.

### Sony ZV-1F as the camera
Measured on this PC (USB streaming, DirectShow, joints simulated), 10 s runs:

| Requested | Delivered | fps measured | Frame interval median / p95 / max | Gaps |
|---|---|---|---|---|
| 1280×720 | 1280×720 | 30.01 | 33.3 / 42.2 / 47.0 ms | 0 |
| 1920×1080 | 1280×720 (request ignored) | 29.99 | 33.4 / 42.9 / 45.6 ms | 0 |

So the dataset camera stream is **720p at 30 fps**. OpenCV lists the ZV-1F as
index 0; NVIDIA Broadcast's virtual cameras are indexes 1 and 2 (640×480).
Indexes can shift when devices are added, so check `--list-cameras` each session.

- The ZV-1F has a fixed ultra-wide lens, so the arm will look small in frame.
  Put the camera close enough that the arm and objects fill most of the image,
  because training downsizes everything to 224×224.
- Set the camera to a fixed image before recording anything real: **manual focus,
  manual exposure/ISO, fixed white balance, SteadyShot off** (stabilization shifts the
  image frame to frame). Auto adjustments make the same scene look different across
  episodes.
- Disable auto power-off, and watch battery/heat in long sessions. If the camera
  stops streaming mid-session, the recorder refuses to start the next episode.
- **Mount it rigidly and don't move it between episodes.** Camera position becomes
  part of what the policy learns.

## Before recording real data (needs the rebuilt arm)
1. Run `pot_calibrate.ino` and paste the ranges into `teleop.ino`.
2. Record which gripper angle is closed (20° or 160°); the converter will read it
   from config.
3. Fix and lock the camera placement and settings.
4. **Measure camera latency.** Frame timestamps include exposure + USB + decode
   delay, which `sync_report.py` cannot see. Record a few sharp, visible motions
   (e.g. snap the gripper shut) and compare when the command changes vs when the
   image changes. The resulting constant offset gets applied at conversion.
5. Record 1–2 trial episodes, run `sync_report.py`, and push them through the
   converter before recording a full set.
