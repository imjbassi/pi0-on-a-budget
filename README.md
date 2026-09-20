# pi0-on-a-budget

Fine-tuning Physical Intelligence's [π0-FAST](https://github.com/Physical-Intelligence/openpi)
with LoRA on a consumer RTX 4070 (12 GB), using demonstrations from
[gello-lite](https://github.com/imjbassi/gello-lite) — a 4-DOF, SG90-servo,
potentiometer-leader teleoperation arm — then evaluating open-loop vs closed-loop.

Negative results are documented as results. Measured numbers live in [RESULTS.md](RESULTS.md).

## Status

| Phase | State |
|---|---|
| 0. Recorder: joints + camera | Built. Real ZV-1F camera tested (720p, 30 fps, no gaps). Joint stream tested with simulated serial only. |
| 1. Recorder → LeRobot converter | Built. Verified on synthetic episodes through openpi's own data loader (chunks match exactly). |
| 1b. WSL2 + openpi + GPU | Done. openpi `215abfb`, JAX sees the RTX 4070. |
| 2a. 12 GB memory test | **No valid measurement yet.** First attempt failed loading weights on the host (another GPU job running, weights on slow `/mnt/d`). See [RESULTS.md](RESULTS.md). |
| 2b. Fine-tuning on real episodes | Blocked: arm disassembled, 0 real episodes. |
| 3. Open-loop eval, closed-loop controller, analysis | Built; tested with fakes (dry-run controller, hold-policy). Not run on real data. |
| Arduino `policy_follower.ino` | Written, **not yet compiled or run** on the Nano. |

### Known risks, stated up front
- **Memory.** openpi documents LoRA fine-tuning as needing **> 22.5 GB**. Its stock
  π0-FAST LoRA config freezes only the language model; the ~400M-param image encoder
  still trains in float32 with AdamW state. This project also freezes the image
  encoder — a departure from openpi's recipe. The base checkpoint is 10.85 GB and WSL
  has 15 GB of RAM, so host memory is a second constraint.
- **Embodiment.** π0's pretraining normalization stats cover ALOHA, Franka, UR5e, ARX
  arms (6–7 DOF, industrial actuators). Nothing resembles a 4-DOF hobby-servo arm, so
  fresh norm stats are used and transfer may be weak.
- **No measured state.** SG90s have no position feedback. "State" is the commanded
  angle, not the arm's actual position — in training and on the real arm alike.
- **Hold baseline.** At 30 fps consecutive commands are nearly identical, so "don't
  move" already scores a low open-loop error. Every open-loop number is reported next
  to that baseline.

## Layout

```
recorder/            Windows: record episodes (serial + camera), timing report
gello_pi0/           shared conventions + everything that runs in openpi (WSL)
  conventions.py       degrees <-> model units, image preprocessing (one definition, both sides)
  episodes.py          load + resample episodes to a uniform 30 fps timeline, held-out split
  convert_to_lerobot.py
  openpi_configs.py    policy transforms + training configs (registered into openpi)
  run.py               runs openpi's train / compute_norm_stats / serve_policy with those configs
  loadback_check.py    verifies the dataset through openpi's own loader
  memory_probe.py      measures peak GPU memory for a few real training steps
  eval_open_loop.py    policy vs recorded actions on held-out episodes
  policy_client.py     Windows client for openpi's policy server
  analyze.py           open-loop vs closed-loop, overall and per condition
closedloop/          Windows: drive the real arm with the policy, score trials
arduino/             policy_follower.ino (teleop mode + serial command mode)
config/robot.json    gripper direction + limits — fill in on the rebuilt arm
tools/               synthetic episode generator
wsl/                 openpi setup + environment wrapper
```

Large files (model weights, datasets, checkpoints) go to `D:\pi0-on-a-budget-runs`;
the WSL virtual disk is on C:, which has little free space.

## Workflow

### 1. Record (Windows)
```
pip install -r recorder/requirements.txt
python recorder/record_episode.py --list-cameras
python recorder/record_episode.py --port COM3 --camera 0 --task "pick up the red block" --condition block_left --outdir D:/pi0-on-a-budget-runs/episodes
python recorder/sync_report.py D:/pi0-on-a-budget-runs/episodes
```
`--condition` labels the scene setup (e.g. block position). It stratifies the held-out
split and is how open-loop and closed-loop results get compared per condition.

### 2. Convert + check (WSL)
```
bash wsl/setup_openpi.sh                  # once
bash wsl/run.sh python -m gello_pi0.convert_to_lerobot --episodes /mnt/d/pi0-on-a-budget-runs/episodes --robot-config config/robot.json --name gello
bash wsl/run.sh python -m gello_pi0.run norm-stats --config-name gello_lora
bash wsl/run.sh python -m gello_pi0.loadback_check --config gello_lora
```
The converter refuses real data until `config/robot.json` has gripper angles and
`confirmed_on_hardware: true`, and refuses to mix synthetic and real episodes.

### 3. Train (WSL)
```
bash wsl/run.sh python -m gello_pi0.memory_probe --config gello_lora --batch-sizes 1 2 4
bash wsl/run.sh python -m gello_pi0.run train gello_lora --exp-name run1 --batch-size <largest that fit>
```

### 4. Evaluate
Open-loop (WSL), once per checkpoint:
```
bash wsl/run.sh python -m gello_pi0.eval_open_loop --config gello_lora --checkpoint /mnt/d/pi0-on-a-budget-runs/checkpoints/gello_lora/run1/2999 --conversion-report /mnt/d/pi0-on-a-budget-runs/lerobot/local/gello_conversion.json
```
Closed-loop: flash `arduino/policy_follower`, start the server in WSL, run trials on Windows:
```
bash wsl/run.sh python -m gello_pi0.run serve policy:checkpoint --policy.config gello_lora --policy.dir /mnt/d/pi0-on-a-budget-runs/checkpoints/gello_lora/run1/2999
python closedloop/closed_loop.py --dashboard --port COM3 --camera 0 --robot-config config/robot.json --task "pick up the red block" --condition block_left --checkpoint-label run1_2999 --trials 10 --outdir D:/pi0-on-a-budget-runs/closed_loop
```
Try it with no hardware first: `python closedloop/closed_loop.py --dry-run --robot-config config/robot_fake.json --task t --condition c --checkpoint-label dry --trials 1`.

### Live dashboard

Add `--dashboard` to a closed-loop run. It opens `http://127.0.0.1:8765/` and
shows the live camera, policy action chunk, commanded joint values, inference
latency, frame/serial age, watchdog count, trial state, and timeline. Pause holds
the current command; Success and Failure score the trial on macOS or Windows;
STOP / HOLD aborts the trial and holds the pose. This is a
software stop, not an emergency stop: the physical power switch remains the
emergency control. Use
`--no-dashboard-browser` to serve it without opening a browser automatically.

The dashboard deliberately labels the joint display **firmware → command**.
SG90 servos have no encoder feedback, so neither value is a measured physical
joint angle. No visual joint tracking is required. It also does not invent a
natural-language chain of thought: π0-FAST returns an action chunk, and that real
chunk is what the policy panel reports.

To inspect a real camera and Arduino on macOS without loading a policy or
sending motion commands, use monitor-only mode (servo power may remain
disconnected):
```
python closedloop/closed_loop.py --dashboard --monitor-only --backend any \
  --port /dev/cu.usbserial-110 --camera 0 --robot-config config/robot.json \
  --task monitor --condition bench --checkpoint-label monitor
```
The Arduino must be running `policy_follower.ino`. Monitor-only mode permits an
unconfirmed robot config because it only reads telemetry. The dashboard's
STOP / HOLD button is the sole exception: pressing it sends the firmware's hold
command and exits the monitor.

Analysis:
```
python -m gello_pi0.analyze --open-loop run1_2999=<.../2999/open_loop/summary.json> --open-loop run1_1000=<...> --closed-loop D:/pi0-on-a-budget-runs/closed_loop
```

### Tests
`python -m pytest tests` (Windows, no hardware). `loadback_check` is the WSL-side test.

## Sony ZV-1F camera
Measured on this PC (USB streaming, DirectShow), 10 s runs:

| Requested | Delivered | fps measured | Frame interval median / p95 / max | Gaps |
|---|---|---|---|---|
| 1280×720 | 1280×720 | 30.01 | 33.3 / 42.2 / 47.0 ms | 0 |
| 1920×1080 | 1280×720 (request ignored) | 29.99 | 33.4 / 42.9 / 45.6 ms | 0 |

OpenCV lists the ZV-1F as index 0; NVIDIA Broadcast virtual cameras are 1 and 2.
Indexes can shift, so check `--list-cameras` each session.

- Fixed ultra-wide lens: mount it close enough that arm and objects fill the frame
  (training downsizes to 224×224, letterboxed).
- Manual focus, manual exposure/ISO, fixed white balance, SteadyShot off.
- Disable auto power-off. Mount rigidly; don't move it between episodes or trials.

## Frame/joint timing
Frames are paired with joint commands by **arrival time** on the PC. The camera's
own delay is not subtracted — deliberately: `closed_loop.py` pairs the newest frame
with the current command the same way, so training matches deployment.
