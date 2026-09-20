# Results log

What actually happened, including failures. Newest first.

## 2026-09-20 — Dashboard hardware path and RTX shadow setup

- The browser dashboard, real camera path, and Arduino telemetry are working.
- Deployment camera changed from the Sony ZV-1F to a Logitech Brio 101. The
  project requests 1280×720 at 30 fps; delivered resolution, frame pacing, and
  gaps have not yet been benchmarked for the Brio, so no Sony measurements are
  being reused as if they applied to the new camera.
- Shoulder and elbow servos are now MG90S; base and gripper remain SG90. These
  hobby servos report no measured position, so dashboard joint values remain
  firmware/command angles.
- A real base π0-FAST shadow-server attempt restored from the 10.85 GB checkpoint
  on `/mnt/d` and failed before serving with TensorStore `OS error 12: ENOMEM`.
  This was host/checkpoint-read memory exhaustion, not a CUDA out-of-memory
  result. At inspection time WSL was limited to roughly 15 GB RAM and 4 GB swap
  on a 32 GB Windows machine.
- The documented recovery uses a 24 GB WSL memory cap, 16 GB swap, and a
  WSL-ext4 copy of the base params at `~/pi0-cache/pi0_fast_base/params`.
- Moving the params fixed the read failure: Orbax restored 5.4 GiB at 101.8
  MiB/s in 54.81 seconds. The next attempt exposed a separate configuration
  error: `gello_fake_lora` expects `lora_a`/`lora_b` tensors that do not exist in
  the unmodified base checkpoint. Shadow serving therefore uses the dedicated
  non-LoRA `gello_fake_base` policy config. That configuration restored the same
  checkpoint in 46.25 seconds, loaded normalization stats from the shadow
  checkpoint, and successfully listened on `0.0.0.0:8000`. The verification
  server was then stopped deliberately so it would not continue occupying the GPU.

## 2026-09-19 — Arm rebuilt (EEZYbotARM), joint limits measured

Follower is now a 3D-printed [EEZYbotARM](https://www.instructables.com/EEZYbotARM/),
replacing the acrylic SNAM1500-style kit. The shoulder and elbow were subsequently
changed to MG90S; base and gripper use SG90. Limits found with `limit_finder.ino`:

| Joint | Min | Max | Travel |
|---|---|---|---|
| base | 0 | 180 | 180° |
| shoulder | 60 | 130 | 70° |
| elbow | 75 | 120 | 45° |
| gripper | 40 | 180 | 140° |

The linkage costs most of the arm's travel: the shoulder keeps 39% of a servo's range and
the elbow 25%. Consequences to watch: demonstrations occupy a small joint-space volume, so
recorded actions will have small variance, which makes the "hold still" open-loop baseline
even harder to beat. Whether the reachable workspace covers the task at all is the first
thing to check when placing the camera and the block.

Gripper direction was later confirmed: 180° open and 40° closed. Pot range was
confirmed as 0–1023 on all four channels.

## 2026-09-16 — 12 GB memory test: first attempt did not produce a measurement

**Setup.** `gello_fake_lora` (π0-FAST, `gemma_2b_lora`, LLM + SigLIP image encoder frozen,
action_dim 4, horizon 15), synthetic dataset (9 train episodes / 1,602 frames), batch size 1,
20 steps, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. openpi `215abfb`, JAX 0.5.3, WSL2 Ubuntu 24.04,
RTX 4070 12,282 MiB, WSL RAM 15.5 GB.

**What happened.** The run failed while restoring `pi0_fast_base` weights, before anything was
placed on the GPU (GPU memory never rose above baseline):

```
ValueError: RESOURCE_EXHAUSTED: Error reading "params.PaliGemma.llm.layers.attn.attn_vec_einsum.w/0.0.0.0" ...
Failed to read from file [OS error 12: ENOMEM Cannot allocate memory]
```

So this is **not** a GPU out-of-memory result. Contributing causes found:

1. **Another GPU training job was running in WSL** (`robomimic/scripts/train.py --config
   configs_dp/ted_s1.json`, ~3.5 h in, GPU utilization 76%, ~6 GB host RAM across processes).
   WSL had ~7.4 GB RAM available; openpi restores the 10.85 GB float32 checkpoint into host
   memory. Any GPU memory number measured alongside that job would also be wrong.
2. **Weights were stored on `/mnt/d`** (Windows D: via WSL's 9p bridge). The checkpoint is a
   few very large chunk files (up to 2.24 GB each). A standalone read of one 2.24 GB chunk
   did not finish within 14 minutes — too slow to be usable even when it doesn't fail.
3. **Windows apps held ~3.0 GB of GPU memory before the run** (Discord, Steam, Spotify, Riot
   Client, Edge WebView, ChatGPT/Codex, Explorer/shell), plus whatever the robomimic job held.

Also found: after the error, openpi's main thread exited but tensorstore worker threads kept
the process alive indefinitely. `memory_probe.py` now kills the process group 20 s after a
traceback and classifies host-memory/read failures separately from GPU OOM.

**Still unknown:** whether π0-FAST LoRA with a frozen image encoder fits in 12 GB. Needs a
rerun with the GPU and WSL RAM free and the weights on WSL's own disk.

## 2026-09-16 — Pipeline verified on synthetic data

- Converter: 12 synthetic episodes (3 conditions) → `local/gello_fake` (9 episodes, 1,602
  frames) + `local/gello_fake_heldout` (3 episodes, 533 frames, one per condition), LeRobot
  v2.1, 30 fps.
- openpi `compute_norm_stats` ran; joint action means ≈ 0 (delta actions applied), gripper
  mean 0.61 (absolute), as intended.
- `loadback_check`: openpi's data loader returns state (4,), actions (15, 4), image (3, 224,
  224); action chunks match resampled commands with **max abs diff 0.0** at episode starts,
  middles and padded ends; DeltaActions correct; full training batch tokenizes to 41 of 180
  tokens.
- Closed-loop controller dry run (fake follower, fake camera, hold policy): ~28 control
  ticks/s, re-plans every 5 ticks, starts from the follower's reported pose, clamps and
  rate-limits commands.

## 2026-09-16 — Camera

Sony ZV-1F over USB streaming: 1280×720 at 30.0 fps, frame interval median 33.3 ms / p95 42 ms /
max 47 ms, no gaps in 10 s. Requesting 1080p still delivers 720p.
