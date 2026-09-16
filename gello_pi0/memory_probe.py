#!/usr/bin/env python3
"""
memory_probe.py — does π0-FAST LoRA training fit on this GPU? Measure, don't guess.

For each batch size, runs openpi's real train.py (real data loader, real
pi0_fast_base weights, real optimizer) for a few steps in a fresh process, and
records:
  - peak total GPU memory from nvidia-smi, sampled every 0.25 s
  - baseline GPU memory already used before the run (Windows desktop, other apps)
  - whether it hit an out-of-memory error, and the loss / step times it printed

JAX preallocation is disabled so nvidia-smi reflects what JAX actually grabbed.
JAX's allocator does not return memory, so the peak is an upper bound on what
training needed — slightly pessimistic, never optimistic.

    cd ~/projects/openpi
    uv run python -m gello_pi0.memory_probe --config gello_fake_lora --batch-sizes 1 2 4 --steps 20
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

OOM_PATTERNS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory", "OOM")


def gpu_memory():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip().splitlines()[0]
    used, total, name = [x.strip() for x in out.split(",")]
    return int(used), int(total), name


class Sampler(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.samples = []
        self.stop_flag = threading.Event()

    def run(self):
        start = time.time()
        while not self.stop_flag.is_set():
            try:
                used, _, _ = gpu_memory()
                self.peak = max(self.peak, used)
                self.samples.append((round(time.time() - start, 2), used))
            except Exception:
                pass
            time.sleep(self.interval)


def host_ram_used_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            info[key] = int(value.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) // 1024


def probe(config, batch_size, steps, log_dir, timeout_s):
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["GELLO_SKIP_CHECKPOINTS"] = "1"
    cmd = [sys.executable, "-m", "gello_pi0.run", "train", config,
           f"--exp-name=memprobe_bs{batch_size}", "--overwrite",
           f"--batch-size={batch_size}", f"--num-train-steps={steps}", "--log-interval=1"]

    baseline, total, name = gpu_memory()
    log_path = log_dir / f"bs{batch_size}.log"
    sampler = Sampler()
    ram_peak = host_ram_used_mb()
    start = time.time()
    sampler.start()
    timed_out = False
    fatal_seen_at = None
    with open(log_path, "w") as log:
        # New session so the whole process group (data loader workers, tensorstore threads) can be killed.
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while proc.poll() is None:
            ram_peak = max(ram_peak, host_ram_used_mb())
            if time.time() - start > timeout_s:
                timed_out = True
                break
            # After a fatal error openpi's main thread exits but worker threads can keep the
            # process alive indefinitely. Give it 20 s to exit on its own, then kill it.
            if fatal_seen_at is None and "Traceback (most recent call last)" in log_path.read_text(errors="ignore"):
                fatal_seen_at = time.time()
            if fatal_seen_at is not None and time.time() - fatal_seen_at > 20:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    sampler.stop_flag.set()
    sampler.join()
    elapsed = time.time() - start

    text = log_path.read_text(errors="ignore")
    losses = [float(m) for m in re.findall(r"Step \d+: .*?loss=([0-9.eE+-]+)", text)]
    step_lines = re.findall(r"Step (\d+):", text)
    host_oom = "ENOMEM" in text or "Cannot allocate memory" in text
    oom = any(p in text for p in OOM_PATTERNS) and not host_oom       # GPU out of memory
    completed = proc.returncode == 0 and not oom and not timed_out

    return {
        "batch_size": batch_size,
        "completed": completed,
        "returncode": proc.returncode,
        "out_of_memory": oom,
        "host_ram_exhausted_or_read_failed": host_oom,
        "failed_before_training": len(step_lines) == 0,
        "timed_out": timed_out,
        "steps_logged": len(step_lines),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "gpu": name,
        "gpu_total_mb": total,
        "gpu_baseline_used_mb": baseline,
        "gpu_peak_used_mb": sampler.peak,
        "gpu_peak_attributable_mb": sampler.peak - baseline,
        "gpu_headroom_at_peak_mb": total - sampler.peak,
        "host_ram_peak_used_mb": ram_peak,
        "wall_time_s": round(elapsed, 1),
        "log": str(log_path),
        "memory_trace": sampler.samples[::4],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gello_fake_lora")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-min", type=float, default=45)
    parser.add_argument("--stop-after-failure", action="store_true", default=True)
    args = parser.parse_args()

    from gello_pi0.openpi_configs import RUNS_DIR
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RUNS_DIR / "memory_probe" / f"{args.config}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for bs in sorted(args.batch_sizes):
        print(f"[probe] {args.config} batch_size={bs} steps={args.steps} ...", flush=True)
        r = probe(args.config, bs, args.steps, log_dir, args.timeout_min * 60)
        results.append(r)
        status = ("FITS" if r["completed"] else "GPU OOM" if r["out_of_memory"]
                  else "HOST MEMORY / READ FAILURE" if r["host_ram_exhausted_or_read_failed"] else "FAILED")
        print(f"[probe]   {status}: peak {r['gpu_peak_used_mb']} / {r['gpu_total_mb']} MiB total "
              f"(baseline {r['gpu_baseline_used_mb']} MiB before start), "
              f"loss {r['loss_first']} -> {r['loss_last']}, {r['wall_time_s']} s, log {r['log']}", flush=True)
        (log_dir / "results.json").write_text(json.dumps(results, indent=2))
        if not r["completed"] and args.stop_after_failure:
            break
    print(f"[probe] results: {log_dir / 'results.json'}")


if __name__ == "__main__":
    main()
