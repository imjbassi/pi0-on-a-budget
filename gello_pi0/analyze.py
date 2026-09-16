#!/usr/bin/env python3
"""
analyze.py — does open-loop error predict closed-loop success, overall and within conditions?

Inputs:
  --open-loop LABEL=path/to/summary.json   (repeat per checkpoint; LABEL must match the
                                           --checkpoint-label used for closed-loop trials)
  --closed-loop DIR                        closed_loop.py --outdir

For each checkpoint label: closed-loop success rate (Wilson 95% CI) overall and per
condition, next to open-loop policy MSE and "beats hold" fraction overall and per condition.

Correlations (Spearman, open-loop MSE vs success rate; negative = agree):
  aggregate         one point per checkpoint
  cells             one point per (checkpoint, condition)
  within condition  one point per checkpoint, separately for each condition

The research question is whether "aggregate" looks good while "within condition" does
not. With few checkpoints these numbers are fragile — n and an exact permutation
p-value are printed next to every correlation, and nothing is computed below n = 3.

Dry-run trials are excluded.

    python -m gello_pi0.analyze --open-loop step1000=.../1000/open_loop/summary.json \
        --open-loop step2999=.../2999/open_loop/summary.json --closed-loop closed_loop_trials
"""

import argparse
import itertools
import json
import math
import os

import numpy as np


def wilson(successes, n, z=1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(center - half, 3), round(center + half, 3))


def ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for value in np.unique(x):                      # average ranks for ties
        tie = x == value
        r[tie] = r[tie].mean()
    return r


def spearman(x, y):
    """Spearman rho plus exact (n <= 8) or Monte Carlo two-sided permutation p-value."""
    n = len(x)
    if n < 3:
        return {"n": n, "rho": None, "p": None, "note": "fewer than 3 points"}
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"n": n, "rho": None, "p": None, "note": "no variation in one variable"}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if n <= 8:
        perms = [np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)]
    else:
        rng = np.random.default_rng(0)
        perms = [np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(20000)]
    p = float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))
    return {"n": n, "rho": round(rho, 3), "p": round(p, 4)}


def load_trials(folder):
    trials = []
    for root, _, files in os.walk(folder):
        if "trial.json" in files:
            with open(os.path.join(root, "trial.json")) as f:
                t = json.load(f)
            if not t.get("dry_run"):
                trials.append(t)
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-loop", action="append", required=True, metavar="LABEL=SUMMARY_JSON")
    parser.add_argument("--closed-loop", required=True)
    parser.add_argument("--out", default="analysis.json")
    args = parser.parse_args()

    open_loop = {}
    for item in args.open_loop:
        label, path = item.split("=", 1)
        with open(path) as f:
            open_loop[label] = json.load(f)
    trials = load_trials(args.closed_loop)

    table = []
    for label, ol in sorted(open_loop.items()):
        mine = [t for t in trials if t["checkpoint_label"] == label]
        conditions = sorted({t["condition"] for t in mine} | set(ol["by_condition"]))
        for cond in [None] + conditions:
            cl = mine if cond is None else [t for t in mine if t["condition"] == cond]
            ol_stats = ol["overall"] if cond is None else ol["by_condition"].get(cond)
            s = sum(t["success"] for t in cl)
            table.append({
                "checkpoint": label,
                "condition": cond or "ALL",
                "trials": len(cl),
                "success_rate": round(s / len(cl), 3) if cl else None,
                "success_ci95": wilson(s, len(cl)),
                "open_loop_mse": ol_stats["policy_mse"] if ol_stats else None,
                "hold_mse": ol_stats["hold_mse"] if ol_stats else None,
                "beats_hold_frac": ol_stats["policy_beats_hold_frac"] if ol_stats else None,
                "open_loop_chunks": ol_stats["n"] if ol_stats else 0,
            })

    def usable(rows):
        return [r for r in rows if r["success_rate"] is not None and r["open_loop_mse"] is not None]

    overall_rows = usable([r for r in table if r["condition"] == "ALL"])
    cell_rows = usable([r for r in table if r["condition"] != "ALL"])
    conditions = sorted({r["condition"] for r in cell_rows})
    correlations = {
        "aggregate": spearman([r["open_loop_mse"] for r in overall_rows], [r["success_rate"] for r in overall_rows]),
        "cells": spearman([r["open_loop_mse"] for r in cell_rows], [r["success_rate"] for r in cell_rows]),
        "within_condition": {
            c: spearman([r["open_loop_mse"] for r in cell_rows if r["condition"] == c],
                        [r["success_rate"] for r in cell_rows if r["condition"] == c])
            for c in conditions
        },
    }

    result = {"table": table, "correlations": correlations, "trials_used": len(trials),
              "synthetic_open_loop": any(ol.get("synthetic_data") for ol in open_loop.values())}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'checkpoint':<16}{'condition':<16}{'trials':>7}{'success':>9}{'95% CI':>16}{'OL mse':>11}{'hold mse':>11}{'>hold':>7}")
    for r in table:
        ci = f"{r['success_ci95'][0]}-{r['success_ci95'][1]}" if r["trials"] else "-"
        fmt = lambda v, spec: format(v, spec) if v is not None else "-"
        print(f"{r['checkpoint']:<16}{r['condition']:<16}{r['trials']:>7}{fmt(r['success_rate'], '.2f'):>9}{ci:>16}"
              f"{fmt(r['open_loop_mse'], '.5f'):>11}{fmt(r['hold_mse'], '.5f'):>11}{fmt(r['beats_hold_frac'], '.2f'):>7}")
    print("\nSpearman(open-loop MSE, success rate) — negative means the metrics agree:")
    print(f"  aggregate (per checkpoint): {correlations['aggregate']}")
    print(f"  per (checkpoint, condition): {correlations['cells']}")
    for c, v in correlations["within_condition"].items():
        print(f"  within {c}: {v}")
    if result["synthetic_open_loop"]:
        print("\n[!!] open-loop results come from SYNTHETIC data")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
