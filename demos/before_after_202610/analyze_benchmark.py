#!/usr/bin/env python3
"""Summarise the same-scene benchmark and compute scenario-bootstrap CIs.

Paired contrasts are formed at the scenario level: for each scenario the three
NEW replicates are averaged first (``new_fixed3_mean``), then the contrast against
the OLD checkpoint is taken per scenario, then a 10000-draw scenario bootstrap over
the 64 scenarios gives the 95% CI. The same procedure is used for rule - old and
rule - new_fixed3_mean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import demo_common as C  # noqa: E402

BOOT_N = 10000
BOOT_SEED = 590301


def load_raw(path):
    by_ctrl = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_ctrl.setdefault(r["controller"], {})[int(r["seed"])] = r
    return by_ctrl


def boot_ci(diffs, n=BOOT_N, seed=BOOT_SEED):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrast(by_ctrl, a_fn, b_fn, seeds):
    diffs = np.array([a_fn(s) - b_fn(s) for s in seeds], dtype=float)
    lo, hi = boot_ci(diffs)
    win = int((diffs > 1e-12).sum())
    loss = int((diffs < -1e-12).sum())
    tie = len(seeds) - win - loss
    return {
        "mean_diff": float(diffs.mean()),
        "ci95": [lo, hi],
        "win": win, "loss": loss, "tie": tie,
        "n": len(seeds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(C.DEMO_DIR / "results" / "report_raw.jsonl"))
    ap.add_argument("--out", default=str(C.DEMO_DIR / "results" / "report_summary.json"))
    args = ap.parse_args()

    by_ctrl = load_raw(args.raw)
    seeds = C.report_seeds()
    assert all(len(by_ctrl[n]) == len(seeds) for n in C.CONTROLLER_NAMES), "incomplete grid"

    surv = {n: np.array([by_ctrl[n][s]["final_survival"] for s in seeds]) for n in C.CONTROLLER_NAMES}
    new_mean = np.mean([surv[n] for n in C.NEW_REPS], axis=0)
    new_by_seed = {s: float(new_mean[i]) for i, s in enumerate(seeds)}
    old_by_seed = {s: float(surv["old_ppo"][i]) for i, s in enumerate(seeds)}
    rule_by_seed = {s: float(surv[C.RULE_NAME][i]) for i, s in enumerate(seeds)}
    rep0_by_seed = {s: float(surv["new_rep0"][i]) for i, s in enumerate(seeds)}

    summary = {"kind": "before_after_benchmark_summary", "n_scenarios": len(seeds),
               "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "per_controller": {}}
    for n in C.CONTROLLER_NAMES:
        recs = [by_ctrl[n][s] for s in seeds]
        summary["per_controller"][n] = {
            "mean_final_survival": float(surv[n].mean()),
            "std": float(surv[n].std(ddof=1)),
            "mean_final_num_alive": float(np.mean([r["final_num_alive"] for r in recs])),
            "mean_steps": float(np.mean([r["steps"] for r in recs])),
            "total_step_one_deaths": int(sum(r["num_step_one_deaths"] for r in recs)),
            "mean_survival_at": {
                str(ph): float(np.mean([r["survival_at"][str(ph)] for r in recs]))
                for ph in C.PHASE_STEPS
            },
        }
    summary["new_fixed3_mean"] = {
        "mean_final_survival": float(new_mean.mean()),
        "std": float(new_mean.std(ddof=1)),
    }

    contrasts = {
        "new_fixed3_mean_minus_old": contrast(by_ctrl, new_by_seed.__getitem__, old_by_seed.__getitem__, seeds),
        "rule_minus_old": contrast(by_ctrl, rule_by_seed.__getitem__, old_by_seed.__getitem__, seeds),
        "rule_minus_new_fixed3_mean": contrast(by_ctrl, rule_by_seed.__getitem__, new_by_seed.__getitem__, seeds),
        "new_rep0_minus_old": contrast(by_ctrl, rep0_by_seed.__getitem__, old_by_seed.__getitem__, seeds),
    }
    summary["contrasts"] = contrasts

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
