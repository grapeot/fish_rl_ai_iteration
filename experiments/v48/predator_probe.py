#!/usr/bin/env python3
"""Predator-only trajectory probe (near-free; predator does not depend on
fish, so no fish or policy simulation is needed).

Question: on a per-episode basis, how much of the episode does the predator
spend at a given radius from the origin? Reports per-phase sample statistics of
the predator radius (mean and per-episode minimum), to characterize how its
spatial coverage changes over the 500-step horizon. These are sample statistics
over the observed seeds, not a per-trajectory monotonicity claim and not a
global safety lower bound.

Reproduces the reset predator init exactly (spawn jitter, heading bias, pre-roll
with angle/speed jitter and speed bias) using the same code path as the env.

Usage:
  experiments/v48/.venv/bin/python experiments/v48/predator_probe.py \
      --out experiments/v48/artifacts/results/predator_probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

# Bin edges over the 500-step horizon (step index 1..500).
BINS = [(1, 50), (51, 100), (101, 150), (151, 250), (251, 500)]


def run_predator_only(seed):
    """Instantiate the env for its reset logic, then advance only the predator
    via env._update_predator; fish state is irrelevant and never read."""
    env = common.make_env()
    env.reset(seed=seed)
    radii = []
    for step in range(1, 501):
        env._update_predator()
        radii.append(float(np.linalg.norm(env.predator_pos)))
    env.close()
    return radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="report")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = common.resolve_seeds(args.seeds)

    all_r = np.array([run_predator_only(int(s)) for s in seeds])  # (n_ep, 500)
    out = {
        "n_episodes": len(seeds),
        "stage_radius": 10.0,
        "phase_radius": {},
    }
    for lo, hi in BINS:
        sub = all_r[:, lo - 1:hi]
        out["phase_radius"][f"{lo}-{hi}"] = {
            "mean": float(sub.mean()),
            "per_episode_min_mean": float(sub.min(axis=1).mean()),
            "per_episode_min_p05": float(np.percentile(sub.min(axis=1), 5)),
            "global_min": float(sub.min()),
            "frac_steps_radius_lt_7": float((sub < 7.0).mean()),
            "frac_steps_radius_lt_6": float((sub < 6.0).mean()),
        }
    # Per-episode min radius over the whole episode and over the second half.
    out["per_episode_min_radius_all"] = all_r.min(axis=1).tolist()
    out["per_episode_min_radius_second_half"] = all_r[:, 249:].min(axis=1).tolist()
    out["second_half_min_radius_gt_6_frac"] = float((all_r[:, 249:].min(axis=1) > 6.0).mean())
    out["second_half_min_radius_gt_7_frac"] = float((all_r[:, 249:].min(axis=1) > 7.0).mean())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
