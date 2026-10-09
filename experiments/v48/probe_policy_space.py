#!/usr/bin/env python3
"""v48 policy-space / difficulty probe.

Setting question: under the current environment (bouncing predator, local
vision 3.33, 30-degree discrete turns, speed cap, escape boost, no fish-fish
interaction), how much strategy room is there, and is a simple reactive rule
near the ceiling?

Measurements (evaluation-side only, no env change):
  1. Predator dynamics: radius / speed / heading-change, per step. Characterize
     the "gravity ball" motion.
  2. Threat visibility at death: fraction of deaths in which the predator was
     visible to that fish on the previous step. High visible-death share =
     threat mostly actionable from local obs; high invisible share = luck.

Usage:
  experiments/v48/.venv/bin/python experiments/v48/probe_policy_space.py \
      --seeds dev --control policy_per_fish \
      --model experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip \
      --out experiments/v48/artifacts/results/probe_policy_space.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402


def probe(model_path, seeds, control):
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO

    env = common.make_env()
    model = PPO.load(model_path, device="cpu") if model_path else None

    pred_radius, pred_speed, pred_heading = [], [], []
    vis_steps = 0
    total_fish_steps = 0
    ep_all_see = []
    deaths_visible = 0
    deaths_invisible = 0
    deaths_step1 = 0
    first_death_steps = []
    # per-death warning window and boundary state
    warning_windows = []
    death_margin = []

    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        prev_heading = None
        ep_vis = 0
        ep_fish_steps = 0
        fd = None
        # per-world-fish consecutive visible-step counter
        vis_streak = np.zeros(env.NUM_FISH, dtype=np.int32)
        while True:
            pv = env.predator_vel
            pp = env.predator_pos
            sp = float(np.linalg.norm(pv))
            pred_radius.append(float(np.linalg.norm(pp)))
            pred_speed.append(sp)
            h = float(np.degrees(np.arctan2(pv[1], pv[0]))) if sp > 1e-6 else 0.0
            pred_heading.append(h)

            alive_world = np.where(env.fish_alive)[0]
            vis_before = (obs[:, 5] > 0.5) if len(obs) > 0 else np.array([], dtype=bool)
            # update streaks from this step's observation
            for k, w in enumerate(alive_world):
                if vis_before[k]:
                    vis_streak[w] += 1
                else:
                    vis_streak[w] = 0
            if len(obs) > 0:
                ep_vis += int(vis_before.sum())
                ep_fish_steps += len(obs)
            before_alive = env.fish_alive.copy()

            if control == "policy_per_fish" and len(obs) > 0:
                batch, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch)]
            else:
                actions = [4] * len(obs)

            obs, _r, term, trunc, info = env.step(actions)

            died_world = np.where(before_alive & ~env.fish_alive)[0]
            if died_world.size:
                if fd is None:
                    fd = int(env.timestep)
                for w in died_world:
                    pos = np.where(alive_world == w)[0]
                    if pos.size and vis_before[pos[0]]:
                        deaths_visible += 1
                    else:
                        deaths_invisible += 1
                    if env.timestep == 1:
                        deaths_step1 += 1
                    else:
                        warning_windows.append(int(vis_streak[w]))
                        # boundary margin of the fish just before death (approx:
                        # current position; fish barely moved in one step)
                        m = (env.STAGE_RADIUS - float(np.linalg.norm(env.fish_positions[w]))) / env.STAGE_RADIUS
                        death_margin.append(m)
            if term or trunc:
                break
        ep_all_see.append(ep_vis / max(ep_fish_steps, 1))
        vis_steps += ep_vis
        total_fish_steps += ep_fish_steps
        first_death_steps.append(fd if fd is not None else -1)
    env.close()

    heading_changes = [abs((pred_heading[i] - pred_heading[i - 1] + 180) % 360 - 180)
                       for i in range(1, len(pred_heading))]
    total_deaths = deaths_visible + deaths_invisible
    return {
        "control": control,
        "model": model_path,
        "n_episodes": len(seeds),
        "predator": {
            "radius_mean": float(np.mean(pred_radius)),
            "radius_p95": float(np.percentile(pred_radius, 95)),
            "speed_mean": float(np.mean(pred_speed)),
            "speed_max": float(np.max(pred_speed)),
            "heading_change_deg_mean": float(np.mean(heading_changes)) if heading_changes else 0.0,
            "heading_change_deg_p95": float(np.percentile(heading_changes, 95)) if heading_changes else 0.0,
            "heading_change_deg_max": float(np.max(heading_changes)) if heading_changes else 0.0,
        },
        "visibility": {
            "frac_fish_steps_predator_visible": vis_steps / max(total_fish_steps, 1),
            "mean_per_episode_frac_fish_seeing": float(np.mean(ep_all_see)),
        },
        "deaths": {
            "total": total_deaths,
            "visible_prev_step": deaths_visible,
            "invisible_prev_step": deaths_invisible,
            "frac_visible": deaths_visible / max(total_deaths, 1),
            "step1_deaths": deaths_step1,
            "nontrivial_deaths": len(warning_windows),
            "warning_window_steps_mean": float(np.mean(warning_windows)) if warning_windows else 0.0,
            "warning_window_steps_median": float(np.median(warning_windows)) if warning_windows else 0.0,
            "warning_window_steps_p90": float(np.percentile(warning_windows, 90)) if warning_windows else 0.0,
            "non_step1_deaths_near_boundary_frac": float(np.mean([m < 0.15 for m in death_margin])) if death_margin else 0.0,
        },
        "first_death_step_mean": float(np.mean([d for d in first_death_steps if d > 0])) if any(d > 0 for d in first_death_steps) else -1,
        "first_death_step_is_1_frac": float(np.mean([d == 1 for d in first_death_steps])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="dev")
    ap.add_argument("--control", default="policy_per_fish")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = common.resolve_seeds(args.seeds)
    res = probe(args.model, seeds, args.control)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
