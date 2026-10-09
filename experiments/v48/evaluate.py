#!/usr/bin/env python3
"""Evaluation CLI — reproducible, paired, multi-control survival benchmark.

Purpose (diagnostic/baseline):
  1. Provide a single, fully reproducible eval entry point that records env
     config, dependency versions, exact seeds, per-episode results and timing.
  2. Re-evaluate the historical best single policy
     (v45 ms_baseline_v42cfg_seed700000 / model_iter_20.zip, documented report
     0.899) under per-fish control, and compare it, on the SAME paired episode
     seeds, against simple control arms: constant-action rules (hold,
     accelerate, decelerate, turn) and uniform-random, plus predator-avoid +
     boundary-avoid rules that read only the policy's visible observation.
     Only the avoid rules carry fixed hand-tuned thresholds; constant and random
     actions do not.
  3. Contrast the two control semantics for the same checkpoint:
     - policy_per_fish : independent per-fish action (current eval semantics)
     - policy_broadcast: single action broadcast to all alive fish, observation
       sampled round-robin over alive fish (faithful training semantics of
       SingleFishEnv.step / _single_observation).

Statistical unit is the episode. Paired differences vs a reference arm are
reported with bootstrap CIs. No episode is dropped.

Usage:
  experiments/v48/.venv/bin/python experiments/v48/evaluate.py \
      --seeds report --reference policy_per_fish \
      --arm policy_per_fish=experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip \
      --arm policy_broadcast=experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip \
      --arm rule_hold --arm rule_accelerate --arm rule_turn_left \
      --arm rule_turn_right --arm rule_random --arm rule_flee \
      --workers 8 --out experiments/v48/artifacts/results/eval_report.jsonl
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

# --- Rule thresholds (fixed a priori, NOT tuned on the report set) ---------
# Boundary avoidance kicks in at normalized distance-to-boundary obs[4] < 0.15.
# obs[4] = (STAGE_RADIUS - dist_to_center)/STAGE_RADIUS = dist_to_boundary/10,
# and the env's own boundary penalty starts at dist_to_boundary < 1.0 (=0.10).
# 0.15 acts slightly before the penalty; it is read off the env constants, not
# fit to any eval result.
BOUNDARY_AVOID_THRESHOLD = 0.15
# Forward if current heading is within this many degrees of desired heading.
HEADING_DEADZONE_DEG = 15.0
# Env constants mirrored from fish_env.py, used only to un-normalize obs.
VISION = 10.0 / 3.0
FISH_MAX_SPEED = 2.0

# --- Pre-set safe-zone candidate (fixed a priori, NOT tuned on results) -----
# Obs[0],obs[1] are normalized position in [-1,1] (pos/STAGE_RADIUS). "Top" is
# the world -y direction. Target radius 0.55 (well inside the wall, clear of the
# boundary penalty at margin<0.10) and target angle -90 degrees (-y). These are
# fixed design constants chosen from geometry, not fit to any eval number.
SAFE_TARGET_R = 0.55
SAFE_TARGET_ANGLE_DEG = -90.0  # world top = -y (predator gravity pushes toward +y)
# Once inside this normalized radius of the target point, switch to
# decelerate-then-hold. Chosen as 0.15 of normalized stage radius.
SAFE_ARRIVE_R = 0.15


def _steer(obs, desired):
    """Turn toward 'desired' (a vector) using one discrete turn/forward action."""
    vx, vy = float(obs[2]), float(obs[3])
    speed = (vx * vx + vy * vy) ** 0.5
    n = np.linalg.norm(desired)
    if n < 1e-6 or speed < 1e-6:
        return 0
    desired = desired / n
    cur = np.array([vx, vy], dtype=np.float64) / speed
    cross = cur[0] * desired[1] - cur[1] * desired[0]
    dot = float(np.clip(cur[0] * desired[0] + cur[1] * desired[1], -1.0, 1.0))
    angle = np.degrees(np.arctan2(cross, dot))
    if abs(angle) <= HEADING_DEADZONE_DEG:
        return 0
    return 1 if angle > 0 else 2


def _safe_top(obs):
    """Fixed safe-zone rule: head to a fixed world-top point, then slow and hold.

    Uses only own position (obs[0:2]) and own velocity (obs[2:4]); the predator
    is not read at all. After arrival: decelerate while speed non-negligible,
    then hold (maintain current velocity vector).
    """
    tx = SAFE_TARGET_R * np.cos(np.deg2rad(SAFE_TARGET_ANGLE_DEG))
    ty = SAFE_TARGET_R * np.sin(np.deg2rad(SAFE_TARGET_ANGLE_DEG))
    dx, dy = tx - float(obs[0]), ty - float(obs[1])
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= SAFE_ARRIVE_R:
        vx, vy = float(obs[2]), float(obs[3])
        speed = (vx * vx + vy * vy) ** 0.5
        return 3 if speed > 0.03 else 4
    return _steer(obs, np.array([dx, dy], dtype=np.float64))


def rule_action(rule, obs, rng):
    """Map a single fish observation (11+ dims) to a discrete action.

    Reads ONLY the observation the policy would see: obs[2:4] velocity,
    obs[4] boundary margin, obs[5] predator-visible flag, obs[6:8] predator
    relative position, obs[8:10] predator velocity (all normalized). Hidden
    predator state is never accessed.
    """
    if rule == "hold":
        return 4
    if rule == "accelerate":
        return 0
    if rule == "decelerate":
        return 3
    if rule == "turn_left":
        return 1
    if rule == "turn_right":
        return 2
    if rule == "random":
        return int(rng.integers(0, 5))
    if rule == "safe_top":
        return _safe_top(obs)
    if rule in ("flee", "flee_lead", "flee_orbit") or rule.startswith("flee_lead_t"):
        if obs[4] < BOUNDARY_AVOID_THRESHOLD:
            desired = np.array([-float(obs[0]), -float(obs[1])], dtype=np.float64)
        elif obs[5] > 0.5:
            rel = np.array([float(obs[6]), float(obs[7])], dtype=np.float64) * VISION
            if rule == "flee":
                desired = -rel
            elif rule == "flee_orbit":
                perp = np.array([-rel[1], rel[0]], dtype=np.float64)
                if float(np.dot(perp, -rel)) < 0:
                    perp = -perp
                desired = perp
            else:
                pvel = np.array([float(obs[8]), float(obs[9])], dtype=np.float64) * FISH_MAX_SPEED
                pspeed = float(np.linalg.norm(pvel))
                dist = float(np.linalg.norm(rel))
                if rule.startswith("flee_lead_t"):
                    tau = float(rule[len("flee_lead_t"):])
                else:
                    tau = dist / pspeed if pspeed > 1e-6 else 0.0
                    tau = min(tau, 5.0)
                desired = -(rel + pvel * tau)
        else:
            return 4
        return _steer(obs, desired)
    raise ValueError(f"unknown rule: {rule}")


# Survivals sampled at these simulation steps (step counter values; step 1 is
# the first post-reset step, which includes overlapping spawns).
PHASE_STEPS = (1, 100, 250, 500)


def run_episode_policy(model, env, seed, control):
    """Run one episode under a PPO policy. Returns a per-episode record."""
    obs, info = env.reset(seed=seed)
    pre_roll = (info or {}).get("pre_roll_stats")
    total_reward = 0.0
    steps = 0
    fish_pointer = 0
    track = _TrackState()
    while True:
        if len(obs) > 0:
            if control == "policy_per_fish":
                batch, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch)]
            elif control == "policy_broadcast":
                idx = fish_pointer % len(obs)
                fish_pointer = (idx + 1) % max(len(obs), 1)
                a, _ = model.predict(np.asarray(obs[idx]), deterministic=True)
                actions = [int(a)] * len(obs)
            else:
                raise ValueError(control)
        else:
            actions = []
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(np.mean(rewards)) if len(rewards) else 0.0
        steps += 1
        track.update(env, actions, steps, info)
        if terminated or truncated:
            break
    return _record(seed, control, info, steps, total_reward, pre_roll, track)


def run_episode_rule(env, seed, rule, rng_seed):
    obs, info = env.reset(seed=seed)
    pre_roll = (info or {}).get("pre_roll_stats")
    rng = np.random.default_rng(rng_seed)
    total_reward = 0.0
    steps = 0
    track = _TrackState()
    while True:
        if len(obs) > 0:
            actions = [rule_action(rule, obs[i], rng) for i in range(len(obs))]
        else:
            actions = []
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(np.mean(rewards)) if len(rewards) else 0.0
        steps += 1
        track.update(env, actions, steps, info)
        if terminated or truncated:
            break
    return _record(seed, f"rule_{rule}", info, steps, total_reward, pre_roll, track)


class _TrackState:
    """Collect per-episode phased survival, 2nd-half predator extent, and
    per-action counts. Kept tiny so it can run on every episode cheaply."""

    def __init__(self):
        self.surv_at = {}
        self.second_half_radius = []
        self.second_half_xy = []
        self.action_counts = [0] * 5

    def update(self, env, actions, step, info):
        for ph in PHASE_STEPS:
            if step == ph:
                self.surv_at[ph] = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
        if step >= 250:  # second half of the 500-step horizon
            r = float(np.linalg.norm(env.predator_pos))
            self.second_half_radius.append(r)
            self.second_half_xy.append((float(env.predator_pos[0]), float(env.predator_pos[1])))
        for a in actions:
            if 0 <= int(a) < 5:
                self.action_counts[int(a)] += 1

    def summarise(self):
        total_actions = sum(self.action_counts) or 1
        xy = self.second_half_xy
        if xy:
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            extent = {
                "n": len(xy),
                "radius_mean": float(np.mean(self.second_half_radius)),
                "radius_min": float(np.min(self.second_half_radius)),
                "radius_max": float(np.max(self.second_half_radius)),
                "x_span": float(max(xs) - min(xs)),
                "y_span": float(max(ys) - min(ys)),
            }
        else:
            extent = {"n": 0}
        return {
            "survival_at": {str(k): v for k, v in self.surv_at.items()},
            "predator_second_half": extent,
            "action_dist": [c / total_actions for c in self.action_counts],
        }


def _record(seed, control, info, steps, total_reward, pre_roll, track=None):
    num_alive = int(info.get("num_alive", 0))
    death_ts = info.get("death_timesteps")
    step_one = info.get("step_one_deaths")
    rec = {
        "seed": int(seed),
        "control": control,
        "final_num_alive": num_alive,
        "final_survival": num_alive / float(common.NUM_FISH),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "avg_reward": float(total_reward / max(steps, 1)),
        "first_death_step": int(info.get("first_death_step", -1)),
        "num_step_one_deaths": len(step_one) if step_one else 0,
        "num_dead_in_episode": (sum(1 for d in death_ts if d is not None and d >= 0)
                                if death_ts else common.NUM_FISH - num_alive),
        "avg_predator_speed": (float(pre_roll.get("final_speed", 0.0))
                               if pre_roll else None),
    }
    if track is not None:
        rec.update(track.summarise())
    return rec


def build_arm(spec):
    """spec: 'name=modelpath' for policy arms, or bare 'rule_name'."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return {"name": name, "kind": "policy", "path": path}
    if spec.startswith("rule_"):
        return {"name": spec, "kind": "rule", "rule": spec[len("rule_"):]}
    raise ValueError(f"cannot parse arm spec: {spec}")


def worker(job):
    import torch

    torch.set_num_threads(1)
    arm, seeds = job
    env = common.make_env()
    records = []
    model = None
    if arm["kind"] == "policy":
        from stable_baselines3 import PPO

        model = PPO.load(arm["path"], device="cpu")
    try:
        for seed in seeds:
            t0 = time.time()
            if arm["kind"] == "policy":
                rec = run_episode_policy(model, env, seed, arm["name"])
            else:
                rec = run_episode_rule(env, seed, arm["rule"], rng_seed=seed * 2654435761 % (2 ** 31))
            rec["elapsed_sec"] = time.time() - t0
            records.append(rec)
    finally:
        env.close()
    return arm["name"], records


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=481003):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="report",
                    help="seed set: smoke|dev|report|legacy_report")
    ap.add_argument("--arm", action="append", required=True,
                    help="repeatable; policy: name=path, rule: rule_<name>")
    ap.add_argument("--reference", default=None,
                    help="arm name to compute paired differences against")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="per-episode JSONL path")
    ap.add_argument("--summary-out", default=None, help="summary JSON path")
    args = ap.parse_args()

    seeds = common.resolve_seeds(args.seeds)
    arms = [build_arm(s) for s in args.arm]
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate arm names")

    # Split each arm's seeds across workers (load model once per chunk).
    jobs = []
    n_chunks = max(1, min(args.workers, len(seeds)))
    chunks = [list(c) for c in np.array_split(np.array(seeds), n_chunks)]
    for arm in arms:
        for c in chunks:
            jobs.append((arm, [int(x) for x in c]))

    t_start = time.time()
    all_records = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for name, recs in ex.map(worker, jobs):
            all_records.setdefault(name, []).extend(recs)
    wall = time.time() - t_start

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name in names:
            for rec in sorted(all_records[name], key=lambda r: r["seed"]):
                f.write(json.dumps(rec) + "\n")

    # Paired summary
    by_seed = {}
    for name in names:
        for rec in all_records[name]:
            by_seed.setdefault(rec["seed"], {})[name] = rec

    summary = {
        "seed_set": args.seeds,
        "n_seeds": len(seeds),
        "workers": args.workers,
        "wall_sec": wall,
        "dependency_snapshot": common.dependency_snapshot(),
        "env_config": {k: v for k, v in common.env_config().items() if k != "predator_heading_bias"},
        "arms": {},
        "paired_vs_reference": {},
        "reference": args.reference,
    }
    for name in names:
        vals = np.array([r["final_survival"] for r in all_records[name]], dtype=float)
        # Phased survival: mean over episodes where the phase step was reached.
        phased = {}
        for ph in PHASE_STEPS:
            got = [r["survival_at"][str(ph)] for r in all_records[name]
                   if str(ph) in r.get("survival_at", {})]
            phased[str(ph)] = float(np.mean(got)) if got else None
        # Action distribution: mean over episodes.
        adist = np.array([r.get("action_dist", [0, 0, 0, 0, 0]) for r in all_records[name]], dtype=float)
        mean_adist = adist.mean(axis=0).tolist() if adist.size else [0, 0, 0, 0, 0]
        # Second-half predator extent.
        ext_n = [r["predator_second_half"].get("n", 0) for r in all_records[name]]
        radius_min = [r["predator_second_half"]["radius_min"] for r in all_records[name]
                      if "radius_min" in r["predator_second_half"]]
        radius_max = [r["predator_second_half"]["radius_max"] for r in all_records[name]
                      if "radius_max" in r["predator_second_half"]]
        x_span = [r["predator_second_half"]["x_span"] for r in all_records[name]
                  if "x_span" in r["predator_second_half"]]
        summary["arms"][name] = {
            "n": len(vals),
            "mean_final_survival": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean_steps": float(np.mean([r["steps"] for r in all_records[name]])),
            "total_step_one_deaths": int(sum(r["num_step_one_deaths"] for r in all_records[name])),
            "mean_elapsed_per_episode_sec": float(np.mean([r["elapsed_sec"] for r in all_records[name]])),
            "mean_survival_at_step": {k: v for k, v in phased.items()},
            "mean_action_dist": mean_adist,
            "predator_second_half": {
                "episodes_with_data": int(sum(1 for n in ext_n if n > 0)),
                "mean_radius_min": float(np.mean(radius_min)) if radius_min else None,
                "mean_radius_max": float(np.mean(radius_max)) if radius_max else None,
                "mean_x_span": float(np.mean(x_span)) if x_span else None,
            },
        }
    if args.reference:
        if args.reference not in names:
            raise SystemExit(f"reference {args.reference} not among arms")
        common_seeds = sorted(by_seed)
        for name in names:
            if name == args.reference:
                continue
            diffs = []
            for s in common_seeds:
                if name in by_seed[s] and args.reference in by_seed[s]:
                    diffs.append(by_seed[s][name]["final_survival"]
                                 - by_seed[s][args.reference]["final_survival"])
            lo, hi = bootstrap_ci(diffs)
            d = np.array(diffs, dtype=float)
            summary["paired_vs_reference"][name] = {
                "n_pairs": len(diffs),
                "mean_diff": float(d.mean()),
                "std_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "win_rate": float((d > 0).mean()),
                "tie_rate": float((d == 0).mean()),
                "loss_rate": float((d < 0).mean()),
            }

    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[written] {out_path}\n[written] {summary_path}")


if __name__ == "__main__":
    main()
