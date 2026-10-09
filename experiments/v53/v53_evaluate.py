#!/usr/bin/env python3
"""v53 evaluation: paired multi-arm survival benchmark on the fresh v53 seed banks.

Fork of experiments/v50/v50_evaluate.py with:
  * v53 seed banks (530100 smoke / 530101 selection / 530102 report), disjoint
    from every prior bank;
  * all modules renamed `v53_*` to avoid worker-process import collisions;
  * a report gate (`--require-manifest`) that binds the report run to the frozen
    selection manifest INCLUDING the full env config (`predator_heading_bias`
    included, closing the v50 binding gap).

Semantics are unchanged: statistical unit is the episode; policy arms run
per-fish deterministic inference over the base 96-fish env (neighbor off, 11-dim
obs); rule arms read only the base 11 dims; paired diffs vs the reference use
episode bootstrap CIs. No episode is dropped. Evaluation runs the base multi-fish
env directly, so the training-time termination semantics of either arm do not
enter evaluation.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v53_common as common  # noqa: E402
import v53_verify as verify  # noqa: E402

BOUNDARY_AVOID_THRESHOLD = 0.15
HEADING_DEADZONE_DEG = 15.0
VISION = 10.0 / 3.0
FISH_MAX_SPEED = 2.0
SAFE_TARGET_R = 0.55
SAFE_TARGET_ANGLE_DEG = -90.0
SAFE_ARRIVE_R = 0.15
PHASE_STEPS = (1, 100, 250, 500)


def _steer(obs, desired):
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


class _TrackState:
    def __init__(self):
        self.surv_at = {}
        self.second_half_radius = []
        self.second_half_xy = []
        self.action_counts = [0] * 5

    def update(self, env, actions, step, info):
        for ph in PHASE_STEPS:
            if step == ph:
                self.surv_at[ph] = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
        if step >= 250:
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


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=530201):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _make_env(neighbor: bool):
    return common.make_base_env(include_neighbor_features=neighbor)


def run_episode_policy(model, env, seed):
    obs, info = env.reset(seed=seed)
    pre_roll = (info or {}).get("pre_roll_stats")
    total_reward = 0.0
    steps = 0
    track = _TrackState()
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=True)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(np.mean(rewards)) if len(rewards) else 0.0
        steps += 1
        track.update(env, actions, steps, info)
        if terminated or truncated:
            break
    return _record(seed, info, steps, total_reward, pre_roll, track)


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
    return _record(seed, info, steps, total_reward, pre_roll, track)


def _record(seed, info, steps, total_reward, pre_roll, track=None):
    num_alive = int(info.get("num_alive", 0))
    rec = {
        "seed": int(seed),
        "final_num_alive": num_alive,
        "final_survival": num_alive / float(common.NUM_FISH),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "avg_reward": float(total_reward / max(steps, 1)),
        "first_death_step": int(info.get("first_death_step", -1)),
        "num_step_one_deaths": len(info.get("step_one_deaths") or []),
    }
    if track is not None:
        rec.update(track.summarise())
    return rec


def build_arm(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
        kind = "untrained" if path == "untrained" else "policy"
        return {"name": name, "kind": kind, "path": path}
    if spec.startswith("rule_"):
        return {"name": spec, "kind": "rule", "rule": spec[len("rule_"):]}
    raise ValueError(f"cannot parse arm spec: {spec}")


def worker(job):
    import torch

    torch.set_num_threads(1)
    arm, seeds, neighbor = job
    env = _make_env(neighbor)
    records = []
    model = None
    if arm["kind"] == "policy":
        from stable_baselines3 import PPO

        model = PPO.load(arm["path"], device="cpu")
    try:
        for seed in seeds:
            t0 = time.time()
            if arm["kind"] in ("policy", "untrained"):
                rec = run_episode_policy(model, env, seed)
            else:
                rec = run_episode_rule(env, seed, arm["rule"],
                                       rng_seed=seed * 2654435761 % (2 ** 31))
            rec["control"] = arm["name"]
            rec["elapsed_sec"] = time.time() - t0
            records.append(rec)
    finally:
        env.close()
    return arm["name"], records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="report")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--neighbor", choices=("on", "off"), default="off")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-manifest", default=None)
    args = ap.parse_args()

    neighbor = args.neighbor == "on"
    seeds = common.resolve_seeds(args.seeds)
    arms = [build_arm(s) for s in args.arm]
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate arm names")

    # FULL env config: heading bias included (the v50 binding gap, closed here).
    actual_env_config = common.env_config(neighbor)

    arm_hashes = {}
    for a in arms:
        if a["kind"] == "policy":
            arm_hashes[a["name"]] = verify.policy_tensor_sha256_from_zip(a["path"])

    if args.require_manifest:
        try:
            manifest = verify.load_manifest(args.require_manifest)
            verify.validate_manifest_gate(
                manifest,
                seeds_name=args.seeds,
                actual_seeds=seeds,
                neighbor=neighbor,
                actual_env_config=actual_env_config,
                arm_specs=arms,
                arm_hashes=arm_hashes,
            )
        except verify.ManifestError as exc:
            raise SystemExit(f"refusing to run: manifest gate failed: {exc}")
        print(f"[manifest gate] OK against {args.require_manifest}")

    jobs = []
    n_chunks = max(1, min(args.workers, len(seeds)))
    chunks = [list(c) for c in np.array_split(np.array(seeds), n_chunks)]
    for arm in arms:
        for c in chunks:
            jobs.append((arm, [int(x) for x in c], neighbor))

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

    by_seed = {}
    for name in names:
        for rec in all_records[name]:
            by_seed.setdefault(rec["seed"], {})[name] = rec

    summary = {
        "seed_set": args.seeds,
        "neighbor_features": neighbor,
        "n_seeds": len(seeds),
        "workers": args.workers,
        "wall_sec": wall,
        "dependency_snapshot": common.dependency_snapshot(),
        "env_config": actual_env_config,
        "arm_model_sha256": arm_hashes,
        "arms": {},
        "paired_vs_reference": {},
        "reference": args.reference,
    }
    for name in names:
        vals = np.array([r["final_survival"] for r in all_records[name]], dtype=float)
        phased = {}
        for ph in PHASE_STEPS:
            got = [r["survival_at"][str(ph)] for r in all_records[name]
                   if str(ph) in r.get("survival_at", {})]
            phased[str(ph)] = float(np.mean(got)) if got else None
        adist = np.array([r.get("action_dist", [0, 0, 0, 0, 0]) for r in all_records[name]],
                         dtype=float)
        summary["arms"][name] = {
            "n": len(vals),
            "mean_final_survival": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean_steps": float(np.mean([r["steps"] for r in all_records[name]])),
            "total_step_one_deaths": int(sum(r["num_step_one_deaths"] for r in all_records[name])),
            "mean_survival_at_step": phased,
            "mean_action_dist": adist.mean(axis=0).tolist() if adist.size else [0] * 5,
        }
    if args.reference:
        if args.reference not in names:
            raise SystemExit(f"reference {args.reference} not among arms")
        for name in names:
            if name == args.reference:
                continue
            diffs = [by_seed[s][name]["final_survival"] - by_seed[s][args.reference]["final_survival"]
                     for s in sorted(by_seed)
                     if name in by_seed[s] and args.reference in by_seed[s]]
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

    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[written] {out_path}\n[written] {out_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
