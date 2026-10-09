#!/usr/bin/env python3
"""v56 evaluation: paired multi-controller, multi-condition survival probe on the
frozen v56 scenario banks.

For every fresh scenario seed:
  1. reset the unchanged v50 `FishEscapeEnv` into the nominal world
     (`initial_escape_boost=True, escape_boost_speed=0.8`, full heading/speed bias
     and pre-roll), snapshot it;
  2. for each condition {scale0p75, scale1p00, scale1p25}: restore the snapshot,
     scale ONLY the predator velocity vector by the positive factor, recompute
     observations, then drive the controller to episode end.

Controllers:
  * the three frozen v50 corrected `survival_only` FINAL policies, run with
    per-fish deterministic inference over the base 96-fish env (shared policy,
    neighbor off, 11-dim obs);
  * rule_hold / rule_flee_lead / rule_safe_top, reading ONLY the 11 base dims
    (no hidden predator leak).

The statistical unit is the episode (scenario); the 96 fish are NOT independent
samples. No episode is dropped: a scenario where the whole population dies is kept
with a fixed /96 denominator. Survival is recorded at steps 1/100/250/500; a phase
beyond an early all-death termination is 0 (fixed denominator).

The condition's applied/effective predator speed is recorded from the state (not
from an action or a nominal field) via the nominal vs applied predator speed and the
zero-norm guard flag. The nominal initial predator speed is recorded as a
pre-intervention field, never as the intervention value.

Usage:
  experiments/v48/.venv/bin/python experiments/v56/v56_evaluate.py \
      --seeds report --workers 4 \
      --arm rep0_ppo=.../rep0_survival_only/checkpoints/model_final.zip \
      --arm rep1_ppo=... --arm rep2_ppo=... \
      --arm rule_hold --arm rule_flee_lead --arm rule_safe_top \
      --out experiments/v56/artifacts/results/report.jsonl \
      --require-manifest experiments/v56/artifacts/frozen_config/v56_frozen_manifest.json
"""

import argparse
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

import v56_common as common  # noqa: E402
import v56_env as venv  # noqa: E402
import v56_verify as verify  # noqa: E402

# Rule logic identical to v49/v50/v53/v54/v55 evaluate.py (reads only the base 11 dims).
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
        self.action_counts = [0] * 5

    def update(self, info, actions, step):
        for ph in PHASE_STEPS:
            if step == ph:
                self.surv_at[ph] = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
        for a in actions:
            if 0 <= int(a) < 5:
                self.action_counts[int(a)] += 1

    def summarise(self):
        total = sum(self.action_counts) or 1
        # a phase beyond an early all-death termination is 0 (fixed /96 denominator)
        surv = {str(ph): float(self.surv_at.get(ph, 0.0)) for ph in PHASE_STEPS}
        return {
            "survival_at": surv,
            "action_dist": [c / total for c in self.action_counts],
        }


def build_controller(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
        return {"name": name, "kind": "policy", "path": path}
    if spec.startswith("rule_"):
        return {"name": spec, "kind": "rule", "rule": spec[len("rule_"):]}
    raise ValueError(f"cannot parse controller spec: {spec}")


def _run_one(env, seed, condition, controller, model):
    """Run one episode for (seed, condition, controller). Returns a record."""
    obs, snap = venv.reset_nominal(env, seed)
    obs, cond_info = venv.apply_condition(env, snap, condition)
    rng = np.random.default_rng(seed * 2654435761 % (2 ** 31))
    steps = 0
    track = _TrackState()
    while True:
        if len(obs) > 0:
            if controller["kind"] == "policy":
                batch, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch)]
            else:
                actions = [rule_action(controller["rule"], obs[i], rng) for i in range(len(obs))]
        else:
            actions = []
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        track.update(info, actions, steps)
        if terminated or truncated:
            break
    summary = track.summarise()
    num_alive = int(info.get("num_alive", 0))
    rec = {
        "controller": controller["name"],
        "condition": None,
        "seed": int(seed),
        "final_num_alive": num_alive,
        "final_survival": num_alive / float(common.NUM_FISH),
        "steps": int(steps),
        "first_death_step": int(info.get("first_death_step", -1)),
        "num_step_one_deaths": len(info.get("step_one_deaths") or []),
        # pre-intervention descriptor: the nominal post-reset predator speed field
        "nominal_predator_speed": cond_info["nominal_predator_speed"],
        # effective intervention state: the applied predator speed after scaling
        "applied_predator_speed": cond_info["applied_predator_speed"],
        "zero_predator_norm": bool(cond_info["zero_predator_norm"]),
        "mean_init_fish_speed": float(np.linalg.norm(snap["fish_velocities"], axis=1).mean()),
    }
    rec.update(summary)
    return rec


def worker(job):
    import torch

    torch.set_num_threads(1)
    controller, seeds, conditions = job
    env = common.make_base_env(include_neighbor_features=False)
    model = None
    if controller["kind"] == "policy":
        from stable_baselines3 import PPO

        model = PPO.load(controller["path"], device="cpu")
    records = []
    try:
        for seed in seeds:
            for cond in conditions:
                t0 = time.time()
                rec = _run_one(env, seed, cond, controller, model)
                rec["condition"] = cond
                rec["elapsed_sec"] = time.time() - t0
                records.append(rec)
    finally:
        env.close()
    return controller["name"], records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="report")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-manifest", default=None)
    args = ap.parse_args()

    seeds = common.resolve_seeds(args.seeds)
    controllers = [build_controller(s) for s in args.arm]
    names = [c["name"] for c in controllers]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate controller names")

    actual_env_config = common.env_config(False)

    controller_paths, controller_hashes = {}, {}
    for c in controllers:
        if c["kind"] == "policy":
            controller_paths[c["name"]] = c["path"]
            controller_hashes[c["name"]] = verify.policy_tensor_sha256_from_zip(c["path"])

    if args.require_manifest:
        try:
            manifest = verify.load_manifest(args.require_manifest)
            verify.validate_controller_binding(
                manifest,
                controller_paths=controller_paths,
                controller_hashes=controller_hashes,
                actual_env_config=actual_env_config,
                conditions=common.CONDITIONS,
                seeds_name=args.seeds,
                actual_seeds=seeds,
                speed_factors=common.SPEED_FACTORS,
            )
        except verify.ManifestError as exc:
            raise SystemExit(f"refusing to run: frozen-manifest gate failed: {exc}")
        print(f"[frozen gate] OK against {args.require_manifest}")

    jobs = []
    n_chunks = max(1, min(args.workers, len(seeds)))
    chunks = [list(c) for c in np.array_split(np.array(seeds), n_chunks)]
    for c in controllers:
        for ch in chunks:
            jobs.append((c, [int(x) for x in ch], list(common.CONDITIONS)))

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
            for rec in sorted(all_records[name], key=lambda r: (r["seed"], r["condition"])):
                f.write(json.dumps(rec) + "\n")

    summary = {
        "seed_set": args.seeds,
        "n_seeds": len(seeds),
        "conditions": list(common.CONDITIONS),
        "speed_factors": common.SPEED_FACTORS,
        "workers": args.workers,
        "wall_sec": wall,
        "dependency_snapshot": common.dependency_snapshot(),
        "env_config": actual_env_config,
        "controller_model_sha256": controller_hashes,
        "controllers": {},
        "source_sha256": common.evaluation_source_sha256(),
    }
    for name in names:
        for cond in common.CONDITIONS:
            recs = [r for r in all_records[name] if r["condition"] == cond]
            vals = np.array([r["final_survival"] for r in recs], dtype=float)
            phased = {
                str(ph): float(np.mean([r["survival_at"][str(ph)] for r in recs]))
                for ph in PHASE_STEPS
            }
            adist = np.array([r.get("action_dist", [0] * 5) for r in recs], dtype=float)
            applied = np.array([r["applied_predator_speed"] for r in recs], dtype=float)
            nominal = np.array([r["nominal_predator_speed"] for r in recs], dtype=float)
            summary["controllers"].setdefault(name, {})[cond] = {
                "n": len(vals),
                "mean_final_survival": float(vals.mean()) if len(vals) else None,
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()) if len(vals) else None,
                "max": float(vals.max()) if len(vals) else None,
                "mean_steps": float(np.mean([r["steps"] for r in recs])) if recs else None,
                "total_step_one_deaths": int(sum(r["num_step_one_deaths"] for r in recs)),
                "zero_predator_norm_count": int(sum(bool(r["zero_predator_norm"]) for r in recs)),
                "mean_nominal_predator_speed": float(nominal.mean()) if nominal.size else None,
                "mean_applied_predator_speed": float(applied.mean()) if applied.size else None,
                "mean_survival_at_step": phased,
                "mean_action_dist": adist.mean(axis=0).tolist() if adist.size else [0] * 5,
            }

    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["controllers"], indent=2))
    print(f"\n[written] {out_path}\n[written] {out_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
