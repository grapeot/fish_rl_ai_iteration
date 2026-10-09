#!/usr/bin/env python3
"""v51 action-selection ablation evaluator (round 3 of 10).

Question: on the FROZEN v49 policies, does deterministic argmax versus sampling
from the policy distribution change survival, and how does that difference move
between the selection-stage and final checkpoints? The answer this round is:
no universal stochastic benefit (the point-estimate sign is model-specific), but
for specific models sampling does change performance. See dev_v51.md.

Design (fixed before looking at results)
----------------------------------------
* Six frozen v49 models: three runs x {selection-stage, final}. The exact update
  counts (149/99/49 and 200/200/200) are pinned in `v51_common.MANIFEST_MODELS`
  and re-verified from each zip before the run (fail-closed).
* 40 fresh scenarios from bank 510102 (the 2-episode debug uses 510101). No v49
  scenario bank and no v49 result file is an input.
* Two action-selection arms per model:
    - ``det``  : `model.predict(obs, deterministic=True)` (argmax).
    - ``stoch``: `model.predict(obs, deterministic=False)` (sample from the
      policy's own categorical distribution), one draw per alive fish row.
  The stochastic arm runs `N_ACTION_STREAMS` fixed replicates per scenario; each
  replicate is a SeedSequence-derived action-stream seed. Within a scenario the
  replicates are averaged into ONE stochastic value before the paired inference;
  the raw per-replicate values are also kept so the across-stream spread stays
  inspectable. Only 3 fixed streams were run: their mean-difference signs agree
  across the 40 scenarios, which supports limited repeatability, but this does
  NOT establish that the effect is free of lucky-trajectory influence or stable
  over a broad range of action RNGs.
* Rule anchors `rule_hold` and `rule_flee_lead` are run on the SAME scenarios
  (deterministic, no action RNG) as light same-scenario references.

RNG hygiene (the load-bearing detail)
-------------------------------------
The world RNG (`FishEscapeEnv.np_random`) is seeded EXPLICITLY per episode via
`env.reset(seed=scenario_seed)`. The stochastic arm reseeds the global torch RNG
exactly once per episode (`torch.manual_seed(action_seed)`) immediately after the
reset and before the step loop, and never per step. Because the env reset uses a
numpy Generator seeded independently, consuming torch RNG for action sampling
cannot alter the world's initial state; and because the reset seed is the
scenario seed (never the action seed), the action stream cannot change the world
either. The deterministic arm never seeds torch from the action seed at all, so
it is independent of the action-stream seed by construction.

Statistical unit: the episode. Paired diff ``mean_stoch - det`` per model uses an
episode bootstrap CI over the 40 scenarios. This conditions on the six FIXED
models and on the fixed action-sampling scheme; it contains NO training
uncertainty (the models are frozen and the three training runs are not resampled).
The 3-run dispersion is descriptive only.

Usage (timing/debug):
  experiments/v48/.venv/bin/python experiments/v51/v51_eval.py \
      --scenarios debug --models run1_sel --arms det stoch --workers 2 \
      --out experiments/v51/artifacts/results/debug.jsonl
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

import v51_common as common  # noqa: E402

# ---------------------------------------------------------------------------
# Rule anchors (logic identical to v49/v50 evaluate.py; inlined to avoid a
# second module named `evaluate` colliding in spawned workers).
# ---------------------------------------------------------------------------
BOUNDARY_AVOID_THRESHOLD = 0.15
HEADING_DEADZONE_DEG = 15.0
VISION = 10.0 / 3.0
FISH_MAX_SPEED = 2.0
PHASE_STEPS = (1, 100, 250, 500)
DEATH_PHASE_BUCKETS = ((1, 1), (2, 100), (101, 250), (251, 500))


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


def rule_action(rule, obs, rng):
    if rule == "hold":
        return 4
    if rule in ("flee", "flee_lead", "flee_orbit"):
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
                tau = dist / pspeed if pspeed > 1e-6 else 0.0
                tau = min(tau, 5.0)
                desired = -(rel + pvel * tau)
        else:
            return 4
        return _steer(obs, desired)
    raise ValueError(f"unknown rule: {rule}")


class _Track:
    """Per-episode world-level diagnostics.

    `post_collision_survivor_speed` is a SURVIVOR-CONDITIONAL metric: it averages
    the speed of the fish still alive at each step where at least one fish died,
    i.e. over a selected subset, NOT over all individuals. It is not a per-fish
    causal measure.
    """

    __slots__ = ("surv_at", "action_counts", "survivor_speed_sum", "survivor_speed_n",
                 "post_collision_speed_sum", "post_collision_speed_n", "deaths_at",
                 "death_phase_counts", "prev_alive")

    def __init__(self, initial_alive):
        self.surv_at = {}
        self.action_counts = [0] * 5
        self.survivor_speed_sum = 0.0
        self.survivor_speed_n = 0
        self.post_collision_speed_sum = 0.0
        self.post_collision_speed_n = 0
        self.deaths_at = []
        self.death_phase_counts = {f"{lo}-{hi}": 0 for lo, hi in DEATH_PHASE_BUCKETS}
        self.prev_alive = int(initial_alive)

    def update(self, env, actions, step, info):
        num_alive = int(info.get("num_alive", 0))
        for ph in PHASE_STEPS:
            if step == ph:
                self.surv_at[ph] = num_alive / float(common.NUM_FISH)
        for a in actions:
            if 0 <= int(a) < 5:
                self.action_counts[int(a)] += 1
        if num_alive < self.prev_alive:
            died = self.prev_alive - num_alive
            self.deaths_at.append(step)
            for lo, hi in DEATH_PHASE_BUCKETS:
                if lo <= step <= hi:
                    self.death_phase_counts[f"{lo}-{hi}"] += died
            alive_idx = np.where(env.fish_alive)[0]
            if alive_idx.size:
                sp = np.linalg.norm(env.fish_velocities[alive_idx], axis=1)
                self.post_collision_speed_sum += float(sp.sum())
                self.post_collision_speed_n += int(sp.size)
        self.prev_alive = num_alive
        alive_idx = np.where(env.fish_alive)[0]
        if alive_idx.size:
            sp = np.linalg.norm(env.fish_velocities[alive_idx], axis=1)
            self.survivor_speed_sum += float(sp.sum())
            self.survivor_speed_n += int(sp.size)

    def summarise(self):
        total = sum(self.action_counts) or 1
        return {
            "survival_at": {str(k): v for k, v in self.surv_at.items()},
            "action_dist": [c / total for c in self.action_counts],
            "mean_survivor_speed": (self.survivor_speed_sum / self.survivor_speed_n
                                    if self.survivor_speed_n else None),
            "post_collision_survivor_speed": (self.post_collision_speed_sum /
                                              self.post_collision_speed_n
                                              if self.post_collision_speed_n else None),
            "post_collision_survivor_samples": self.post_collision_speed_n,
            "death_steps": self.deaths_at,
            "death_phase_counts": self.death_phase_counts,
        }


def _record(scenario_index, scenario_seed, info, steps, track):
    num_alive = int(info.get("num_alive", 0))
    rec = {
        "scenario_index": int(scenario_index),
        "scenario_seed": int(scenario_seed),
        "final_num_alive": num_alive,
        "final_survival": num_alive / float(common.NUM_FISH),
        "steps": int(steps),
        "first_death_step": int(info.get("first_death_step", -1)),
        "num_step_one_deaths": len(info.get("step_one_deaths") or []),
    }
    rec.update(track.summarise())
    return rec


def run_episode_policy(model, env, scenario_seed, mode, action_seed=None):
    import torch

    # World RNG is pinned first, from the SCENARIO seed only. The stochastic arm
    # then seeds the global torch RNG exactly once for this episode (never per
    # step); the world's independent np_random is untouched by that draw.
    obs, info = env.reset(seed=scenario_seed)
    if mode == "stoch":
        torch.manual_seed(int(action_seed))
    steps = 0
    track = _Track(env.fish_alive.sum())
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=(mode == "det"))
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        track.update(env, actions, steps, info)
        if terminated or truncated:
            break
    return _record(0, scenario_seed, info, steps, track)


def run_episode_rule(env, scenario_seed, rule):
    obs, info = env.reset(seed=scenario_seed)
    rng = np.random.default_rng(0)
    steps = 0
    track = _Track(env.fish_alive.sum())
    while True:
        if len(obs) > 0:
            actions = [rule_action(rule, obs[i], rng) for i in range(len(obs))]
        else:
            actions = []
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        track.update(env, actions, steps, info)
        if terminated or truncated:
            break
    return _record(0, scenario_seed, info, steps, track)


def _make_base_env(neighbor):
    return common.make_base_env(include_neighbor_features=neighbor)


def worker(job):
    import torch

    torch.set_num_threads(1)
    kind = job["kind"]
    neighbor = job["neighbor"]
    env = _make_base_env(neighbor)
    records = []
    model = None
    action_seeds = job.get("action_seeds")
    try:
        if kind in ("det", "stoch"):
            from stable_baselines3 import PPO

            model = PPO.load(job["model_path"], device="cpu")
        for pos, (ci, cs) in enumerate(zip(job["scenario_indices"], job["scenario_seeds"])):
            t0 = time.time()
            if kind == "rule":
                rec = run_episode_rule(env, cs, job["rule"])
            elif kind == "det":
                rec = run_episode_policy(model, env, cs, "det")
            else:  # stoch: one action seed per episode (repeated in debug mode)
                aseed = action_seeds[pos]
                rec = run_episode_policy(model, env, cs, "stoch", action_seed=aseed)
            rec["scenario_index"] = int(ci)
            rec["scenario_seed"] = int(cs)
            rec["elapsed_sec"] = time.time() - t0
            records.append(rec)
    finally:
        env.close()
    return kind, records


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=510301):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _chunks(items, n):
    n = max(1, min(n, len(items)))
    return [list(c) for c in np.array_split(np.array(items), n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", choices=("debug", "fresh"), default="fresh")
    ap.add_argument("--models", nargs="+", default=list(common.MODEL_NAMES))
    ap.add_argument("--arms", nargs="+", choices=("det", "stoch", "rule"), default=["det", "stoch"])
    ap.add_argument("--rules", nargs="+", default=["hold", "flee_lead"])
    ap.add_argument("--neighbor", choices=("on", "off"), default="off")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-out", default=None)
    args = ap.parse_args()

    neighbor = args.neighbor == "on"
    models_root = ROOT

    manifest = common.build_manifest(models_root)
    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2))

    if args.scenarios == "debug":
        seeds = common.episode_seeds(common.DEBUG_SCENARIO_RNG_SEED, common.N_DEBUG_SCENARIOS)
        reps = common.REPLICATE_INDEX[:2]
        action_seed_fn = common.debug_action_stream_seed
    else:
        seeds = common.episode_seeds(common.SCENARIO_RNG_SEED, common.N_SCENARIOS)
        reps = common.REPLICATE_INDEX
        action_seed_fn = common.action_stream_seed
    indices = list(range(len(seeds)))

    model_by_name = {m["name"]: m for m in manifest["models"]}

    jobs = []
    if "det" in args.arms:
        for name in args.models:
            for ci in _chunks(indices, args.workers):
                jobs.append({"kind": "det", "neighbor": neighbor,
                             "model_path": str(models_root / model_by_name[name]["path"]),
                             "model_name": name,
                             "scenario_indices": ci,
                             "scenario_seeds": [seeds[i] for i in ci]})
    if "stoch" in args.arms:
        for name in args.models:
            for rep in reps:
                for ci in _chunks(indices, args.workers):
                    if args.scenarios == "debug":
                        aseeds = [action_seed_fn(rep) for _ in ci]
                    else:
                        aseeds = [common.action_stream_seed(i, rep) for i in ci]
                    jobs.append({"kind": "stoch", "neighbor": neighbor,
                                 "model_path": str(models_root / model_by_name[name]["path"]),
                                 "model_name": name, "replicate": rep,
                                 "action_seeds": aseeds,
                                 "scenario_indices": ci,
                                 "scenario_seeds": [seeds[i] for i in ci]})
    if "rule" in args.arms:
        for rule in args.rules:
            for ci in _chunks(indices, args.workers):
                jobs.append({"kind": "rule", "neighbor": neighbor, "rule": rule,
                             "scenario_indices": ci,
                             "scenario_seeds": [seeds[i] for i in ci]})

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        results = list(ex.map(worker, jobs))
    wall = time.time() - t0

    rows = []
    for j, (_kind, recs) in zip(jobs, results):
        for r in recs:
            r["arm"] = j["kind"]
            if _kind == "det":
                r["model"] = j["model_name"]
            elif _kind == "stoch":
                r["model"] = j["model_name"]
                r["replicate"] = j["replicate"]
            else:
                r["model"] = "rule_" + j["rule"]
            rows.append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[written] {out_path}  ({len(rows)} episode rows, wall {wall:.1f}s)")
    return rows, manifest, wall


if __name__ == "__main__":
    main()
