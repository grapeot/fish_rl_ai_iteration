#!/usr/bin/env python3
"""control_probe: fork-point counterfactual for v48 setting triage.

One question, no training, no env modification:

  At step 100 on the SAME trajectory (identical fish pos/vel/alive, predator
  pos/vel, timestep and RNG state), does continuing the original rule beat
  freezing to HOLD (action 4, which preserves velocity and coasts, NOT a stop)
  for the remaining 400 steps?

Design
------
* Fresh episode seeds from rng=482002 -- disjoint from smoke(481000),
  dev(481001) and report(481002); the historical report number is NOT reused.
* For each seed, two rule families are forked at step 100:
    - flee_lead : reads the visible predator when present, else HOLD;
    - safe_top  : never reads the predator, aims at a fixed world-top point.
  Each family is run to 100, the full env is deep-copied, then branch A
  continues the rule and branch B holds (action 4) for all alive fish.
* Hard assertions per pair: branch start states identical (all fish pos/vel/
  alive, predator pos/vel, timestep, RNG bit-generator state); the two branches
  cannot cross-modify (checked by snapshotting one while the other runs).
* Meaningful test: a separate, un-forked full run of the original rule must
  match the forked-continue result exactly (final alive + death timesteps).
* Deaths are never leaked: actions are built from the observation rows, which
  already correspond 1:1 to alive fish; an episode that dies out early still
  counts in the denominator (final survival 0).
* Third branch (lead only, opt-in via --shielded): from the same step-100 state,
  copy the policy observation and encode the predator as invisible (flag 0,
  position/velocity/distance fields 0) WITHOUT touching the real env. Boundary
  control is retained. This separates threat information from boundary
  correction. Only meaningful once the two lead arms differ enough.

Read-only reuse: imports experiments/v48/evaluate.py (rule_action + fixed
constants) and common.py. Neither is modified. No git writes.

Usage:
  experiments/v48/.venv/bin/python experiments/v48/control_probe.py \
      --episodes 2 --shielded off \
      --out-dir experiments/v48/artifacts/control_probe
"""

import argparse
import copy
import hashlib
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
import evaluate as ev  # noqa: E402

ROOT = HERE.parents[1]

SEED_RNG = 482002
N_SEEDS = 40
SPLIT_STEP = 100
HORIZON = 500
NEAR_STATIC_SPEED = 0.05   # actual speed; near-static, NOT a HOLD proxy
# ev._safe_top gates deceleration on NORMALIZED speed obs[2:4] > 0.03. The
# diagnostic below is kept in ACTUAL (world) speed units for both the guide
# gate and the world-space arrival radius, so the two constants are distinct:
#   normalized threshold 0.03  ->  actual threshold 0.03 * FISH_MAX_SPEED = 0.06
GUIDE_SPEED_NORMALIZED = 0.03          # as written in evaluate.py _safe_top
GUIDE_SPEED = GUIDE_SPEED_NORMALIZED * 2.0   # actual units; FISH_MAX_SPEED = 2.0
BOOT_N = 10000
BOOT_SEED = 482002

# Fixed safe-zone target used by ev.rule_safe_top, expressed in WORLD units so
# it can be compared against env.fish_positions (which are unnormalized).
# ev constants are normalized by STAGE_RADIUS and FISH_MAX_SPEED.
STAGE_RADIUS = 10.0
FISH_MAX_SPEED = 2.0
TARGET = STAGE_RADIUS * np.array(
    [
        ev.SAFE_TARGET_R * np.cos(np.deg2rad(ev.SAFE_TARGET_ANGLE_DEG)),
        ev.SAFE_TARGET_R * np.sin(np.deg2rad(ev.SAFE_TARGET_ANGLE_DEG)),
    ],
    dtype=np.float64,
)
ARRIVE_R = ev.SAFE_ARRIVE_R * STAGE_RADIUS


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #
def seeds_for(n):
    rng = np.random.default_rng(SEED_RNG)
    all_seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=N_SEEDS)]
    return all_seeds[:n]


# --------------------------------------------------------------------------- #
# action builders (read only the observation the policy would see)
# --------------------------------------------------------------------------- #
def rule_actions(rule, obs, rng):
    return [ev.rule_action(rule, obs[i], rng) for i in range(len(obs))]


def hold_actions(obs):
    return [4] * len(obs)


def shielded_actions(obs, rng):
    """Predator-invisible encoding for the policy copy only. Real env untouched."""
    masked = obs.copy()
    masked[:, 5] = 0.0        # visible flag off
    masked[:, 6:11] = 0.0     # rel pos, rel vel, distance -> invisible encoding
    return [ev.rule_action("flee_lead", masked[i], rng) for i in range(len(masked))]


# --------------------------------------------------------------------------- #
# per-episode window metrics
# --------------------------------------------------------------------------- #
class Window:
    """Post-split diagnostics, all computed AFTER env.step's collision pass.

    SURVIVOR-ONLY: every per-step loop below iterates `np.where(env.fish_alive)`,
    so a fish captured on step t contributes nothing on step t -- its distance
    at the moment of capture never enters `min_pred_dist`. Do NOT read these
    numbers as a capture-margin or collision-free proof, and do not use them to
    claim a survival mechanism.

    Fields are per-episode; speed/predator distance are averaged over survivor
    fish-steps then averaged across episodes (not a globally pooled fish-step
    mean). `near_static` uses the pre-set NEAR_STATIC_SPEED=0.05 actual-speed
    cut; the observed ratio is 0 in this batch, so the batch is NOT described as
    near-static under that definition. `safe_park`/`safe_guide` are NOT action
    counts: park is arrival-radius survivor fish-steps with speed <= GUIDE_SPEED
    (actual 0.06, which includes speeds between 0.05 and 0.06), guide is the
    rest inside the radius; motion outside the radius is not counted at all.
    """

    def __init__(self):
        self.act = {1: [0] * 5, 2: [0] * 5, 3: [0] * 5}
        self.speed_sum = 0.0
        self.speed_n = 0
        self.near_static = 0
        self.min_pred_dist = float("inf")
        self.safe_guide = 0   # arrival-radius survivor fish-steps with speed > GUIDE_SPEED
        self.safe_park = 0    # arrival-radius survivor fish-steps with speed <= GUIDE_SPEED
        self._inside = {}
        self._entry_count = {}

    def copy(self):
        w = Window()
        w.act = {k: list(v) for k, v in self.act.items()}
        w.speed_sum = self.speed_sum
        w.speed_n = self.speed_n
        w.near_static = self.near_static
        w.min_pred_dist = self.min_pred_dist
        w.safe_guide = self.safe_guide
        w.safe_park = self.safe_park
        w._inside = dict(self._inside)
        w._entry_count = dict(self._entry_count)
        return w

    def note_pre(self, actions):
        for a in actions:
            a = int(a)
            if 0 <= a < 5:
                self.act[1][a] += 1

    def update(self, step, env, actions):
        if step <= SPLIT_STEP:
            return
        phase = 2 if step <= 250 else 3
        for a in actions:
            a = int(a)
            if 0 <= a < 5:
                self.act[phase][a] += 1
        for i in np.where(env.fish_alive)[0]:
            i = int(i)
            pos = env.fish_positions[i]
            speed = float(np.linalg.norm(env.fish_velocities[i]))
            self.speed_sum += speed
            self.speed_n += 1
            if speed < NEAR_STATIC_SPEED:
                self.near_static += 1
            d = float(np.linalg.norm(pos - env.predator_pos))
            if d < self.min_pred_dist:
                self.min_pred_dist = d
            td = float(np.linalg.norm(pos - TARGET))
            if td <= ARRIVE_R:
                if speed > GUIDE_SPEED:
                    self.safe_guide += 1
                else:
                    self.safe_park += 1
                if not self._inside.get(i, False):
                    self._entry_count[i] = self._entry_count.get(i, 0) + 1
                self._inside[i] = True
            else:
                self._inside[i] = False

    def dist(self, phase):
        total = sum(self.act[phase]) or 1
        return [c / total for c in self.act[phase]]

    def reentries(self):
        return sum(max(0, c - 1) for c in self._entry_count.values())


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def same_state(a, b):
    """Exact equality of fish + predator + timestep + RNG state."""
    if not np.array_equal(a.fish_alive, b.fish_alive):
        return False
    if not np.array_equal(a.fish_positions, b.fish_positions):
        return False
    if not np.array_equal(a.fish_velocities, b.fish_velocities):
        return False
    if not np.array_equal(a.predator_pos, b.predator_pos):
        return False
    if not np.array_equal(a.predator_vel, b.predator_vel):
        return False
    if int(a.timestep) != int(b.timestep):
        return False
    return _rng_state(a) == _rng_state(b)


def _rng_state(env):
    """Canonical, hashable view of gymnasium's np_random bit-generator state."""
    return json.dumps(env.np_random.bit_generator.state, default=int, sort_keys=True)


def snapshot(env):
    return {
        "alive": env.fish_alive.copy(),
        "pos": env.fish_positions.copy(),
        "vel": env.fish_velocities.copy(),
        "ppos": env.predator_pos.copy(),
        "pvel": env.predator_vel.copy(),
        "t": int(env.timestep),
    }


def snapshot_equal(snap, env):
    return (
        np.array_equal(snap["alive"], env.fish_alive)
        and np.array_equal(snap["pos"], env.fish_positions)
        and np.array_equal(snap["vel"], env.fish_velocities)
        and np.array_equal(snap["ppos"], env.predator_pos)
        and np.array_equal(snap["pvel"], env.predator_vel)
        and snap["t"] == int(env.timestep)
    )


def make_split(seed, rule, env_factory):
    """Run `rule` from reset to SPLIT_STEP (or early termination)."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    win = Window()
    rng = np.random.default_rng(seed)
    while env.timestep < SPLIT_STEP and int(env.fish_alive.sum()) > 0:
        actions = rule_actions(rule, obs, rng) if len(obs) > 0 else []
        win.note_pre(actions)
        obs, _, _, _, _ = env.step(actions)
    return env, obs, win


def _continue(env, obs, action_callable, win):
    while env.timestep < HORIZON and int(env.fish_alive.sum()) > 0:
        actions = action_callable(obs) if len(obs) > 0 else []
        obs, _, _, _, _ = env.step(actions)
        win.update(env.timestep, env, actions)


def full_run(seed, rule, env_factory):
    """Independent un-forked run of `rule` for the meaningful replay check."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while env.timestep < HORIZON and int(env.fish_alive.sum()) > 0:
        actions = rule_actions(rule, obs, rng) if len(obs) > 0 else []
        obs, _, _, _, _ = env.step(actions)
    alive = int(env.fish_alive.sum())
    dt = env.fish_death_timesteps.copy()
    env.close()
    return alive, dt


def record(seed, tag, env, alive_split, win):
    final_alive = int(env.fish_alive.sum())
    return {
        "seed": int(seed),
        "control": tag,
        "num_alive_at_split": int(alive_split),
        "final_num_alive": final_alive,
        "new_deaths_100_500": int(alive_split - final_alive),
        "final_survival": final_alive / float(common.NUM_FISH),
        "steps": int(env.timestep),
        "action_dist_1_100": win.dist(1),
        "action_dist_101_250": win.dist(2),
        "action_dist_251_500": win.dist(3),
        "mean_speed_101_500": (win.speed_sum / win.speed_n) if win.speed_n else None,
        "near_static_ratio_101_500": (win.near_static / win.speed_n) if win.speed_n else None,
        "min_pred_dist_101_500": win.min_pred_dist if win.speed_n else None,
        "safe_guide_steps": int(win.safe_guide),
        "safe_park_steps": int(win.safe_park),
        "safe_reentries": int(win.reentries()),
    }


# --------------------------------------------------------------------------- #
# per-seed worker
# --------------------------------------------------------------------------- #
def shielded_only_seed(seed, env_factory):
    """Rebuild the step-100 lead state and run ONLY the shielded third branch."""
    env_split, obs_split, win_split = make_split(seed, "flee_lead", env_factory)
    alive_split = int(env_split.fish_alive.sum())
    snap = snapshot(env_split)
    env_s = copy.deepcopy(env_split)
    obs_s = obs_split.copy()
    win_s = win_split.copy()
    assert same_state(env_s, env_split), f"shielded start state differs (seed {seed})"
    rng_s = np.random.default_rng(seed)
    _continue(env_s, obs_s, lambda o: shielded_actions(o, rng_s), win_s)
    assert snapshot_equal(snap, env_split), f"shielded cross-modified split env (seed {seed})"
    rec = record(seed, "flee_lead_shielded", env_s, alive_split, win_s)
    env_split.close()
    return int(seed), rec


def safe_zone_seed(seed, env_factory):
    """Re-derive safe-zone diagnostics (guide/park/reentries) with corrected
    world units for the lead and safe_top continued/holded branches, forked
    from each family's own bit-identical step-100 state. The shielded arm is
    already recomputed by --shielded-only with the corrected constants."""
    out = []
    for fam in ("flee_lead", "safe_top"):
        env_split, obs_split, win_split = make_split(seed, fam, env_factory)
        alive_split = int(env_split.fish_alive.sum())
        snap = snapshot(env_split)

        env_c = copy.deepcopy(env_split)
        win_c = win_split.copy()
        rng = np.random.default_rng(seed)
        _continue(env_c, obs_split.copy(), lambda o, f=fam: rule_actions(f, o, rng), win_c)
        out.append(record(seed, f"{fam}_continued", env_c, alive_split, win_c))

        env_h = copy.deepcopy(env_split)
        win_h = win_split.copy()
        assert same_state(env_h, env_split), f"{fam} hold start state differs (seed {seed})"
        _continue(env_h, obs_split.copy(), hold_actions, win_h)
        out.append(record(seed, f"{fam}_hold", env_h, alive_split, win_h))
        assert snapshot_equal(snap, env_split), f"{fam} relabel cross-modified split (seed {seed})"
        env_split.close()
    return int(seed), out


def run_safe_zone_pass(args, seeds, out_dir):
    summary_path = out_dir / "control_probe.summary.json"
    jsonl_path = out_dir / "control_probe.jsonl"
    summary = json.loads(summary_path.read_text())
    existing = [json.loads(l) for l in jsonl_path.read_text().splitlines()]

    jobs = [(s, common.make_env) for s in seeds]
    t0 = time.time()
    fresh = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for seed, recs in ex.map(_job_safe, jobs):
            fresh[seed] = {r["control"]: r for r in recs}
    wall = time.time() - t0

    relabel = ("flee_lead_continued", "flee_lead_hold",
               "safe_top_continued", "safe_top_hold")
    for r in existing:
        if r["control"] in relabel and r["seed"] in fresh:
            f = fresh[r["seed"]][r["control"]]
            for k in ("safe_guide_steps", "safe_park_steps", "safe_reentries"):
                r[k] = f[k]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in sorted(existing, key=lambda r: (r["control"], r["seed"])):
            f.write(json.dumps(r) + "\n")

    by_arm = {}
    for r in existing:
        by_arm.setdefault(r["control"], []).append(r)
    for name in relabel:
        if name in by_arm:
            summary["arms"][name] = aggregate(name, by_arm[name])
    summary["safe_zone_relabel_wall_sec"] = wall
    summary["dependency_sha256"]["experiments/v48/control_probe.py"] = sha256_of("experiments/v48/control_probe.py")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"safe_zone_relabel_ran": True, "wall_sec": wall,
                      "arms": {n: summary["arms"][n] for n in relabel if n in summary["arms"]}},
                     indent=2))
    print(f"\n[written] {jsonl_path}\n[written] {summary_path}")


def process_seed(seed, env_factory):
    recs = []
    checks = {}

    # ---- flee_lead fork ----
    env_split, obs_split, win_split = make_split(seed, "flee_lead", env_factory)
    alive_split = int(env_split.fish_alive.sum())
    snap = snapshot(env_split)

    full_alive, full_dt = full_run(seed, "flee_lead", env_factory)

    env_c = copy.deepcopy(env_split)
    obs_c = obs_split.copy()
    win_c = win_split.copy()
    assert same_state(env_c, env_split), f"lead continue start state differs (seed {seed})"
    rng_c = np.random.default_rng(seed)
    _continue(env_c, obs_c, lambda o: rule_actions("flee_lead", o, rng_c), win_c)
    assert snapshot_equal(snap, env_split), f"lead continue cross-modified split env (seed {seed})"

    env_h = copy.deepcopy(env_split)
    obs_h = obs_split.copy()
    win_h = win_split.copy()
    assert same_state(env_h, env_split), f"lead hold start state differs (seed {seed})"
    _continue(env_h, obs_h, hold_actions, win_h)

    recs.append(record(seed, "flee_lead_continued", env_c, alive_split, win_c))
    recs.append(record(seed, "flee_lead_hold", env_h, alive_split, win_h))

    checks["lead_replay_final_alive_match"] = int(env_c.fish_alive.sum()) == full_alive
    checks["lead_replay_death_match"] = bool(np.array_equal(env_c.fish_death_timesteps, full_dt))

    # ---- safe_top fork ----
    env_split2, obs_split2, win_split2 = make_split(seed, "safe_top", env_factory)
    alive_split2 = int(env_split2.fish_alive.sum())
    snap2 = snapshot(env_split2)

    full_alive2, full_dt2 = full_run(seed, "safe_top", env_factory)

    env_c2 = copy.deepcopy(env_split2)
    obs_c2 = obs_split2.copy()
    win_c2 = win_split2.copy()
    assert same_state(env_c2, env_split2), f"safe_top continue start state differs (seed {seed})"
    rng_c2 = np.random.default_rng(seed)
    _continue(env_c2, obs_c2, lambda o: rule_actions("safe_top", o, rng_c2), win_c2)
    assert snapshot_equal(snap2, env_split2), f"safe_top continue cross-modified split env (seed {seed})"

    env_h2 = copy.deepcopy(env_split2)
    obs_h2 = obs_split2.copy()
    win_h2 = win_split2.copy()
    assert same_state(env_h2, env_split2), f"safe_top hold start state differs (seed {seed})"
    _continue(env_h2, obs_h2, hold_actions, win_h2)

    recs.append(record(seed, "safe_top_continued", env_c2, alive_split2, win_c2))
    recs.append(record(seed, "safe_top_hold", env_h2, alive_split2, win_h2))

    checks["safe_top_replay_final_alive_match"] = int(env_c2.fish_alive.sum()) == full_alive2
    checks["safe_top_replay_death_match"] = bool(np.array_equal(env_c2.fish_death_timesteps, full_dt2))

    # ---- shielded third branch is run in a separate pass (see main) ----
    env_split.close()
    env_split2.close()
    return int(seed), recs, checks


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(diffs, n=BOOT_N, seed=BOOT_SEED):
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_stats(diffs):
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return None
    lo, hi = bootstrap_ci(d)
    return {
        "n_pairs": int(d.size),
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "win_rate": float((d > 0).mean()),
        "tie_rate": float((d == 0).mean()),
        "loss_rate": float((d < 0).mean()),
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
    }


def aggregate(name, recs):
    surv = np.array([r["final_survival"] for r in recs], dtype=float)
    deaths = np.array([r["new_deaths_100_500"] for r in recs], dtype=float)
    adist = {p: np.array([r[f"action_dist_{p}"] for r in recs], dtype=float).mean(axis=0).tolist()
             for p in ("1_100", "101_250", "251_500")}
    speeds = [r["mean_speed_101_500"] for r in recs if r["mean_speed_101_500"] is not None]
    near = [r["near_static_ratio_101_500"] for r in recs if r["near_static_ratio_101_500"] is not None]
    mpd = [r["min_pred_dist_101_500"] for r in recs if r["min_pred_dist_101_500"] is not None]
    return {
        "n": len(recs),
        "mean_final_survival": float(surv.mean()) if surv.size else None,
        "std_final_survival": float(surv.std(ddof=1)) if surv.size > 1 else 0.0,
        "min_final_survival": float(surv.min()) if surv.size else None,
        "max_final_survival": float(surv.max()) if surv.size else None,
        "mean_new_deaths_100_500": float(deaths.mean()) if deaths.size else None,
        "mean_action_dist": adist,
        "mean_speed_101_500": float(np.mean(speeds)) if speeds else None,
        "mean_near_static_ratio_101_500": float(np.mean(near)) if near else None,
        "mean_min_pred_dist_101_500": float(np.mean(mpd)) if mpd else None,
        "total_safe_guide_steps": int(sum(r["safe_guide_steps"] for r in recs)),
        "total_safe_park_steps": int(sum(r["safe_park_steps"] for r in recs)),
        "total_safe_reentries": int(sum(r["safe_reentries"] for r in recs)),
    }


def sha256_of(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def refresh_hashes(out_dir):
    """Record the exact stored code version without re-simulating."""
    summary_path = out_dir / "control_probe.summary.json"
    summary = json.loads(summary_path.read_text())
    summary["dependency_sha256"] = {
        "experiments/v48/control_probe.py": sha256_of("experiments/v48/control_probe.py"),
        "experiments/v48/common.py": sha256_of("experiments/v48/common.py"),
        "experiments/v48/evaluate.py": sha256_of("experiments/v48/evaluate.py"),
        "fish_env.py": sha256_of("fish_env.py"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["dependency_sha256"], indent=2))
    print(f"\n[written] {summary_path}")


# Text of the post-review metadata correction. This block only documents a
# METADATA/COMMENT revision; the raw 200 records and every arm/paired number
# are unchanged. The original runtime hash in `dependency_sha256` is preserved
# verbatim and is NOT overwritten by this pass.
POST_REVIEW_CORRECTION = {
    "date": "2026-10-08",
    "scope": "metadata/comments only; no simulation, no rule change, no stats change",
    "records_changed": False,
    "reasons": [
        "guide_speed_threshold relabelled to ACTUAL units 0.06; the normalized "
        "0.03 is recorded separately. The rule and the diagnostic value are unchanged.",
        "metric_semantics added: min_pred_dist is survivor-only (post-collision), "
        "near_static ratio is 0 under the pre-set <0.05 definition, and "
        "safe_park/safe_guide are survivor fish-step labels, not action counts.",
        "env_config_note added: predator_heading_bias is applied but omitted from "
        "the summary view; see common.py for the exact spec.",
    ],
    "runtime_provenance_limitation": (
        "dependency_sha256 is the RAW hash recorded at the last run pass and is "
        "kept verbatim. Because each pass rewrites it and --refresh-hashes can "
        "rewrite it without simulating, a matching hash only shows the stored "
        "version matched the metadata at that moment; it does not prove this "
        "source produced every batch's records or that all replay checks were "
        "runtime hard-asserted. No immutable per-pass source snapshot was kept."
    ),
    "source_sha256_before_this_patch": None,  # filled at run time
    "source_sha256_after_this_patch": None,   # filled at run time
}


def apply_post_review_correction(out_dir):
    """No-simulation metadata correction on an existing summary.

    Corrects the threshold label to actual units, adds metric/field semantics,
    and appends a post_review_metadata_correction block. It deliberately does
    NOT overwrite dependency_sha256 (the raw recorded runtime hash).
    """
    summary_path = out_dir / "control_probe.summary.json"
    summary = json.loads(summary_path.read_text())

    before = summary["dependency_sha256"].get("experiments/v48/control_probe.py")
    now = sha256_of("experiments/v48/control_probe.py")

    summary["guide_speed_threshold"] = GUIDE_SPEED                 # 0.06 actual
    summary["guide_speed_threshold_normalized"] = GUIDE_SPEED_NORMALIZED  # 0.03
    summary["metric_semantics"] = {
        "min_pred_dist_101_500": "SURVIVOR-ONLY post-collision; not a capture margin.",
        "near_static_ratio_101_500": "survivor fish-steps with actual speed < 0.05; observed 0.",
        "safe_park_steps": "arrival-radius survivor fish-steps with speed <= 0.06 actual.",
        "safe_guide_steps": "arrival-radius survivor fish-steps with speed > 0.06 actual.",
    }
    summary["env_config_note"] = (
        "predator_heading_bias is applied by common.env_config() but omitted from "
        "this summary view; see common.py:50-53,98 for the exact spec."
    )
    block = dict(POST_REVIEW_CORRECTION)
    block["source_sha256_before_this_patch"] = before
    block["source_sha256_after_this_patch"] = now
    block["note_on_hash_change"] = (
        "control_probe.py changed by this metadata-only patch; the recorded "
        "runtime hash above is retained verbatim and is not claimed to match "
        "the post-patch source."
    )
    summary["post_review_metadata_correction"] = block
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(block, indent=2))
    print(f"\n[written] {summary_path}")


def run_shielded_pass(args, seeds, out_dir):
    """Standalone shielded third branch: rebuild step-100 states, run shielded,
    then merge the new records into the existing JSONL and summary."""
    summary_path = out_dir / "control_probe.summary.json"
    jsonl_path = out_dir / "control_probe.jsonl"
    summary = json.loads(summary_path.read_text())
    existing = [json.loads(l) for l in jsonl_path.read_text().splitlines()]

    jobs = [(s, common.make_env) for s in seeds]
    t0 = time.time()
    sh_recs = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for seed, rec in ex.map(_job_shielded, jobs):
            sh_recs.append(rec)
    wall = time.time() - t0

    merged = [r for r in existing if r["control"] != "flee_lead_shielded"] + sh_recs
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in sorted(merged, key=lambda r: (r["control"], r["seed"])):
            f.write(json.dumps(r) + "\n")

    by_arm = {}
    for r in merged:
        by_arm.setdefault(r["control"], []).append(r)
    summary["arms"]["flee_lead_shielded"] = aggregate("flee_lead_shielded", by_arm["flee_lead_shielded"])

    by_seed = {}
    for r in merged:
        by_seed.setdefault(r["seed"], {})[r["control"]] = r
    seeds_in = sorted(by_seed)
    for other in ("flee_lead_continued", "flee_lead_hold"):
        diffs = [by_seed[s]["flee_lead_shielded"]["final_survival"]
                 - by_seed[s][other]["final_survival"]
                 for s in seeds_in
                 if "flee_lead_shielded" in by_seed[s] and other in by_seed[s]]
        summary["paired"][f"flee_lead_shielded_vs_{other.replace('flee_lead_', '')}"] = paired_stats(diffs)

    summary["shielded_ran"] = True
    summary["shielded_wall_sec"] = wall
    summary["dependency_sha256"]["experiments/v48/control_probe.py"] = sha256_of("experiments/v48/control_probe.py")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"shielded_ran": True, "n": len(sh_recs), "wall_sec": wall,
                      "arm": summary["arms"]["flee_lead_shielded"]}, indent=2))
    print(f"\n[written] {jsonl_path}\n[written] {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=N_SEEDS,
                    help="number of the 40 fresh seeds to run (throughput first)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shielded", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--shielded-only", action="store_true",
                    help="run ONLY the shielded third branch (rebuilds step100), "
                         "merging into the existing summary/JSONL")
    ap.add_argument("--relabel-safe", action="store_true",
                    help="recompute the safe_top residual-guidance labels with "
                         "corrected world units and merge into the summary/JSONL")
    ap.add_argument("--refresh-hashes", action="store_true",
                    help="recompute dependency_sha256 in an existing summary "
                         "(no simulation; records the exact stored code version)")
    ap.add_argument("--apply-post-review-correction", action="store_true",
                    help="metadata-only fix on an existing summary: correct the "
                         "threshold label to actual units, add field semantics, and "
                         "append post_review_metadata_correction; no simulation, does "
                         "not overwrite the recorded runtime hash")
    ap.add_argument("--out-dir", default="experiments/v48/artifacts/control_probe")
    args = ap.parse_args()

    seeds = seeds_for(args.episodes)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shielded_only:
        return run_shielded_pass(args, seeds, out_dir)
    if args.relabel_safe:
        return run_safe_zone_pass(args, seeds, out_dir)
    if args.refresh_hashes:
        return refresh_hashes(out_dir)
    if args.apply_post_review_correction:
        return apply_post_review_correction(out_dir)

    do_shielded = args.shielded == "on"

    jobs = [(s, common.make_env) for s in seeds]
    t0 = time.time()
    all_recs = []
    all_checks = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for seed, recs, checks in ex.map(_job, jobs):
            all_recs.extend(recs)
            all_checks[str(seed)] = checks
    wall = time.time() - t0

    by_arm = {}
    for r in all_recs:
        by_arm.setdefault(r["control"], []).append(r)
    by_seed = {}
    for r in all_recs:
        by_seed.setdefault(r["seed"], {})[r["control"]] = r

    summary = {
        "probe": "control_probe",
        "seed_rng": SEED_RNG,
        "n_episodes": len(seeds),
        "n_seeds_available": N_SEEDS,
        "split_step": SPLIT_STEP,
        "horizon": HORIZON,
        "near_static_speed_threshold": NEAR_STATIC_SPEED,
        "guide_speed_threshold": GUIDE_SPEED,  # ACTUAL units = 0.06
        "guide_speed_threshold_normalized": GUIDE_SPEED_NORMALIZED,  # 0.03, as in evaluate.py
        "shielded_requested": args.shielded,
        "shielded_ran": do_shielded,
        "workers": args.workers,
        "wall_sec": wall,
        "seeds": seeds,
        "dependency_sha256": {
            "experiments/v48/control_probe.py": sha256_of("experiments/v48/control_probe.py"),
            "experiments/v48/common.py": sha256_of("experiments/v48/common.py"),
            "experiments/v48/evaluate.py": sha256_of("experiments/v48/evaluate.py"),
            "fish_env.py": sha256_of("fish_env.py"),
        },
        "dependency_snapshot": common.dependency_snapshot(),
        "env_config": {k: v for k, v in common.env_config().items() if k != "predator_heading_bias"},
        "env_config_note": (
            "predator_heading_bias is intentionally omitted from this summary but "
            "IS applied by common.env_config(); see common.py:50-53,98 for the exact spec."
        ),
        "metric_semantics": {
            "min_pred_dist_101_500": (
                "SURVIVOR-ONLY: computed after env.step collisions; a fish captured "
                "on step t contributes nothing on step t. Not a capture margin, not "
                "a collision-free proof, not a survival-mechanism claim."
            ),
            "near_static_ratio_101_500": (
                "fraction of survivor fish-steps with actual speed < 0.05; observed 0 "
                "in this batch, so the batch is NOT described as near-static."
            ),
            "safe_park_steps": (
                "arrival-radius survivor fish-steps with speed <= 0.06 (actual); NOT "
                "an action count and NOT a strict-stop count."
            ),
            "safe_guide_steps": (
                "arrival-radius survivor fish-steps with speed > 0.06; NOT an action "
                "count and does not count turns taken outside the radius."
            ),
        },
        "arms": {name: aggregate(name, recs) for name, recs in by_arm.items()},
        "paired": {},
        "assertions": {
            "all_pairs_passed": all(
                c.get("lead_replay_final_alive_match", False)
                and c.get("lead_replay_death_match", False)
                and c.get("safe_top_replay_final_alive_match", False)
                and c.get("safe_top_replay_death_match", False)
                for c in all_checks.values()
            ),
            "per_seed": all_checks,
        },
    }

    for fam in ("flee_lead", "safe_top"):
        cont, hold = f"{fam}_continued", f"{fam}_hold"
        diffs = []
        for s in seeds:
            row = by_seed.get(s, {})
            if cont in row and hold in row:
                diffs.append(row[cont]["final_survival"] - row[hold]["final_survival"])
        summary["paired"][f"{fam}_continued_vs_hold"] = paired_stats(diffs)

    # upgrade condition: lead continued significantly beats HOLD
    lead = summary["paired"].get("flee_lead_continued_vs_hold")
    cond = {
        "rule": "lead continued beats HOLD: mean_diff>0 and ci95_low>0",
        "mean_diff": lead["mean_diff"] if lead else None,
        "ci95_low": lead["ci95_low"] if lead else None,
        "met": bool(lead and lead["mean_diff"] > 0 and lead["ci95_low"] > 0),
    }
    summary["upgrade_condition"] = cond

    summary_path = out_dir / "control_probe.summary.json"
    jsonl_path = out_dir / "control_probe.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in sorted(all_recs, key=lambda r: (r["control"], r["seed"])):
            f.write(json.dumps(r) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in
                      ("seed_rng", "n_episodes", "split_step", "wall_sec",
                       "shielded_ran", "upgrade_condition", "assertions")}, indent=2))
    print("\npaired:")
    for k, v in summary["paired"].items():
        print(f"  {k}: {v}")
    print(f"\n[written] {jsonl_path}\n[written] {summary_path}")

    # Third branch is a separate pass so each command stays inside budget and
    # the upgrade condition is recorded before it runs.
    if args.shielded == "on" or (args.shielded == "auto" and cond["met"]):
        print("\n[shielded] running third branch (predator input masked, boundary kept)")
        run_shielded_pass(args, seeds, out_dir)


def _job(job):
    import torch

    torch.set_num_threads(1)
    seed, env_factory = job
    return process_seed(seed, env_factory)


def _job_shielded(job):
    import torch

    torch.set_num_threads(1)
    seed, env_factory = job
    return shielded_only_seed(seed, env_factory)


def _job_safe(job):
    import torch

    torch.set_num_threads(1)
    seed, env_factory = job
    return safe_zone_seed(seed, env_factory)


if __name__ == "__main__":
    main()
