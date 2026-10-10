#!/usr/bin/env python3
"""v52: frozen-policy input-perturbation ablation on accepted v50 corrected
survival_only FINAL models.

Question
--------
Take three independently accepted v50 corrected ``survival_only`` FINAL policies
(200 updates / 614400 steps; the accepted manifest records ``selected_is_final``
true for rep0 and false for rep1/rep2, but this round keys on the FINAL
checkpoint identity in all cases), frozen deterministic, and feed the same
base-11 observation under three input modes:

  * ``full``               : unchanged base-11 observation.
  * ``mask_predator``      : copy of the row with indices 5:11 zeroed
                             (visible flag, relative pos, predator velocity and
                             distance all hidden). When the predator is already
                             invisible the env writes exactly zeros there, so the
                             mask matches the env's natural encoding.
  * ``mask_velocity_only`` : copy of the row with indices 8:10 zeroed
                             (predator velocity hidden; flag / relative pos /
                             distance retained).

The fish's own position / velocity / boundary channels (0:5) are never touched,
and the real world physics is never modified: the mask is applied only to the
array handed to ``model.predict``, exactly at the production call site. The
world advances from the actions the controller emits, so trajectories may di er.

``rule_flee_lead`` is carried as a positive control with the same three modes.
No HOLD arm is added; HOLD is already the env's default action when the rule
sees no predator, so it is reachable inside ``mask_predator``.

Interpretation limits (fixed before results)
--------------------------------------------
This measures the *sensitivity of the frozen policy input* to removing an
information channel. It does NOT prove the masked information is absolutely
useless, nor that retraining without it is unnecessary. ``mask_velocity_only``
feeds a zero predator velocity that the policy may not have seen during
training, so that arm can be out-of-distribution. If PPO is unchanged while the
rule arm degrades, that is reported as is; no improvement is manufactured.

Provenance
----------
Rule logic is copied from ``experiments/v50/v50_evaluate.py`` (sha256
4a840ad19f4b29588aba13a5fe63c06851770966d3edc1010fbd00355ee2f9e5), itself the
v49/v48 lineage. Env construction and policy tensor hashing are reused from
``v50_common`` / ``v50_verify`` read-only; neither is modified.

Usage
-----
  experiments/v48/.venv/bin/python \
      experiments/v52/current_baseline/v52_eval.py \
      --out experiments/v52/current_baseline/artifacts/raw_episodes.jsonl \
      --manifest experiments/v52/current_baseline/artifacts/v52_spec_manifest.json \
      --workers 4
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
ROOT = HERE.parents[2]
V50 = ROOT / "experiments" / "v50"
for p in (str(ROOT), str(V50)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v50_common as common  # noqa: E402  (read-only, exact v50 env config)
import v50_verify as verify  # noqa: E402  (read-only, policy tensor hash)

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# -- frozen controller / mode configuration ---------------------------------
MODES = ("full", "mask_predator", "mask_velocity_only")

PPO_CONTROLLERS = ("rep0_survival_only_final", "rep1_survival_only_final",
                   "rep2_survival_only_final")
RULE_CONTROLLER = "rule_flee_lead"
CONTROLLERS = PPO_CONTROLLERS + (RULE_CONTROLLER,)

# final_policy_tensor_sha256 is the frozen identity from the accepted v50
# corrected selection manifest (hash_scheme == policy_tensor_sha256).
PPO_MODEL = {
    "rep0_survival_only_final": {
        "checkpoint": "experiments/v50/artifacts/corrected_streams/runs/"
                      "rep0_survival_only/checkpoints/model_final.zip",
        "final_policy_tensor_sha256":
            "aac9610481879e56335e8bc18d3144df0aab6d6ab0c22a3e3c0d77651257eec8",
    },
    "rep1_survival_only_final": {
        "checkpoint": "experiments/v50/artifacts/corrected_streams/runs/"
                      "rep1_survival_only/checkpoints/model_final.zip",
        "final_policy_tensor_sha256":
            "6aeb33ee08a715a87457a03fe9eba16ecfa241b6ed80d766c34f7c968457cc17",
    },
    "rep2_survival_only_final": {
        "checkpoint": "experiments/v50/artifacts/corrected_streams/runs/"
                      "rep2_survival_only/checkpoints/model_final.zip",
        "final_policy_tensor_sha256":
            "f77a0ec8eba6ed3d86e20eecfd6938816633baf5e4bda1677e3276466c3f5cce",
    },
}

# Fresh report bank (never used by v48/v49/v50/v51 training or eval).
SEED_RNG = 520202
N_SCENARIOS = 40
DEBUG_SEED_RNG = 5202012
DEBUG_N_SCENARIOS = 3

PHASE_STEPS = (100, 250, 500)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 520202
ALPHA = 0.05

# -- rule positive control (copied from v50_evaluate.py, see module docstring)
BOUNDARY_AVOID_THRESHOLD = 0.15
HEADING_DEADZONE_DEG = 15.0
VISION = 10.0 / 3.0
FISH_MAX_SPEED = 2.0


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


def rule_flee_lead(obs):
    """v50/v49 ``rule_flee_lead`` logic, applied to an 11-dim base observation."""
    if obs[4] < BOUNDARY_AVOID_THRESHOLD:
        desired = np.array([-float(obs[0]), -float(obs[1])], dtype=np.float64)
    elif obs[5] > 0.5:
        rel = np.array([float(obs[6]), float(obs[7])], dtype=np.float64) * VISION
        pvel = np.array([float(obs[8]), float(obs[9])], dtype=np.float64) * FISH_MAX_SPEED
        pspeed = float(np.linalg.norm(pvel))
        dist = float(np.linalg.norm(rel))
        tau = dist / pspeed if pspeed > 1e-6 else 0.0
        tau = min(tau, 5.0)
        desired = -(rel + pvel * tau)
    else:
        return 4
    return _steer(obs, desired)


# -- input modes -------------------------------------------------------------
def apply_mask(obs, mode):
    """Return a NEW float32 array with ``mode`` applied. Never mutates input."""
    out = np.array(obs, dtype=np.float32, copy=True)
    if mode == "full":
        return out
    if mode == "mask_predator":
        out[..., 5:11] = 0.0
    elif mode == "mask_velocity_only":
        out[..., 8:10] = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")
    return out


class MaskedPolicy:
    """Production call site: applies the input mask, then calls model.predict.

    The array actually handed to ``model.predict`` is ``apply_mask(obs, mode)``,
    captured through the optional observability hook. The real env / model are
    untouched by the mask itself.
    """

    def __init__(self, model, mode, controller_name=None, capture=None):
        self.model = model
        self.mode = mode
        self.controller_name = controller_name
        self.capture = capture

    def predict(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        masked = apply_mask(obs, self.mode)
        if self.capture is not None:
            self.capture(obs, masked)
        batch, _ = self.model.predict(masked, deterministic=True)
        return batch


class RulePolicy:
    """Rule controller at the same production call site (applies the mask)."""

    def __init__(self, mode, controller_name=RULE_CONTROLLER):
        self.mode = mode
        self.controller_name = controller_name

    def predict(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return np.array([rule_controller(obs[i], self.mode) for i in range(len(obs))])


def rule_controller(obs, mode):
    masked = apply_mask(obs, mode)
    return int(rule_flee_lead(masked))


# -- seed banks / validation -------------------------------------------------
def scenario_seeds(rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def validate_scenarios(seeds):
    """Reject empty, missing and duplicate scenario seeds."""
    if not seeds:
        raise ValueError("scenario seed bank is empty")
    if len(set(int(s) for s in seeds)) != len(seeds):
        raise ValueError("scenario seed bank contains duplicate seeds")
    return [int(s) for s in seeds]


def check_model_hashes(model_paths, expected_hashes):
    """Raise unless every checkpoint's loaded policy tensor hash matches frozen."""
    got = {}
    for name, path in model_paths.items():
        h = verify.policy_tensor_sha256_from_zip(str(ROOT / path))
        if name not in expected_hashes:
            raise ValueError(f"no frozen hash recorded for {name}")
        if h != expected_hashes[name]:
            raise ValueError(
                f"model identity hash mismatch for {name}: got {h}, "
                f"frozen {expected_hashes[name]}"
            )
        got[name] = h
    return got


# -- episode bookkeeping -----------------------------------------------------
class _TrackState:
    def __init__(self):
        self.surv_at = {}
        self.action_counts = [0] * 5

    def update(self, env, actions, step, info):
        for ph in PHASE_STEPS:
            if step == ph:
                self.surv_at[ph] = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
        for a in actions:
            if 0 <= int(a) < 5:
                self.action_counts[int(a)] += 1

    def summarise(self):
        total = sum(self.action_counts) or 1
        return {
            "survival_at": {str(k): v for k, v in self.surv_at.items()},
            "action_dist": [c / total for c in self.action_counts],
        }


def _record(controller, mode, seed, index, info, steps, pre_roll, track):
    num_alive = int(info.get("num_alive", 0))
    rec = {
        "controller": controller,
        "mode": mode,
        "scenario_index": int(index),
        "seed": int(seed),
        "final_num_alive": num_alive,
        "final_survival": num_alive / float(common.NUM_FISH),
        "steps": int(steps),
        "first_death_step": int(info.get("first_death_step", -1)),
        "num_step_one_deaths": len(info.get("step_one_deaths") or []),
        "pre_roll_final_speed": float((pre_roll or {}).get("final_speed", 0.0)),
        "pre_roll_final_heading_deg": float((pre_roll or {}).get("final_heading_deg", 0.0)),
    }
    rec.update(track.summarise())
    return rec


def run_episode(mode, env, seed, index, policy=None, max_steps=None):
    """Run one episode through the single production policy call site.

    ``policy`` owns masking and is either the PPO ``MaskedPolicy`` or the rule
    ``RulePolicy``; ``policy.controller_name`` labels the record. ``max_steps``
    is test-only (used to truncate a short diagnostic episode); it does not
    alter the masked call chain.
    """
    if policy is None:
        policy = RulePolicy(mode)
    obs, info = env.reset(seed=seed)
    pre_roll = (info or {}).get("pre_roll_stats")
    track = _TrackState()
    steps = 0
    while True:
        if len(obs) > 0:
            batch = policy.predict(obs)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        track.update(env, actions, steps, info)
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return _record(policy.controller_name, mode, seed, index, info, steps, pre_roll, track)


# -- worker ------------------------------------------------------------------
def _build_env():
    return common.make_base_env(include_neighbor_features=False)


def worker(job):
    import torch

    torch.set_num_threads(1)
    controller, mode, seeds = job
    env = _build_env()
    policy = None
    model = None
    records = []
    try:
        if controller != RULE_CONTROLLER:
            from stable_baselines3 import PPO

            path = ROOT / PPO_MODEL[controller]["checkpoint"]
            model = PPO.load(str(path), device="cpu")
            policy = MaskedPolicy(model, mode, controller_name=controller)
        else:
            policy = RulePolicy(mode)
        for index, seed in enumerate(seeds):
            t0 = time.time()
            rec = run_episode(mode, env, seed, index, policy=policy)
            rec["controller"] = controller
            rec["elapsed_sec"] = time.time() - t0
            records.append(rec)
    finally:
        env.close()
    return f"{controller}|{mode}", records


# -- manifest ----------------------------------------------------------------
def build_manifest(seeds, debug):
    hashes = verify.policy_tensor_sha256_from_zip
    return {
        "kind": "v52_spec_manifest",
        "version": 1,
        "script_sha256": SCRIPT_SHA256,
        "debug": bool(debug),
        "seed_rng": DEBUG_SEED_RNG if debug else SEED_RNG,
        "n_scenarios": len(seeds),
        "scenarios": [int(s) for s in seeds],
        "modes": list(MODES),
        "controllers": list(CONTROLLERS),
        "rule_controller": RULE_CONTROLLER,
        "rule_source_sha256":
            "4a840ad19f4b29588aba13a5fe63c06851770966d3edc1010fbd00355ee2f9e5",
        "env_config": common.env_config(include_neighbor_features=False),
        "neighbor": False,
        "models": {
            name: {
                "checkpoint": PPO_MODEL[name]["checkpoint"],
                "final_policy_tensor_sha256":
                    PPO_MODEL[name]["final_policy_tensor_sha256"],
            }
            for name in PPO_CONTROLLERS
        },
        "hash_scheme": verify.HASH_SCHEME,
        "phase_steps": list(PHASE_STEPS),
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "alpha": ALPHA},
        "note": "Frozen before scoring; non-overwritable. Rule logic copied from "
                "v50_evaluate.py. Interpretation limited to frozen-policy input "
                "sensitivity; mask_velocity_only may be OOD.",
    }


def write_manifest_exclusive(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest missing: {p}")
    m = json.loads(p.read_text())
    if m.get("kind") != "v52_spec_manifest":
        raise ValueError("not a v52 spec manifest")
    return m


# -- analysis ----------------------------------------------------------------
def bootstrap_ci(diffs, n_boot=BOOTSTRAP_N, alpha=ALPHA, seed=BOOTSTRAP_SEED):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def verify_existing_manifest(existing, fresh):
    """Read-back gate for ``--skip-manifest-write``: the existing frozen spec's
    run *inputs* must match this run, so a stale or unrelated spec cannot be
    reused silently. ``script_sha256`` is provenance of the generating code and
    is reported but not required to match (a non-executing doc/refactor edit
    changes it without changing run semantics)."""
    fields = ("kind", "seed_rng", "n_scenarios", "scenarios", "modes",
              "controllers", "rule_controller", "rule_source_sha256",
              "env_config", "neighbor", "models", "hash_scheme")
    for f in fields:
        if existing.get(f) != fresh.get(f):
            raise ValueError(f"existing manifest input {f!r} differs from this run")
    return {
        "existing_script_sha256": existing.get("script_sha256"),
        "current_script_sha256": fresh.get("script_sha256"),
        "script_sha256_matches": existing.get("script_sha256") == fresh.get("script_sha256"),
    }


def analyse(records, n_scenarios):
    # controller|mode -> scenario_index -> final_survival
    table = {}
    for r in records:
        key = (r["controller"], r["mode"])
        table.setdefault(key, {})[int(r["scenario_index"])] = float(r["final_survival"])

    def full_vec(controller):
        d = table.get((controller, "full"), {})
        return np.array([d[i] for i in range(n_scenarios)], dtype=float)

    summary = {"per_controller_mode": {}, "paired_vs_full": {}, "ppo_aggregate": {}}
    for controller in CONTROLLERS:
        for mode in MODES:
            vals = table.get((controller, mode), {})
            arr = np.array([vals[i] for i in range(n_scenarios)], dtype=float)
            full = full_vec(controller)
            summary["per_controller_mode"][f"{controller}|{mode}"] = {
                "n": int(len(arr)),
                "mean_final_survival": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
            if mode != "full":
                diffs = arr - full
                lo, hi = bootstrap_ci(diffs)
                summary["paired_vs_full"][f"{controller}|{mode}"] = {
                    "n_pairs": int(len(diffs)),
                    "mean_diff": float(diffs.mean()),
                    "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "win_rate": float((diffs > 0).mean()),
                    "tie_rate": float((diffs == 0).mean()),
                    "loss_rate": float((diffs < 0).mean()),
                }

    # fixed 3 PPO aggregate: per-scene mean across models, then episode bootstrap
    for mode in ("mask_predator", "mask_velocity_only"):
        per_scene = []
        for i in range(n_scenarios):
            ds = []
            for controller in PPO_CONTROLLERS:
                a = table.get((controller, mode), {}).get(i)
                f = table.get((controller, "full"), {}).get(i)
                if a is not None and f is not None:
                    ds.append(a - f)
            per_scene.append(float(np.mean(ds)) if ds else np.nan)
        arr = np.array(per_scene, dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = bootstrap_ci(arr)
        summary["ppo_aggregate"][mode] = {
            "n_scenes": int(len(arr)),
            "mean_diff": float(arr.mean()) if len(arr) else float("nan"),
            "ci95_low": lo,
            "ci95_high": hi,
        }
    return summary


# -- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--skip-manifest-write", action="store_true")
    args = ap.parse_args()

    if args.debug:
        seeds = validate_scenarios(scenario_seeds(DEBUG_SEED_RNG, DEBUG_N_SCENARIOS))
    else:
        seeds = validate_scenarios(scenario_seeds(SEED_RNG, N_SCENARIOS))

    manifest = build_manifest(seeds, args.debug)
    skip_report = None
    if args.skip_manifest_write:
        existing = load_manifest(args.manifest)
        skip_report = verify_existing_manifest(existing, manifest)
        print(f"[manifest] reusing existing spec with input read-back: {skip_report}")
    else:
        mpath = write_manifest_exclusive(args.manifest, manifest)
        print(f"[manifest] frozen at {mpath}")

    # frozen model identity gate (before any scoring)
    print("[hash gate] verifying accepted v50 corrected final models ...")
    got = check_model_hashes(
        {n: PPO_MODEL[n]["checkpoint"] for n in PPO_CONTROLLERS},
        {n: PPO_MODEL[n]["final_policy_tensor_sha256"] for n in PPO_CONTROLLERS},
    )
    for n, h in got.items():
        print(f"  {n}: {h} OK")

    # jobs: one job per (controller, mode), all scenarios
    jobs = [(c, m, seeds) for c in CONTROLLERS for m in MODES]
    workers = max(1, min(args.workers, len(jobs)))
    print(f"[run] {len(jobs)} jobs, {workers} workers, {len(jobs) * len(seeds)} episodes")

    t0 = time.time()
    by_key = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, recs in ex.map(worker, jobs):
            by_key[key] = recs
    wall = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for key in sorted(by_key):
        records.extend(sorted(by_key[key], key=lambda r: r["scenario_index"]))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "kind": "v52_summary",
        "debug": bool(args.debug),
        "n_scenarios": len(seeds),
        "wall_sec": wall,
        "script_sha256": SCRIPT_SHA256,
        "dependency_snapshot": common.dependency_snapshot(),
        "manifest": manifest,
        "model_tensor_sha256": got,
    }
    summary.update(analyse(records, len(seeds)))
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[written] {out_path}\n[written] {out_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
