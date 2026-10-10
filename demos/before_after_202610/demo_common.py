"""Shared definitions for the before/after same-scene benchmark demo.

This demo compares, on one fresh matched scenario bank and with a byte-identical
physical reset, three controller families:

  * ``old_ppo``       the historical best single checkpoint
                      (``experiments/v45/.../model_iter_20.zip``), 18-dim obs
                      (neighbor features ON);
  * ``new_rep0/1/2``  the three accepted v50 corrected ``survival_only`` FINAL
                      models, 11-dim obs (neighbor features OFF);
  * ``rule_flee_lead`` the accepted hand-written lead-evasion heuristic, reads
                      the 11 base dims only.

Physics, reward, termination and reset are the unchanged v50 world
(``experiments/v50/v50_common.env_config``). The only per-controller env
difference is the ``include_neighbor_features`` flag, which changes the
observation vector length (18 vs 11) and never touches the simulation RNG,
positions, velocities or collision physics. That invariance is asserted by
``initial_state_identity.py`` before any score is produced.

Everything here is deterministic: PPO inference is ``deterministic=True``; the
rule controllers are pure functions of the observation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
V50_DIR = ROOT / "experiments" / "v50"
for _p in (ROOT, V50_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v50_common as v50  # noqa: E402

NUM_FISH = v50.NUM_FISH
MAX_TIMESTEPS = v50.MAX_TIMESTEPS
PHASE_STEPS = (1, 100, 250, 500)

# --- fresh, frozen demo banks ------------------------------------------------
REPORT_BANK = {"rng_seed": 590102, "n": 64}
DEBUG_BANK = {"rng_seed": 590101, "n": 2}

# Prior-round banks (rng_seed, n); used only for a disjointness sanity check.
PRIOR_BANKS = [
    (481000, 2), (481001, 12), (481002, 40), (555001, 40), (555002, 40),
    (482100, 2), (482101, 20), (482102, 40),
    (500200, 2), (500201, 24), (500202, 40),
    (510101, 2), (510102, 40),
    (530100, 2), (530101, 24), (530102, 40),
    (540101, 2), (540102, 40),
    (550101, 2), (550102, 40),
    (560101, 2), (560102, 40),
    (57010101, 2), (57010112, 24), (57010240, 40),
    (580101, 2), (580102, 64),
]

# --- controller definitions --------------------------------------------------
OLD_MODEL_RELPATH = "experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip"
NEW_MODEL_RELPATH = {
    0: "experiments/v50/artifacts/corrected_streams/runs/rep0_survival_only/checkpoints/model_final.zip",
    1: "experiments/v50/artifacts/corrected_streams/runs/rep1_survival_only/checkpoints/model_final.zip",
    2: "experiments/v50/artifacts/corrected_streams/runs/rep2_survival_only/checkpoints/model_final.zip",
}
CORRECTED_SELECTION_MANIFEST_RELPATH = "experiments/v50/artifacts/corrected_streams/results/selection_manifest.json"

# Each policy controller: (name, repo-relative checkpoint, include_neighbor_features)
CONTROLLERS = [
    {"name": "old_ppo", "kind": "policy", "path": OLD_MODEL_RELPATH, "neighbor": True},
    {"name": "new_rep0", "kind": "policy", "path": NEW_MODEL_RELPATH[0], "neighbor": False},
    {"name": "new_rep1", "kind": "policy", "path": NEW_MODEL_RELPATH[1], "neighbor": False},
    {"name": "new_rep2", "kind": "policy", "path": NEW_MODEL_RELPATH[2], "neighbor": False},
    {"name": "rule_flee_lead", "kind": "rule", "rule": "flee_lead", "neighbor": False},
]
CONTROLLER_NAMES = [c["name"] for c in CONTROLLERS]
NEW_REPS = ("new_rep0", "new_rep1", "new_rep2")
RULE_NAME = "rule_flee_lead"


def episode_seeds(rng_seed, n):
    rng = np.random.default_rng(int(rng_seed))
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def report_seeds():
    return episode_seeds(REPORT_BANK["rng_seed"], REPORT_BANK["n"])


def debug_seeds():
    return episode_seeds(DEBUG_BANK["rng_seed"], DEBUG_BANK["n"])


def make_env(include_neighbor_features: bool):
    return v50.make_base_env(include_neighbor_features=include_neighbor_features)


def policy_tensor_sha256_from_zip(path) -> str:
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    state = model.policy.state_dict()
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- rule logic (verbatim from the accepted v50/v53-v58 evaluate.py) ---------
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


def rule_action(rule, obs):
    if rule == "hold":
        return 4
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
                tau = dist / pspeed if pspeed > 1e-6 else 0.0
                tau = min(tau, 5.0)
                desired = -(rel + pvel * tau)
        else:
            return 4
        return _steer(obs, desired)
    raise ValueError(f"unknown rule: {rule}")


def run_episode(env, seed, controller, model):
    """One nominal same-scene episode. Returns a record with phase survival.

    Fixed /96 denominator: the survival at every phase is num_alive/96; a phase
    beyond an early all-death termination is 0. First-step deaths are kept, never
    deducted.
    """
    obs, _info = env.reset(seed=int(seed))
    track = {ph: None for ph in PHASE_STEPS}
    steps = 0
    action_counts = [0] * 5
    while True:
        if len(obs) > 0:
            if controller["kind"] == "policy":
                batch, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch)]
            else:
                actions = [rule_action(controller["rule"], obs[i]) for i in range(len(obs))]
        else:
            actions = []
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        for a in actions:
            if 0 <= int(a) < 5:
                action_counts[int(a)] += 1
        if steps in track:
            track[steps] = int(info.get("num_alive", 0)) / float(NUM_FISH)
        if terminated or truncated:
            break
    num_alive = int(info.get("num_alive", 0))
    surv = {str(ph): float(track[ph] if track[ph] is not None else 0.0) for ph in PHASE_STEPS}
    total = sum(action_counts) or 1
    return {
        "controller": controller["name"],
        "seed": int(seed),
        "final_num_alive": num_alive,
        "final_survival": num_alive / float(NUM_FISH),
        "steps": int(steps),
        "first_death_step": int(info.get("first_death_step", -1)),
        "num_step_one_deaths": len(info.get("step_one_deaths") or []),
        "survival_at": surv,
        "action_dist": [c / total for c in action_counts],
    }
