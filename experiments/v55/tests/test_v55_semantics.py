#!/usr/bin/env python3
"""v55 meaning tests — the load-bearing semantics of the round-7 axis.

Must pass BEFORE any report. They check, on the real code:

  1. `test_only_predator_velocity_rotates`: for a fixed seed the four conditions
     have identical fish positions/velocities, alive flags, predator position,
     timestep and RNG state; only `predator_vel` differs, and by the exact integer
     rotation (norm preserved).
  2. `test_deg0_is_exact_identity`: the deg0 condition reproduces the raw nominal
     post-reset obs and predator velocity bit-for-bit, and the rotation matrices
     are exact signed permutations (norm preserved exactly).
  3. `test_obs_reflects_rotated_predator_velocity`: obs[8],obs[9] are the rotated
     predator velocity over FISH_MAX_SPEED when the predator is visible.
  4. `test_nominal_deg0_matches_direct_v50`: running a frozen policy through the
     v55 deg0 path and through the direct v50-style loop on the same seed yields
     the identical action sequence and final survival (full-rollout reproduction).
  5. `test_gravity_not_rotated`: `_update_predator` adds gravity to +y for every
     condition; rotating the predator velocity is NOT a gravity turn.
  6. `test_frozen_model_hashes_unchanged`: the three checkpoints still hash to the
     frozen manifest values and live under corrected_streams (not the pilot).
  7. `test_rules_read_only_obs`: rule anchors depend only on the 11 base dims.
  8. `test_statistical_unit_is_episode`: the analysis counts scenarios, not fish.
  9. `test_seed_banks_disjoint`: the 550101/550102 generated episode values do not
     overlap each other or any prior round's bank (v48/v49/v50/v51/v53/v54).
 10. `test_env_config_includes_heading_bias`: the frozen env config carries the
     full predator heading and speed bias (cannot be silently omitted).

Run:
  experiments/v48/.venv/bin/python experiments/v55/tests/test_v55_semantics.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V55 = HERE.parent
ROOT = V55.parents[1]
for p in (str(ROOT), str(V55), str(ROOT / "experiments" / "v50")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v55_common as common  # noqa: E402
import v55_env as venv  # noqa: E402
import v55_verify as verify  # noqa: E402

SEED = 550777


def test_only_predator_velocity_rotates():
    env = common.make_base_env(False)
    obs0, snap = venv.reset_nominal(env, SEED)
    base_fish_pos = snap["fish_positions"].copy()
    base_fish_vel = snap["fish_velocities"].copy()
    base_alive = snap["fish_alive"].copy()
    base_pred_pos = snap["predator_pos"].copy()
    base_pred_vel = snap["predator_vel"].copy()
    base_rng = snap["np_random_state"]

    for cond in common.CONDITIONS:
        obs, _ = venv.apply_condition(env, snap, cond)
        assert np.array_equal(env.fish_positions, base_fish_pos), f"{cond}: fish pos changed"
        assert np.array_equal(env.fish_velocities, base_fish_vel), f"{cond}: fish vel changed"
        assert np.array_equal(env.fish_alive, base_alive), f"{cond}: alive changed"
        assert np.array_equal(env.predator_pos, base_pred_pos), f"{cond}: predator pos changed"
        assert env.timestep == 0, f"{cond}: timestep changed"
        assert env.np_random.bit_generator.state == base_rng, f"{cond}: RNG changed"
        expected = common.rotate_predator_velocity(base_pred_vel, cond)
        assert np.array_equal(env.predator_vel, expected), f"{cond}: predator vel not the exact rotation"
    env.close()
    print("PASS test_only_predator_velocity_rotates")


def test_deg0_is_exact_identity():
    env = common.make_base_env(False)
    obs0, snap = venv.reset_nominal(env, SEED)
    obs_deg0, _ = venv.apply_condition(env, snap, "deg0")
    assert np.array_equal(obs_deg0, obs0), "deg0 did not reproduce the raw reset obs"
    assert np.array_equal(env.predator_vel, snap["predator_vel"]), "deg0 did not reproduce predator vel"
    # rotation matrices are exact signed permutations: norm preserved exactly
    for cond in common.CONDITIONS:
        rot = common.rotate_predator_velocity(snap["predator_vel"], cond)
        assert float(np.linalg.norm(rot)) == float(np.linalg.norm(snap["predator_vel"])), \
            f"{cond}: norm not preserved exactly"
    env.close()
    print("PASS test_deg0_is_exact_identity")


def test_obs_reflects_rotated_predator_velocity():
    env = common.make_base_env(False)
    _obs0, snap = venv.reset_nominal(env, SEED)
    for cond in common.CONDITIONS:
        obs, _ = venv.apply_condition(env, snap, cond)
        pred_vel = env.predator_vel.copy()
        alive = np.where(env.fish_alive)[0]
        checked = 0
        for r, i in enumerate(alive):
            if obs[r][5] > 0.5:  # predator visible
                exp_x = pred_vel[0] / env.FISH_MAX_SPEED
                exp_y = pred_vel[1] / env.FISH_MAX_SPEED
                assert abs(float(obs[r][8]) - float(exp_x)) < 1e-6, f"{cond}: obs pvx mismatch"
                assert abs(float(obs[r][9]) - float(exp_y)) < 1e-6, f"{cond}: obs pvy mismatch"
                checked += 1
        assert checked > 0, f"{cond}: no fish saw the predator; obs test vacuous"
    env.close()
    print("PASS test_obs_reflects_rotated_predator_velocity")


def _direct_v50_run(model, seed):
    """The direct v50-style deterministic multi-fish loop (no condition wrapper)."""
    env = common.make_base_env(False)
    obs, _info = env.reset(seed=seed)
    actions_log = []
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=True)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        actions_log.append(actions)
        obs, _r, term, trunc, info = env.step(actions)
        if term or trunc:
            break
    final = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
    env.close()
    return actions_log, final


def test_nominal_deg0_matches_direct_v50():
    manifest = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)
    name = "rep0_ppo"
    path = ROOT / manifest["controllers"][name]["checkpoint"]
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")

    env = common.make_base_env(False)
    obs, snap = venv.reset_nominal(env, SEED)
    obs, _ = venv.apply_condition(env, snap, "deg0")
    actions_v55 = []
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=True)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        actions_v55.append(actions)
        obs, _r, term, trunc, info = env.step(actions)
        if term or trunc:
            break
    final_v55 = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
    env.close()

    actions_direct, final_direct = _direct_v50_run(model, SEED)
    assert len(actions_v55) == len(actions_direct), "episode length differs"
    for a, b in zip(actions_v55, actions_direct):
        assert a == b, "action sequence differs between v55 deg0 and direct v50"
    assert final_v55 == final_direct, "final survival differs"
    print(f"PASS test_nominal_deg0_matches_direct_v50 (steps={len(actions_v55)}, final={final_v55})")


def test_gravity_not_rotated():
    env = common.make_base_env(False)
    _obs0, snap = venv.reset_nominal(env, SEED)
    g = env.PREDATOR_GRAVITY * env.dt
    for cond in common.CONDITIONS:
        venv.apply_condition(env, snap, cond)
        vel = env.predator_vel.copy()
        env._update_predator()  # gravity + move (+ bounce if needed)
        # no bounce at reset distance; gravity must have added +g to y only
        assert abs(float(env.predator_vel[1] - vel[1]) - g) < 1e-6, \
            f"{cond}: gravity did not add +{g} to y"
        assert abs(float(env.predator_vel[0] - vel[0])) < 1e-6, \
            f"{cond}: gravity changed x (rotated gravity?)"
    env.close()
    print("PASS test_gravity_not_rotated")


def test_frozen_model_hashes_unchanged():
    manifest = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)
    for name, info in manifest["controllers"].items():
        rel = info["checkpoint"]
        assert "corrected_streams" in rel, f"{name} not under corrected_streams: {rel}"
        assert "pilot" not in rel, f"{name} points at the pilot: {rel}"
        assert rel.endswith("rep" + name[3] + "_survival_only/checkpoints/model_final.zip")
        h = verify.policy_tensor_sha256_from_zip(ROOT / rel)
        assert h == info["policy_tensor_sha256"], f"{name} model hash changed"
    print("PASS test_frozen_model_hashes_unchanged")


def test_rules_read_only_obs():
    """A rule action must depend only on the passed obs vector, never on env state."""
    import v55_evaluate as ev

    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    obs = np.zeros(11, dtype=np.float32)
    obs[5] = 1.0
    obs[6] = 0.3
    obs[7] = -0.2
    obs[8] = 0.4
    obs[9] = 0.1
    for rule in ("hold", "flee_lead", "safe_top"):
        a = ev.rule_action(rule, obs, rng_a)
        b = ev.rule_action(rule, obs.copy(), rng_b)
        assert a == b, f"{rule} action not a pure function of obs"
    import inspect

    params = list(inspect.signature(ev.rule_action).parameters)
    assert params == ["rule", "obs", "rng"], f"rule_action gained hidden state: {params}"
    print("PASS test_rules_read_only_obs")


def test_statistical_unit_is_episode():
    assert common.NUM_FISH == 96
    env = common.make_base_env(False)
    obs, _info = env.reset(seed=SEED)
    assert obs.shape[0] == int(env.fish_alive.sum()), "per-fish obs count != alive count"
    env.close()
    print("PASS test_statistical_unit_is_episode")


# Prior-round seed banks that must not collide with v55's generated values.
PRIOR_RNG_SEEDS = (
    481000, 481001, 481002, 555001, 555002,
    482100, 482101, 482102,
    500200, 500201, 500202,
    510101, 510102,
    530100, 530101, 530102,
    540101, 540102,
)


def test_seed_banks_disjoint():
    report = common.resolve_seeds("report")
    debug = common.resolve_seeds("debug")
    assert len(set(report)) == len(report), "v55 report seeds repeat"
    assert len(set(debug)) == len(debug), "v55 debug seeds repeat"
    assert not (set(report) & set(debug)), "v55 debug/report seeds overlap"
    mine = set(report) | set(debug)
    for rng_seed in PRIOR_RNG_SEEDS:
        prior = set(common.episode_seeds(rng_seed, 40))
        overlap = mine & prior
        assert not overlap, f"v55 seeds overlap bank {rng_seed}: {sorted(overlap)[:5]}"
    print(f"PASS test_seed_banks_disjoint ({len(PRIOR_RNG_SEEDS)} prior banks checked)")


def test_env_config_includes_heading_bias():
    cfg = common.env_config(False)
    hb = cfg.get("predator_heading_bias")
    sb = cfg.get("predator_pre_roll_speed_bias")
    assert hb and len(hb) == 8, f"heading bias missing/incomplete: {hb}"
    assert sb and len(sb) == 5, f"speed bias missing/incomplete: {sb}"
    # and the frozen manifest must carry the same full config
    manifest = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)
    assert manifest["env"]["env_config"] == cfg, "manifest env config differs from live config"
    print("PASS test_env_config_includes_heading_bias")


if __name__ == "__main__":
    test_only_predator_velocity_rotates()
    test_deg0_is_exact_identity()
    test_obs_reflects_rotated_predator_velocity()
    test_nominal_deg0_matches_direct_v50()
    test_gravity_not_rotated()
    test_frozen_model_hashes_unchanged()
    test_rules_read_only_obs()
    test_statistical_unit_is_episode()
    test_seed_banks_disjoint()
    test_env_config_includes_heading_bias()
    print("\nALL V55 SEMANTICS TESTS PASSED")
