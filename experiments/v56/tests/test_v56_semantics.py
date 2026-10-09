#!/usr/bin/env python3
"""v56 meaning tests — the load-bearing semantics of the round-8 axis.

Must pass BEFORE any report. They check, on the real code:

  1. `test_only_predator_velocity_scales`: for a fixed seed the three conditions have
     identical fish positions/velocities, alive flags, predator position, timestep,
     pre-roll trace and RNG state; only `predator_vel` differs, by the exact scalar
     factor, with the direction (unit vector) preserved and the norm scaled by the
     factor.
  2. `test_base_is_exact_identity`: the scale1p00 condition reproduces the raw nominal
     post-reset obs and predator velocity bit-for-bit.
  3. `test_obs_reflects_scaled_predator_velocity`: obs[8],obs[9] are the scaled predator
     velocity over FISH_MAX_SPEED when the predator is visible; when NOT visible, the
     predator-derived channels (5..10) are the zero/invisible encoding and never leak
     the scaled velocity.
  4. `test_zero_norm_no_divide`: a synthetically zeroed nominal predator velocity keeps
     the zero vector under every condition and sets the zero-norm flag (no divide by
     zero, no NaN).
  5. `test_nominal_scale1p00_matches_direct_v50`: running a frozen policy through the
     v56 base path and through the direct v50-style loop on the same seed yields the
     identical action sequence and final survival (full-rollout reproduction).
  6. `test_gravity_not_changed`: `_update_predator` adds gravity to +y for every
     condition; scaling the predator velocity is not a gravity turn.
  7. `test_frozen_model_hashes_unchanged`: the three checkpoints still hash to the
     frozen manifest values and live under corrected_streams (not the pilot).
  8. `test_rules_read_only_obs`: rule anchors depend only on the 11 base dims.
  9. `test_statistical_unit_is_episode`: the analysis counts scenarios, not fish, and
     the raw-derived slowdown/speedup contrasts match a synthetic analytic value.
 10. `test_seed_banks_disjoint`: the 560101/560102 generated episode values do not
     overlap each other or any prior round's bank (v48/v49/v50/v51/v53/v54/v55).
 11. `test_env_config_includes_heading_bias`: the frozen env config carries the full
     predator heading and speed bias (cannot be silently omitted).

Run:
  experiments/v48/.venv/bin/python experiments/v56/tests/test_v56_semantics.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V56 = HERE.parent
ROOT = V56.parents[1]
for p in (str(ROOT), str(V56), str(ROOT / "experiments" / "v50")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v56_common as common  # noqa: E402
import v56_env as venv  # noqa: E402
import v56_verify as verify  # noqa: E402

SEED = 560777


def test_only_predator_velocity_scales():
    env = common.make_base_env(False)
    obs0, snap = venv.reset_nominal(env, SEED)
    base_fish_pos = snap["fish_positions"].copy()
    base_fish_vel = snap["fish_velocities"].copy()
    base_alive = snap["fish_alive"].copy()
    base_pred_pos = snap["predator_pos"].copy()
    base_pred_vel = snap["predator_vel"].copy()
    base_rng = snap["np_random_state"]
    base_reroll = snap["last_pre_roll_stats"].copy()

    nominal_norm = float(np.linalg.norm(base_pred_vel))
    assert nominal_norm > 1e-6, "fixture seed produced a zero predator norm"

    for cond, factor in common.SPEED_FACTORS.items():
        obs, info = venv.apply_condition(env, snap, cond)
        assert np.array_equal(env.fish_positions, base_fish_pos), f"{cond}: fish pos changed"
        assert np.array_equal(env.fish_velocities, base_fish_vel), f"{cond}: fish vel changed"
        assert np.array_equal(env.fish_alive, base_alive), f"{cond}: alive changed"
        assert np.array_equal(env.predator_pos, base_pred_pos), f"{cond}: predator pos changed"
        assert env.timestep == 0, f"{cond}: timestep changed"
        assert env.np_random.bit_generator.state == base_rng, f"{cond}: RNG changed"
        assert env._last_pre_roll_stats == base_reroll, f"{cond}: pre-roll trace changed"
        expected = (base_pred_vel * np.float32(factor)).astype(np.float32)
        assert np.array_equal(env.predator_vel, expected), f"{cond}: predator vel not the exact scale"
        # norm scales by the factor (up to float32 rounding); direction preserved
        scaled_norm = float(np.linalg.norm(env.predator_vel))
        assert abs(scaled_norm - nominal_norm * factor) < 1e-5, f"{cond}: norm not scaled by factor"
        if factor != 1.0:
            dir0 = base_pred_vel / nominal_norm
            dir1 = env.predator_vel / scaled_norm
            assert float(np.dot(dir0, dir1)) > 1 - 1e-6, f"{cond}: direction not preserved"
        assert not info["zero_predator_norm"], f"{cond}: unexpected zero-norm flag"
        assert abs(info["applied_predator_speed"] - scaled_norm) < 1e-9
    env.close()
    print("PASS test_only_predator_velocity_scales")


def test_base_is_exact_identity():
    env = common.make_base_env(False)
    obs0, snap = venv.reset_nominal(env, SEED)
    obs_base, _info = venv.apply_condition(env, snap, common.BASE_CONDITION)
    assert np.array_equal(obs_base, obs0), "base did not reproduce the raw reset obs"
    assert np.array_equal(env.predator_vel, snap["predator_vel"]), "base did not reproduce predator vel"
    env.close()
    print("PASS test_base_is_exact_identity")


def test_obs_reflects_scaled_predator_velocity():
    env = common.make_base_env(False)
    _obs0, snap = venv.reset_nominal(env, SEED)
    for cond in common.CONDITIONS:
        obs, _info = venv.apply_condition(env, snap, cond)
        pred_vel = env.predator_vel.copy()
        alive = np.where(env.fish_alive)[0]
        visible = 0
        invisible = 0
        for r, _i in enumerate(alive):
            if obs[r][5] > 0.5:  # predator visible
                exp_x = pred_vel[0] / env.FISH_MAX_SPEED
                exp_y = pred_vel[1] / env.FISH_MAX_SPEED
                assert abs(float(obs[r][8]) - float(exp_x)) < 1e-6, f"{cond}: obs pvx mismatch"
                assert abs(float(obs[r][9]) - float(exp_y)) < 1e-6, f"{cond}: obs pvy mismatch"
                visible += 1
            else:  # predator not visible: channels 5..10 must be the invisible encoding
                assert float(obs[r][5]) == 0.0, f"{cond}: invisible flag not 0"
                assert np.all(obs[r][6:11] == 0.0), f"{cond}: invisible fish leaked predator state"
                invisible += 1
        assert visible > 0, f"{cond}: no fish saw the predator; obs test vacuous"
    env.close()
    print(f"PASS test_obs_reflects_scaled_predator_velocity")


def test_zero_norm_no_divide():
    env = common.make_base_env(False)
    _obs0, snap = venv.reset_nominal(env, SEED)
    # force a zero nominal predator velocity in a copied snapshot
    snap = dict(snap)
    snap["predator_vel"] = np.zeros(2, dtype=np.float32)
    for cond in common.CONDITIONS:
        obs, info = venv.apply_condition(env, snap, cond)
        assert np.array_equal(env.predator_vel, np.zeros(2, dtype=np.float32)), \
            f"{cond}: zero norm was not preserved"
        assert info["zero_predator_norm"] is True, f"{cond}: zero-norm flag not set"
        assert np.all(np.isfinite(obs)), f"{cond}: NaN/inf obs under zero norm"
    env.close()
    print("PASS test_zero_norm_no_divide")


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


def test_nominal_scale1p00_matches_direct_v50():
    manifest = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)
    from stable_baselines3 import PPO

    for name in common.PPO_CONTROLLER_NAMES:
        path = ROOT / manifest["controllers"][name]["checkpoint"]
        model = PPO.load(str(path), device="cpu")

        env = common.make_base_env(False)
        obs, snap = venv.reset_nominal(env, SEED)
        obs, _info = venv.apply_condition(env, snap, common.BASE_CONDITION)
        actions_v56 = []
        while True:
            if len(obs) > 0:
                batch, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch)]
            else:
                actions = []
            actions_v56.append(actions)
            obs, _r, term, trunc, info = env.step(actions)
            if term or trunc:
                break
        final_v56 = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
        env.close()

        actions_direct, final_direct = _direct_v50_run(model, SEED)
        assert len(actions_v56) == len(actions_direct), f"{name}: episode length differs"
        for a, b in zip(actions_v56, actions_direct):
            assert a == b, f"{name}: action sequence differs between v56 base and direct v50"
        assert final_v56 == final_direct, f"{name}: final survival differs"
        print(f"  {name}: steps={len(actions_v56)}, final={final_v56}")
    print("PASS test_nominal_scale1p00_matches_direct_v50")


def test_gravity_not_changed():
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
    print("PASS test_gravity_not_changed")


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
    import v56_evaluate as ev

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

    import v56_analyze as an

    # synthetic grid: 3 scenarios x {3 PPO + HOLD} x {slow, base, fast}.
    # PPO average per scenario: base 0.8, slow 0.6, fast 0.9
    # slow effect = -0.2 ; fast effect = +0.1 ; asymmetry fast-slow = +0.3
    grid = {}
    for s in (101, 102, 103):
        grid[("rep0_ppo", s)] = {"scale0p75": 0.7, "scale1p00": 0.9, "scale1p25": 1.0}
        grid[("rep1_ppo", s)] = {"scale0p75": 0.6, "scale1p00": 0.8, "scale1p25": 0.9}
        grid[("rep2_ppo", s)] = {"scale0p75": 0.5, "scale1p00": 0.7, "scale1p25": 0.8}
        grid[("rule_hold", s)] = {"scale0p75": 0.5, "scale1p00": 0.5, "scale1p25": 0.5}
    view = an.speed_robustness_contrasts(grid, n_boot=200)
    assert abs(view["slowdown_scale0p75_minus_base"]["mean"] - (-0.2)) < 1e-9
    assert abs(view["speedup_scale1p25_minus_base"]["mean"] - 0.1) < 1e-9
    assert abs(view["asymmetry_fast_minus_slow"]["mean"] - 0.3) < 1e-9
    assert view["slowdown_scale0p75_minus_base"]["neg_scenarios"] == 3
    assert view["speedup_scale1p25_minus_base"]["pos_scenarios"] == 3
    assert view["asymmetry_fast_minus_slow"]["pos_scenarios"] == 3

    # relative HOLD baseline (same-condition paired advantage):
    # PPO avg - HOLD: slow 0.6-0.5=0.1, base 0.8-0.5=0.3, fast 0.9-0.5=0.4
    # fast advantage - base advantage = 0.4 - 0.3 = 0.1
    rel = an.relative_hold_contrasts(grid, n_boot=200)
    assert abs(rel["ppo_minus_hold"]["scale0p75"]["mean"] - 0.1) < 1e-9
    assert abs(rel["ppo_minus_hold"]["scale1p00"]["mean"] - 0.3) < 1e-9
    assert abs(rel["ppo_minus_hold"]["scale1p25"]["mean"] - 0.4) < 1e-9
    assert abs(rel["advantage_fast_minus_nominal"]["mean"] - 0.1) < 1e-9
    assert rel["ppo_minus_hold"]["scale1p00"]["pos_scenarios"] == 3
    assert rel["advantage_fast_minus_nominal"]["pos_scenarios"] == 3
    print("PASS test_statistical_unit_is_episode")


# Prior-round seed banks that must not collide with v56's generated values.
PRIOR_RNG_SEEDS = (
    481000, 481001, 481002, 555001, 555002,
    482100, 482101, 482102,
    500200, 500201, 500202,
    510101, 510102,
    530100, 530101, 530102,
    540101, 540102,
    550101, 550102,
)


def test_seed_banks_disjoint():
    report = common.resolve_seeds("report")
    debug = common.resolve_seeds("debug")
    assert len(set(report)) == len(report), "v56 report seeds repeat"
    assert len(set(debug)) == len(debug), "v56 debug seeds repeat"
    assert not (set(report) & set(debug)), "v56 debug/report seeds overlap"
    mine = set(report) | set(debug)
    for rng_seed in PRIOR_RNG_SEEDS:
        prior = set(common.episode_seeds(rng_seed, 40))
        overlap = mine & prior
        assert not overlap, f"v56 seeds overlap bank {rng_seed}: {sorted(overlap)[:5]}"
    print(f"PASS test_seed_banks_disjoint ({len(PRIOR_RNG_SEEDS)} prior banks checked)")


def test_env_config_includes_heading_bias():
    cfg = common.env_config(False)
    hb = cfg.get("predator_heading_bias")
    sb = cfg.get("predator_pre_roll_speed_bias")
    assert hb and len(hb) == 8, f"heading bias missing/incomplete: {hb}"
    assert sb and len(sb) == 5, f"speed bias missing/incomplete: {sb}"
    manifest = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)
    assert manifest["env"]["env_config"] == cfg, "manifest env config differs from live config"
    print("PASS test_env_config_includes_heading_bias")


if __name__ == "__main__":
    test_only_predator_velocity_scales()
    test_base_is_exact_identity()
    test_obs_reflects_scaled_predator_velocity()
    test_zero_norm_no_divide()
    test_nominal_scale1p00_matches_direct_v50()
    test_gravity_not_changed()
    test_frozen_model_hashes_unchanged()
    test_rules_read_only_obs()
    test_statistical_unit_is_episode()
    test_seed_banks_disjoint()
    test_env_config_includes_heading_bias()
    print("\nALL V56 SEMANTICS TESTS PASSED")
