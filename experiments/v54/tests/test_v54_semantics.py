#!/usr/bin/env python3
"""v54 meaning tests — the load-bearing semantics of the round-6 axis.

Must pass BEFORE any report. They check, on the real code:

  1. `conditions_share_world_scale_velocity`: for a fixed seed the three
     conditions have identical positions, alive flags, predator pos/vel,
     pre-roll trace and RNG-driven future; only fish_velocities are scaled by the
     exact factor {1.0, 0.5, 0.0}.
  2. `obs_reflects_velocity`: the velocity observation channel is the scaled
     velocity, and nominal factor 1.0 reproduces the raw post-reset obs exactly.
  3. `nominal_matches_direct_v50`: running a frozen policy through the v54
     nominal path and through the direct v50-style loop on the same seed yields
     the identical action sequence and final survival.
  4. `zero_speed_semantics`: at exactly zero velocity, turn (1/2) and decelerate
     (3) and hold (4) are no-ops; forward (0) moves the fish in +x; obs[2..3]==0.
     This is read from `fish_env._update_fish`, not assumed.
  5. `frozen_model_hashes_unchanged`: the three checkpoints still hash to the
     frozen manifest values and live under corrected_streams (not the pilot).
  6. `rules_read_only_obs`: rule anchors depend only on the 11 base dims; a hidden
     predator state change that leaves obs unchanged cannot change a rule action.
  7. `statistical_unit_is_episode`: the analysis counts scenarios, not fish.

Run:
  experiments/v48/.venv/bin/python experiments/v54/tests/test_v54_semantics.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V54 = HERE.parent
ROOT = V54.parents[1]
for p in (str(ROOT), str(V54), str(ROOT / "experiments" / "v50")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v54_common as common  # noqa: E402
import v54_env as venv  # noqa: E402
import v54_verify as verify  # noqa: E402

SEED = 540777


def test_conditions_share_world_scale_velocity():
    env = common.make_base_env(False)
    obs0, snap = venv.reset_nominal(env, SEED)
    base_pos = snap["fish_positions"].copy()
    base_alive = snap["fish_alive"].copy()
    base_pred_pos = snap["predator_pos"].copy()
    base_pred_vel = snap["predator_vel"].copy()
    base_vel = snap["fish_velocities"].copy()
    base_reroll = snap["last_pre_roll_stats"].copy()

    for cond, factor in common.VELOCITY_FACTORS.items():
        obs, _ = venv.apply_condition(env, snap, factor)
        assert np.array_equal(env.fish_positions, base_pos), f"{cond}: positions changed"
        assert np.array_equal(env.fish_alive, base_alive), f"{cond}: alive changed"
        assert np.array_equal(env.predator_pos, base_pred_pos), f"{cond}: predator pos changed"
        assert np.array_equal(env.predator_vel, base_pred_vel), f"{cond}: predator vel changed"
        expected = (base_vel * np.float32(factor)).astype(np.float32)
        assert np.array_equal(env.fish_velocities, expected), f"{cond}: velocity not exact scale"
        # pre-roll trace is restored identically (not re-drawn)
        assert env._last_pre_roll_stats == base_reroll, f"{cond}: pre-roll trace changed"
    # nominal factor 1.0 must reproduce the raw post-reset state bit-exactly
    obs_nom, _ = venv.apply_condition(env, snap, 1.0)
    assert np.array_equal(obs_nom, obs0), "nominal did not reproduce raw reset obs"
    # RNG state is restored: after consuming, the next random draw is the same
    # from the reset snapshot regardless of the condition factor applied.
    venv.apply_condition(env, snap, 1.0)
    r_a = env.np_random.random(4)
    venv.apply_condition(env, snap, 0.0)
    r_b = env.np_random.random(4)
    assert np.array_equal(r_a, r_b), "RNG future diverged across conditions"
    env.close()
    print("PASS test_conditions_share_world_scale_velocity")


def test_obs_reflects_velocity():
    env = common.make_base_env(False)
    _obs0, snap = venv.reset_nominal(env, SEED)
    for factor in (1.0, 0.5, 0.0):
        obs, _ = venv.apply_condition(env, snap, factor)
        alive = np.where(env.fish_alive)[0]
        for r, i in enumerate(alive):
            exp_x = env.fish_velocities[i][0] / env.FISH_MAX_SPEED
            exp_y = env.fish_velocities[i][1] / env.FISH_MAX_SPEED
            assert abs(float(obs[r][2]) - exp_x) < 1e-6, f"factor {factor}: obs vx mismatch"
            assert abs(float(obs[r][3]) - exp_y) < 1e-6, f"factor {factor}: obs vy mismatch"
    env.close()
    print("PASS test_obs_reflects_velocity")


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


def test_nominal_matches_direct_v50():
    """All three frozen models must reproduce the direct v50 loop bit-for-bit
    (same action sequence, same final survival) on the same seed via the nominal
    (factor 1.0) path."""
    manifest = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)
    from stable_baselines3 import PPO

    for name in common.PPO_CONTROLLER_NAMES:
        path = ROOT / manifest["controllers"][name]["checkpoint"]
        model = PPO.load(str(path), device="cpu")

        env = common.make_base_env(False)
        obs, snap = venv.reset_nominal(env, SEED)
        obs, _ = venv.apply_condition(env, snap, 1.0)
        actions_v54 = []
        while True:
            if len(obs) > 0:
                batch, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch)]
            else:
                actions = []
            actions_v54.append(actions)
            obs, _r, term, trunc, info = env.step(actions)
            if term or trunc:
                break
        final_v54 = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
        env.close()

        actions_direct, final_direct = _direct_v50_run(model, SEED)
        assert len(actions_v54) == len(actions_direct), f"{name}: episode length differs"
        for a, b in zip(actions_v54, actions_direct):
            assert a == b, f"{name}: action sequence differs between v54 nominal and direct v50"
        assert final_v54 == final_direct, f"{name}: final survival differs"
        print(f"  {name}: steps={len(actions_v54)}, final={final_v54}")
    print("PASS test_nominal_matches_direct_v50")


def test_zero_speed_semantics():
    env = common.make_base_env(False)
    _obs0, snap = venv.reset_nominal(env, SEED)
    venv.apply_condition(env, snap, 0.0)
    assert np.allclose(env.fish_velocities, 0.0), "zero condition left nonzero velocity"
    fid = int(np.where(env.fish_alive)[0][0])

    def pos_vel():
        return env.fish_positions[fid].copy(), env.fish_velocities[fid].copy()

    # actions 1 (left), 2 (right), 3 (decelerate), 4 (hold): all no-ops at exactly
    # zero velocity (vel = new_dir*speed = 0, or vel*=0.9 => 0, or hold).
    for act in (1, 2, 3, 4):
        venv.apply_condition(env, snap, 0.0)
        p0, v0 = pos_vel()
        env._update_fish(fid, act)
        p1, v1 = pos_vel()
        assert np.allclose(v1, 0.0), f"action {act} gave a zero-velocity fish momentum"
        assert np.allclose(p1, p0), f"action {act} moved a zero-velocity fish"

    # action 0 (forward) moves the fish in +x (heading fallback [1,0]).
    venv.apply_condition(env, snap, 0.0)
    p0, _ = pos_vel()
    env._update_fish(fid, 0)
    p1, v1 = pos_vel()
    dv = p1 - p0
    assert dv[0] > 0 and abs(float(dv[1])) < 1e-9, f"forward at zero did not move +x: {dv}"
    assert v1[0] > 0 and abs(float(v1[1])) < 1e-9, f"forward at zero did not establish +x velocity: {v1}"

    # obs velocity channel is zero under the zero condition
    obs, _ = venv.apply_condition(env, snap, 0.0)
    row = int(np.searchsorted(np.where(env.fish_alive)[0], fid))
    assert abs(float(obs[row][2])) < 1e-9 and abs(float(obs[row][3])) < 1e-9
    env.close()
    print("PASS test_zero_speed_semantics")


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
    import v54_evaluate as ev

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
    # rule_action takes obs only: signature check
    import inspect

    params = list(inspect.signature(ev.rule_action).parameters)
    assert params == ["rule", "obs", "rng"], f"rule_action gained hidden state: {params}"
    print("PASS test_rules_read_only_obs")


def test_statistical_unit_is_episode():
    """The analysis must count scenarios, not fish: verify the aggregation logic
    on a synthetic grid (per scenario average over the fixed 3 PPO, paired to
    HOLD) rather than just checking the 96 denominator."""
    assert common.NUM_FISH == 96
    env = common.make_base_env(False)
    obs, _info = env.reset(seed=SEED)
    assert obs.shape[0] == int(env.fish_alive.sum()), "per-fish obs count != alive count"
    env.close()

    import v54_analyze as an

    # synthetic grid: 3 scenarios x {3 PPO + HOLD} x {nominal, zero}
    grid = {}
    for s in (101, 102, 103):
        grid[("rep0_ppo", s)] = {"nominal": 0.9, "zero": 0.8}
        grid[("rep1_ppo", s)] = {"nominal": 0.8, "zero": 0.7}
        grid[("rep2_ppo", s)] = {"nominal": 0.7, "zero": 0.6}
        grid[("rule_hold", s)] = {"nominal": 0.5, "zero": 0.5}
    rel = an.relative_hold_contrasts(grid, n_boot=200)
    # PPO avg = (0.9+0.8+0.7)/3 = 0.8 nominal, (0.8+0.7+0.6)/3 = 0.7 zero
    # advantage nominal = 0.8-0.5 = 0.3 ; zero = 0.7-0.5 = 0.2 ; DiD = -0.1
    assert abs(rel["nominal_ppo_minus_hold"]["mean"] - 0.3) < 1e-9
    assert abs(rel["zero_ppo_minus_hold"]["mean"] - 0.2) < 1e-9
    assert abs(rel["difference_in_differences"]["mean"] - (-0.1)) < 1e-9
    assert rel["nominal_ppo_minus_hold"]["pos_scenarios"] == 3
    assert rel["difference_in_differences"]["neg_scenarios"] == 3
    print("PASS test_statistical_unit_is_episode")


if __name__ == "__main__":
    test_conditions_share_world_scale_velocity()
    test_obs_reflects_velocity()
    test_nominal_matches_direct_v50()
    test_zero_speed_semantics()
    test_frozen_model_hashes_unchanged()
    test_rules_read_only_obs()
    test_statistical_unit_is_episode()
    print("\nALL V54 SEMANTICS TESTS PASSED")
