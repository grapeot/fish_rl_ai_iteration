#!/usr/bin/env python3
"""v57 meaning tests — the load-bearing semantics of the round-9 axis.

Must pass BEFORE any training result is interpreted. They check, on the real code:

  1. `aug_only_touches_velocity_and_obs`: for a fixed worker seed, an augmented
     reset scales ONLY `fish_velocities` by the drawn factor and recomputes the
     focal obs; positions, alive flags, predator pos/vel, pre-roll trace and the
     base world RNG state are bit-identical to the un-augmented reset.
  2. `aug_rng_independent_and_reproducible`: two wrappers with the same worker seed
     draw the same factor sequence across many resets; a different seed draws a
     varied (non-constant) sequence; the base world RNG is never consumed by the
     augmentation.
  3. `control_mode_equals_v50`: `V57SingleFishEnv(augment_velocity=False)` reproduces
     the accepted `V50SingleFishEnv` step-for-step (obs, reward, terminated,
     truncated) for the same worker seed and action sequence.
  4. `reward_and_horizon_unchanged`: the augmented wrapper keeps the survival_only
     reward (+0.7 alive / -50 death) and the 500-step timeout-bootstrap horizon.
  5. `real_subproc_autoreset_varies_factor`: a real `SubprocVecEnv` of augmented
     wrappers auto-resets into fresh episodes and covers more than one factor.
  6. `evaluation_conditions_are_v54`: the v57 condition factors are exactly the v54
     factors, and the v57 nominal condition reproduces the direct baseline loop.

Run:
  experiments/v48/.venv/bin/python experiments/v57/tests/test_v57_semantics.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V57 = HERE.parent
ROOT = V57.parents[1]
for p in (str(ROOT), str(V57), str(ROOT / "experiments" / "v50"),
          str(ROOT / "experiments" / "v54")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v57_common as common  # noqa: E402
from v57_env import V57SingleFishEnv  # noqa: E402
from v50_env import V50SingleFishEnv  # noqa: E402
import v54_env as v54  # noqa: E402
import v54_common as v54c  # noqa: E402

SEED = 570777


def test_aug_only_touches_velocity_and_obs():
    ctrl = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED, augment_velocity=False)
    aug = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED, augment_velocity=True)
    _o0, _i0 = ctrl.reset()
    obs_a, info_a = aug.reset()
    assert np.array_equal(ctrl.base_env.fish_positions, aug.base_env.fish_positions), "positions changed"
    assert np.array_equal(ctrl.base_env.fish_alive, aug.base_env.fish_alive), "alive changed"
    assert np.array_equal(ctrl.base_env.predator_pos, aug.base_env.predator_pos), "predator pos changed"
    assert np.array_equal(ctrl.base_env.predator_vel, aug.base_env.predator_vel), "predator vel changed"
    assert ctrl.base_env._last_pre_roll_stats == aug.base_env._last_pre_roll_stats, "pre-roll changed"
    # base world RNG untouched by the augmentation
    assert ctrl.base_env.np_random.bit_generator.state == aug.base_env.np_random.bit_generator.state, (
        "augmentation polluted the base world RNG")
    factor = float(info_a["aug_factor"])
    assert factor in common.AUG_FACTORS
    expected = (ctrl.base_env.fish_velocities * np.float32(factor)).astype(np.float32)
    assert np.array_equal(aug.base_env.fish_velocities, expected), "velocity not exact scale"
    # focal obs velocity channel reflects the scaled velocity
    alive = np.where(aug.base_env.fish_alive)[0]
    row = int(np.searchsorted(alive, aug.focal_id))
    assert abs(float(obs_a[2]) - float(aug.base_env.fish_velocities[aug.focal_id][0]) / 2.0) < 1e-6
    assert abs(float(obs_a[3]) - float(aug.base_env.fish_velocities[aug.focal_id][1]) / 2.0) < 1e-6
    ctrl.close()
    aug.close()
    print("PASS test_aug_only_touches_velocity_and_obs")


def test_aug_rng_independent_and_reproducible():
    a = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED, augment_velocity=True)
    b = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED, augment_velocity=True)
    c = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED + 5, augment_velocity=True)
    # advance each with a short non-augmented-mode base RNG consumption to show the
    # aug draws are unaffected: do a handful of explicit-seed resets.
    fa, fb, fc = [], [], []
    for k in range(24):
        _oa, ia = a.reset(seed=1000 + k)
        _ob, ib = b.reset(seed=1000 + k)
        _oc, ic = c.reset(seed=1000 + k)
        fa.append(ia["aug_factor"])
        fb.append(ib["aug_factor"])
        fc.append(ic["aug_factor"])
    assert fa == fb, "same worker seed produced different factor sequences"
    assert len(set(fc)) > 1, "different seed produced a constant factor sequence"
    assert len(set(fa)) > 1, "factor sequence never varied across episodes"
    a.close(); b.close(); c.close()
    print("PASS test_aug_rng_independent_and_reproducible")


def test_control_mode_equals_v50():
    v50env = V50SingleFishEnv(reward_mode="survival_only", worker_seed=SEED)
    v57env = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED, augment_velocity=False)
    o50, _ = v50env.reset()
    o57, _ = v57env.reset()
    assert np.array_equal(o50, o57), "control-mode reset obs differs from v50"
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = int(rng.integers(0, 5))
        out50 = v50env.step(a)
        out57 = v57env.step(a)
        assert np.array_equal(out50[0], out57[0]), "obs differs"
        assert float(out50[1]) == float(out57[1]), "reward differs"
        assert bool(out50[2]) == bool(out57[2]), "terminated differs"
        assert bool(out50[3]) == bool(out57[3]), "truncated differs"
        if out50[2] or out50[3]:
            break
    v50env.close(); v57env.close()
    print("PASS test_control_mode_equals_v50")


def test_reward_and_horizon_unchanged():
    env = V57SingleFishEnv(reward_mode="survival_only", worker_seed=SEED, augment_velocity=True)
    obs, info = env.reset()
    steps = 0
    last_reward = None
    while True:
        obs, reward, terminated, truncated, info = env.step(4)  # hold
        steps += 1
        last_reward = float(reward)
        assert reward in (common.SURVIVAL_STEP_REWARD, common.DEATH_STEP_PENALTY), (
            f"unexpected reward {reward}")
        if terminated or truncated:
            break
        assert steps < common.MAX_TIMESTEPS + 5, "episode exceeded horizon"
    if terminated and not truncated:
        assert last_reward == common.DEATH_STEP_PENALTY, "death step reward is not -50"
    else:
        assert steps == common.MAX_TIMESTEPS, "horizon end not at MAX_TIMESTEPS"
        assert last_reward == common.SURVIVAL_STEP_REWARD, "horizon-alive last reward is not +0.7"
        assert info.get("agent_truncated") is True, "horizon end is not a truncation (bootstrap)"
    env.close()
    print("PASS test_reward_and_horizon_unchanged")


def test_real_subproc_autoreset_varies_factor():
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make(ws):
        def _f():
            import torch

            torch.set_num_threads(1)
            return V57SingleFishEnv(reward_mode="survival_only", worker_seed=ws, augment_velocity=True)

        return _f

    vec = SubprocVecEnv([make(6000011 + i) for i in range(2)])
    vec._seeds = [None] * 2
    vec.reset()
    rng = np.random.default_rng(3)
    dones_total = 0
    for _ in range(4000):
        actions = np.array([[int(rng.integers(0, 5))] for _ in range(2)])
        _obs, _rew, dones, infos = vec.step(actions)
        dones_total += int(np.sum(dones))
        if dones_total >= 4:
            break
    resets = vec.get_attr("_aug_resets")
    counts = vec.get_attr("_aug_factor_counts")
    assert sum(int(r) for r in resets) >= 4, f"too few auto-resets: {resets}"
    merged = {}
    for d in counts:
        for k, v in d.items():
            merged[str(float(k))] = merged.get(str(float(k)), 0) + int(v)
    assert len([k for k, v in merged.items() if v > 0]) > 1, f"factor never varied: {merged}"
    vec.close()
    print("PASS test_real_subproc_autoreset_varies_factor")


def _direct_baseline_run(model, seed):
    env = common.make_base_env(False)
    obs, _info = env.reset(seed=seed)
    log = []
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=True)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        log.append(actions)
        obs, _r, term, trunc, info = env.step(actions)
        if term or trunc:
            break
    final = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
    env.close()
    return log, final


def test_evaluation_conditions_are_v54():
    assert dict(common.VELOCITY_FACTORS) == dict(v54c.VELOCITY_FACTORS), "v57 factors != v54 factors"
    assert tuple(common.CONDITIONS) == tuple(v54c.CONDITIONS), "v57 conditions != v54 conditions"
    from stable_baselines3 import PPO

    path = ROOT / common.CONTROL_RUN_DIR[0] / "checkpoints" / "model_final.zip"
    model = PPO.load(str(path), device="cpu")
    env = common.make_base_env(False)
    obs, snap = v54.reset_nominal(env, SEED)
    obs, _ = v54.apply_condition(env, snap, common.VELOCITY_FACTORS["nominal"])
    log = []
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=True)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        log.append(actions)
        obs, _r, term, trunc, info = env.step(actions)
        if term or trunc:
            break
    final = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
    env.close()
    log_direct, final_direct = _direct_baseline_run(model, SEED)
    assert log == log_direct and final == final_direct, "v57 nominal path != direct baseline"
    print("PASS test_evaluation_conditions_are_v54")


if __name__ == "__main__":
    test_aug_only_touches_velocity_and_obs()
    test_aug_rng_independent_and_reproducible()
    test_control_mode_equals_v50()
    test_reward_and_horizon_unchanged()
    test_real_subproc_autoreset_varies_factor()
    test_evaluation_conditions_are_v54()
    print("\nALL V57 SEMANTICS TESTS PASSED")
