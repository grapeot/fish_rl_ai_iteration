#!/usr/bin/env python3
"""v49 meaning tests: fixed-identity transition mapping for SingleFishControlEnv.

These tests must pass before any long training run. They pin the semantics that
the legacy SingleFishEnv got wrong (broadcast action, rotating obs, mean reward,
cross-fish bootstrap). Note `test_truncation_at_horizon_bootstraps` checks only
the wrapper's truncated/terminated flags, not the SB3 value-target bootstrap;
`test_no_broadcast_only_focal_moves` shows only-focal divergence relative to an
all-HOLD twin.

Run:
  experiments/v48/.venv/bin/python experiments/v49/tests/test_v49_semantics.py
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V49 = HERE.parent
ROOT = V49.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V49))

from fish_env import FishEscapeEnv  # noqa: E402
from single_fish_env import SingleFishControlEnv, HOLD_ACTION  # noqa: E402
import common  # noqa: E402


def make(include_neighbor=False):
    return SingleFishControlEnv(include_neighbor_features=include_neighbor)


def test_fixed_focal_id_whole_episode():
    env = make()
    obs, info = env.reset(seed=1001)
    fid = info["focal_id"]
    assert info["is_agent"] is True
    for _ in range(50):
        obs, r, term, trunc, info = env.step(HOLD_ACTION)
        # obs must always equal the focal fish's own normalized position.
        expected = env.base_env.fish_positions[fid] / env.base_env.STAGE_RADIUS
        assert np.allclose(obs[0:2], expected, atol=1e-5), "obs is not the focal fish"
        assert info["focal_id"] == fid
        if term or trunc:
            break
    env.close()
    print(f"PASS test_fixed_focal_id_whole_episode (focal_id={fid})")


def test_other_fish_death_keeps_focal():
    env = make()
    obs, info = env.reset(seed=2002)
    fid = info["focal_id"]
    # Kill a non-focal fish directly and continue stepping many times.
    other = 0 if fid != 0 else 1
    env.base_env.fish_alive[other] = False
    for _ in range(30):
        obs, r, term, trunc, info = env.step(HOLD_ACTION)
        assert info["focal_id"] == fid, "focal id changed after another fish died"
        assert not term, "episode terminated because another fish died"
        if trunc:
            break
    env.close()
    print("PASS test_other_fish_death_keeps_focal")


def test_focal_death_terminates_once_with_penalty():
    env = make()
    env.reset(seed=3003)
    fid = env.focal_id
    # Teleport the predator onto the focal fish, then step once.
    env.base_env.predator_pos = env.base_env.fish_positions[fid].copy()
    env.base_env.predator_vel = np.zeros(2, dtype=np.float32)
    obs, r, term, trunc, info = env.step(HOLD_ACTION)
    assert term is True and trunc is False, "focal death must set terminated, not truncated"
    assert info["agent_death"] is True
    assert not env.base_env.fish_alive[fid]
    # death penalty (-50) is the only reward contribution for a step that dies.
    assert r <= -49.0, f"expected death penalty, got reward {r}"
    env.close()
    print(f"PASS test_focal_death_terminates_once_with_penalty (reward={r:.2f})")


def test_truncation_at_horizon_bootstraps():
    env = make()
    env.reset(seed=4004)
    fid = env.focal_id
    # Place the focal fish and predator far apart and jump to the last step so
    # the focal fish is guaranteed alive at the horizon.
    env.base_env.fish_positions[fid] = np.array([5.0, 0.0], dtype=np.float32)
    env.base_env.fish_velocities[fid] = np.array([0.0, 0.0], dtype=np.float32)
    env.base_env.predator_pos = np.array([-9.0, 0.0], dtype=np.float32)
    env.base_env.predator_vel = np.zeros(2, dtype=np.float32)
    env.base_env.timestep = env.base_env.MAX_TIMESTEPS - 1
    obs, r, term, trunc, info = env.step(HOLD_ACTION)
    assert trunc is True and term is False, "horizon must be truncation while alive"
    assert info["agent_truncated"] is True
    assert env.base_env.timestep == env.base_env.MAX_TIMESTEPS
    env.close()
    print("PASS test_truncation_at_horizon_bootstraps (steps=500)")


def test_no_broadcast_only_focal_moves():
    a = make()
    b = make()
    a.reset(seed=5005)
    b.reset(seed=5005)
    # Force identical focal ids for a clean comparison.
    a.focal_id = 7
    b.focal_id = 7
    # Step a: focal accelerates; b: everyone (including focal) holds.
    a.step(0)
    b.step(HOLD_ACTION)
    fa = a.base_env.fish_alive
    fb = b.base_env.fish_alive
    assert np.array_equal(fa, fb), "different deaths between focal-accel and hold world"
    diff = np.linalg.norm(a.base_env.fish_positions - b.base_env.fish_positions, axis=1)
    moved = diff > 1e-7
    assert moved[7], "focal fish did not move under its own action"
    assert moved.sum() == 1 and np.where(moved)[0][0] == 7, \
        f"more than the focal fish moved: {np.where(moved)[0].tolist()}"
    a.close(); b.close()
    print("PASS test_no_broadcast_only_focal_moves")


def test_spaces_match_eval_distribution():
    env = make()
    assert env.observation_space.shape == (11,), env.observation_space.shape
    assert int(env.action_space.n) == 5
    base = FishEscapeEnv(**common.env_config(include_neighbor_features=False))
    assert env.observation_space.shape == base.observation_space.shape
    assert int(env.action_space.n) == int(base.action_space.n)
    base.close(); env.close()
    print("PASS test_spaces_match_eval_distribution")


def test_wrapper_policy_usable_under_eval_loop():
    """A PPO trained through the wrapper must consume a base-env observation row
    of the same shape (train input == eval input), and the evaluator's per-fish
    batch predict must return a valid action for each row."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([make])
    model = PPO("MlpPolicy", vec, n_steps=32, batch_size=32, device="cpu",
                seed=0, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    model.learn(total_timesteps=64, progress_bar=False)
    base = FishEscapeEnv(**common.env_config(include_neighbor_features=False))
    obs, _ = base.reset(seed=8008)
    batch, _ = model.predict(np.asarray(obs), deterministic=True)
    actions = [int(a) for a in np.atleast_1d(batch)]
    assert len(actions) == obs.shape[0] == common.NUM_FISH
    assert all(0 <= a < 5 for a in actions)
    # The evaluator feeds exactly this call and this env config.
    assert obs.shape[1] == 11
    base.close(); vec.close()
    print("PASS test_wrapper_policy_usable_under_eval_loop")



def test_focal_reward_matches_world_reward_function():
    """The wrapper's returned reward equals the focal fish's own entry of the
    env reward vector (scaled), not a mean over fish."""
    env = make()
    env.reset(seed=6006)
    fid = env.focal_id
    # Reproduce the step manually on a twin env to compare focal reward.
    twin = make()
    twin.reset(seed=6006)
    twin.focal_id = fid
    actions = np.full(common.NUM_FISH, HOLD_ACTION, dtype=np.int64)
    obs, rewards, *_ = twin.base_env.step(actions)
    expected = float(rewards[fid])
    got = env.step(HOLD_ACTION)[1]
    assert abs(got - expected) < 1e-6, f"reward mismatch: {got} vs {expected}"
    env.close(); twin.close()
    print(f"PASS test_focal_reward_matches_world_reward_function (r={got:.4f})")


def test_obs_not_mean_of_alive():
    """Sanity: the returned obs is the focal fish's row, not an average over fish."""
    env = make()
    env.reset(seed=7007)
    fid = env.focal_id
    env.base_env.fish_positions[fid] = np.array([5.0, 0.0], dtype=np.float32)
    env.base_env.predator_pos = np.array([-9.0, 0.0], dtype=np.float32)
    env.base_env.predator_vel = np.zeros(2, dtype=np.float32)
    obs, _, term, _, _ = env.step(HOLD_ACTION)
    assert not term, "focal fish unexpectedly died"
    alive = np.where(env.base_env.fish_alive)[0]
    rows = env.base_env._get_observations()
    assert rows.shape[0] == len(alive)
    focal_row = rows[int(np.searchsorted(alive, fid))]
    assert np.array_equal(obs, focal_row), "obs is not exactly the focal fish row"
    assert not np.allclose(obs, rows.mean(axis=0)), "obs equals the mean over alive fish"
    env.close()
    print("PASS test_obs_not_mean_of_alive")


if __name__ == "__main__":
    test_fixed_focal_id_whole_episode()
    test_other_fish_death_keeps_focal()
    test_focal_death_terminates_once_with_penalty()
    test_truncation_at_horizon_bootstraps()
    test_no_broadcast_only_focal_moves()
    test_spaces_match_eval_distribution()
    test_wrapper_policy_usable_under_eval_loop()
    test_focal_reward_matches_world_reward_function()
    test_obs_not_mean_of_alive()
    print("\nALL V49 SEMANTICS TESTS PASSED")
