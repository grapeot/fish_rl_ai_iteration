#!/usr/bin/env python3
"""Regression test: training vs evaluation action/observation semantics.

These tests pin down the behavior of `SingleFishEnv` so that a later training
fix (v49) can rely on them and detect regressions:
  1. the observation returned each step belongs to a different alive fish
     (the round-robin pointer advances);
  2. a broadcast single action and per-fish actions produce different world
     trajectories;
  3. the alive-position -> world-index mapping is positional, so it shifts
     after a death;
  4. v45 and v47 `SingleFishEnv` share the same semantics.

Run:
  experiments/v48/.venv/bin/python experiments/v48/tests/test_training_semantics.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V48 = HERE.parent
ROOT = V48.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V48))


def load_train(version):
    path = ROOT / "experiments" / version / "train.py"
    spec = importlib.util.spec_from_file_location(f"train_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_single(env_cls):
    return env_cls(
        num_fish=8,
        seed=7,
        sampling_mode="round_robin",
        include_neighbor_features=True,
        neighbor_radius=3.0,
        neighbor_average_count=6,
        initial_escape_boost=True,
        escape_boost_speed=0.8,
        escape_jitter_std=0.35,
        predator_spawn_jitter_radius=1.6,
        predator_pre_roll_steps=16,
        predator_pre_roll_angle_jitter=0.3,
        predator_pre_roll_speed_jitter=0.2,
    )


def test_sampled_fish_identity_changes_across_steps():
    """Claim: the observation returned each step is a different alive fish
    (round-robin pointer advances), so 'the' observation is not a fixed fish."""
    m = load_train("v45")
    env = make_single(m.SingleFishEnv)
    _, info = env.reset(seed=1001)
    seen = [info["sampled_fish_index"]]
    for _ in range(7):
        _, _, term, trunc, info = env.step(4)
        seen.append(info["sampled_fish_index"])
        if term or trunc:
            break
    assert len(set(seen)) > 1, f"identity did not change: {seen}"
    assert seen[:5] == [0, 1, 2, 3, 4], f"round_robin order broken: {seen}"
    env.close()
    print(f"PASS test_sampled_fish_identity_changes_across_steps: {seen}")


def test_broadcast_vs_per_fish_diverge():
    """Claim: broadcasting one action to all fish produces a different world
    trajectory than per-fish actions, so train/eval action semantics differ."""
    from fish_env import FishEscapeEnv

    def base():
        return FishEscapeEnv(
            num_fish=8,
            include_neighbor_features=True,
            neighbor_radius=3.0,
            neighbor_average_count=6,
            initial_escape_boost=True,
            escape_boost_speed=0.8,
            escape_jitter_std=0.35,
            predator_spawn_jitter_radius=1.6,
            predator_pre_roll_steps=16,
            predator_pre_roll_angle_jitter=0.3,
            predator_pre_roll_speed_jitter=0.2,
        )

    rng = np.random.default_rng(0)
    per_fish_actions = [int(rng.integers(0, 5)) for _ in range(8)]
    broadcast_action = 0

    e1 = base(); e1.reset(seed=2024)
    for _ in range(30):
        e1.step(per_fish_actions)

    e2 = base(); e2.reset(seed=2024)
    for _ in range(30):
        e2.step([broadcast_action] * 8)

    assert not np.allclose(e1.fish_positions, e2.fish_positions), \
        "broadcast and per-fish trajectories were identical"
    e1.close(); e2.close()
    print("PASS test_broadcast_vs_per_fish_diverge: trajectories differ")


def test_alive_index_positional_shift():
    """Claim: _select_index indexes the alive list positionally, so once a fish
    dies the alive-position -> world-index mapping shifts (identity is
    positional, not stable)."""
    m = load_train("v45")
    env = make_single(m.SingleFishEnv)
    env.reset(seed=333)
    # Force a death in the middle of the world list, then check the mapping.
    env.base_env.fish_alive[0] = False
    alive_world = np.where(env.base_env.fish_alive)[0]
    obs = env.base_env._get_observations()
    # observation row 0 now corresponds to world fish 1, not 0.
    assert alive_world[0] == 1
    assert obs.shape[0] == len(alive_world)
    # position 0's obs[0] must equal world fish 1's normalized x.
    expected = env.base_env.fish_positions[1][0] / env.base_env.STAGE_RADIUS
    assert abs(float(obs[0][0]) - float(expected)) < 1e-6, \
        "alive-position 0 did not map to world fish 1 after a death"
    env.close()
    print("PASS test_alive_index_positional_shift: mapping is positional")


def test_v45_and_v47_single_fish_env_agree():
    """Claim: the broadcast semantics are identical across v45 and v47 (both
    descend from the same wrapper)."""
    m45 = load_train("v45")
    m47 = load_train("v47")
    e45 = make_single(m45.SingleFishEnv)
    e47 = make_single(m47.SingleFishEnv)
    _, i45 = e45.reset(seed=55)
    _, i47 = e47.reset(seed=55)
    for _ in range(12):
        o45, r45, t45, tr45, i45 = e45.step(3)
        o47, r47, t47, tr47, i47 = e47.step(3)
        if t45 or tr45:
            break
    assert np.allclose(o45, o47), "v45/v47 observations diverged"
    assert abs(r45 - r47) < 1e-6, "v45/v47 rewards diverged"
    assert i45["sampled_fish_index"] == i47["sampled_fish_index"]
    e45.close(); e47.close()
    print("PASS test_v45_and_v47_single_fish_env_agree")


if __name__ == "__main__":
    test_sampled_fish_identity_changes_across_steps()
    test_broadcast_vs_per_fish_diverge()
    test_alive_index_positional_shift()
    test_v45_and_v47_single_fish_env_agree()
    print("\nALL TESTS PASSED")
