#!/usr/bin/env python3
"""Environment checks: decomposability of the survival objective.

Confirms two facts the diagnostic relies on:
  A. The predator trajectory is independent of fish actions and fish state.
     => for a fixed predator trajectory each fish's survival depends only on
        its own state, so a per-fish control policy matches this eval semantics
        and the broadcast training action is an artificial coupling.
  B. Fish do not collide with each other. Only fish-predator distance < 0.7
     kills. => density reward terms do not change death outcomes.

Run:
  experiments/v48/.venv/bin/python experiments/v48/tests/test_env_decomposability.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from fish_env import FishEscapeEnv  # noqa: E402


def make():
    return FishEscapeEnv(
        num_fish=16,
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


def test_predator_independent_of_fish_actions():
    e1 = make(); e1.reset(seed=9001)
    e2 = make(); e2.reset(seed=9001)
    for _ in range(250):
        # e1: everyone holds; e2: everyone accelerates then turns.
        e1.step([4] * 16)
        e2.step([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0])
    assert np.allclose(e1.predator_pos, e2.predator_pos), "predator depended on fish actions"
    assert np.allclose(e1.predator_vel, e2.predator_vel), "predator velocity depended on fish actions"
    e1.close(); e2.close()
    print("PASS test_predator_independent_of_fish_actions")


def test_predator_independent_of_fish_state():
    # Move all fish to wildly different positions; predator path must be identical
    # for the same seed because _update_predator never reads fish state.
    e1 = make(); e1.reset(seed=9002)
    e2 = make(); e2.reset(seed=9002)
    e2.fish_positions = np.array([[5.0, 0.0]] * 16, dtype=np.float32)  # pile on predator side
    for _ in range(120):
        e1.step([4] * 16)
        e2.step([4] * 16)
    assert np.allclose(e1.predator_pos, e2.predator_pos), "predator depended on fish positions"
    e1.close(); e2.close()
    print("PASS test_predator_independent_of_fish_state")


def test_no_fish_fish_collision():
    env = make(); env.reset(seed=9003)
    # Stack all fish onto one point, away from the predator; nobody should die.
    env.fish_positions = np.array([[9.0, 0.0]] * 16, dtype=np.float32)
    env.predator_pos = np.array([-9.0, 0.0], dtype=np.float32)
    env.predator_vel = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(20):
        env.step([4] * 16)
    assert int(env.fish_alive.sum()) == 16, "fish died from fish-fish overlap"
    e1 = env
    # Now two fish overlapping within 0.4 (FISH_SIZE*2=0.4) still both alive.
    e1.close()
    e2 = make(); e2.reset(seed=9004)
    e2.fish_positions[0] = [9.0, 0.0]
    e2.fish_positions[1] = [9.0, 0.05]
    e2.predator_pos = np.array([-9.0, 0.0], dtype=np.float32)
    e2.predator_vel = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(20):
        e2.step([4] * 16)
    assert e2.fish_alive[0] and e2.fish_alive[1], "overlapping fish died"
    e2.close()
    print("PASS test_no_fish_fish_collision")


def test_kill_only_near_predator():
    env = make(); env.reset(seed=9005)
    env.fish_positions = np.array([[5.0, 5.0]] * 15 + [[0.0, 0.0]], dtype=np.float32)
    env.predator_pos = np.array([0.0, 0.0], dtype=np.float32)
    env.predator_vel = np.array([0.0, 0.0], dtype=np.float32)
    env.step([4] * 16)
    # fish index 15 sits exactly on the predator -> dead; the rest alive.
    assert not env.fish_alive[15], "fish on top of predator survived"
    assert int(env.fish_alive.sum()) == 15, "unexpected extra deaths"
    env.close()
    print("PASS test_kill_only_near_predator")


if __name__ == "__main__":
    test_predator_independent_of_fish_actions()
    test_predator_independent_of_fish_state()
    test_no_fish_fish_collision()
    test_kill_only_near_predator()
    print("\nALL TESTS PASSED")
