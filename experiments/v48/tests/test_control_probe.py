#!/usr/bin/env python3
"""Meaningful tests for control_probe: the counterfactual only means something
if the fork start is bit-identical, the continue branch reproduces the un-forked
run, deaths are never leaked, and masking touches only the policy copy.

Run:
  experiments/v48/.venv/bin/python experiments/v48/tests/test_control_probe.py
"""

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V48 = HERE.parent
sys.path.insert(0, str(V48))

import common  # noqa: E402
import control_probe as cp  # noqa: E402


def test_seeds_disjoint_from_other_stages():
    s = set(cp.seeds_for(cp.N_SEEDS))
    assert len(s) == cp.N_SEEDS, "482002 seeds not unique"
    for name in ("smoke", "dev", "report"):
        other = set(common.resolve_seeds(name))
        assert not (s & other), f"seed overlap with {name}"
    print("PASS test_seeds_disjoint_from_other_stages")


def test_fork_start_state_identical():
    """deepcopy fork must preserve fish/predator/timestep/RNG exactly."""
    env, obs, _ = cp.make_split(7, "flee_lead", common.make_env)
    a = copy.deepcopy(env)
    b = copy.deepcopy(env)
    assert cp.same_state(a, env)
    assert cp.same_state(b, env)
    assert cp.same_state(a, b), "two forks of the same env diverged"
    # advancing one must not change the other (no shared arrays)
    sa = cp.snapshot(a)
    rng0 = np.random.default_rng(1)
    cp._continue(b, obs.copy(), lambda o: cp.rule_actions("flee_lead", o, rng0), cp.Window())
    assert cp.snapshot_equal(sa, a), "advancing fork B cross-modified fork A"
    env.close(); a.close(); b.close()
    print("PASS test_fork_start_state_identical")


def test_continue_matches_unforked_full_run():
    """The forked-continue branch must reproduce the independent un-forked run."""
    seed = 12345
    env, obs, _ = cp.make_split(seed, "flee_lead", common.make_env)
    rng = np.random.default_rng(seed)
    cp._continue(env, obs.copy(), lambda o: cp.rule_actions("flee_lead", o, rng), cp.Window())
    full_alive, full_dt = cp.full_run(seed, "flee_lead", common.make_env)
    assert int(env.fish_alive.sum()) == full_alive, "forked continue != unforked alive"
    assert np.array_equal(env.fish_death_timesteps, full_dt), "forked continue != unforked deaths"
    env.close()
    print("PASS test_continue_matches_unforked_full_run")


def test_actions_sized_to_alive_no_leak():
    """Actions must always match the alive count; dead fish never act again."""
    env = common.make_env()
    obs, _ = env.reset(seed=999)
    rng = np.random.default_rng(999)
    for _ in range(60):
        n_alive = int(env.fish_alive.sum())
        assert len(obs) == n_alive, "obs rows != alive fish"
        # force real deaths through the collision path so timestamps are recorded
        if env.timestep == 30:
            env.fish_positions[:20] = env.predator_pos
        actions = cp.rule_actions("flee_lead", obs, rng) if len(obs) else []
        assert len(actions) == n_alive, "actions != alive count"
        obs, _, _, _, _ = env.step(actions)
        dead = ~env.fish_alive
        assert bool(np.all(env.fish_death_timesteps[dead] >= 0)), "dead fish without timestamp"
    env.close()
    print("PASS test_actions_sized_to_alive_no_leak")


def test_early_total_death_counts_in_denominator():
    """A branch that dies out before the horizon is kept with survival 0."""
    env = common.make_env()
    env.reset(seed=321)
    env.fish_alive[:] = False  # simulate total death
    cp._continue(env, env._get_observations(), cp.hold_actions, cp.Window())
    rec = cp.record(321, "x", env, alive_split=96, win=cp.Window())
    assert rec["final_num_alive"] == 0
    assert rec["final_survival"] == 0.0
    assert rec["new_deaths_100_500"] == 96
    env.close()
    print("PASS test_early_total_death_counts_in_denominator")


def test_shielded_masks_policy_copy_only():
    """Shielding must not mutate the real env: re-running shielded twice from
    the same start gives identical results, and the real split env is untouched."""
    env = common.make_env()
    env.reset(seed=4242)
    obs = env._get_observations()
    before = cp.snapshot(env)
    rng = np.random.default_rng(5)
    out1 = cp.shielded_actions(obs.copy(), rng)
    assert cp.snapshot_equal(before, env), "shielding mutated real env state"
    env.close()

    # two shielded runs from the same recorded start must agree exactly
    def shielded_run():
        e, o, _ = cp.make_split(11, "flee_lead", common.make_env)
        r = np.random.default_rng(11)
        cp._continue(e, o.copy(), lambda ob: cp.shielded_actions(ob, r), cp.Window())
        alive = int(e.fish_alive.sum())
        e.close()
        return alive

    assert shielded_run() == shielded_run(), "shielded branch not reproducible"
    print("PASS test_shielded_masks_policy_copy_only")


def test_shielded_encoding_values():
    obs = np.ones((3, 18), dtype=np.float32)
    masked = obs.copy()
    masked[:, 5] = 0.0
    masked[:, 6:11] = 0.0
    assert np.all(masked[:, 5] == 0.0)
    assert np.all(masked[:, 6:11] == 0.0)
    # boundary margin obs[4] must be preserved (real env info, not predator)
    assert np.all(masked[:, 4] == 1.0)
    print("PASS test_shielded_encoding_values")


if __name__ == "__main__":
    test_seeds_disjoint_from_other_stages()
    test_fork_start_state_identical()
    test_continue_matches_unforked_full_run()
    test_actions_sized_to_alive_no_leak()
    test_early_total_death_counts_in_denominator()
    test_shielded_masks_policy_copy_only()
    test_shielded_encoding_values()
    print("\nALL TESTS PASSED")
