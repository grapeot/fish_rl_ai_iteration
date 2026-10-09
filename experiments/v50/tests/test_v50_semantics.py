#!/usr/bin/env python3
"""v50 tests: reward-mode equivalence, survival_only signal, world invariance,
seed disjointness, post-update checkpoint hook, and the report/selection gate.

Run:
  experiments/v48/.venv/bin/python experiments/v50/tests/test_v50_semantics.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V50 = HERE.parent
ROOT = V50.parents[1]
V49 = ROOT / "experiments" / "v49"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V50))
# v49 on the path so `import single_fish_env` resolves to the v49 module.
sys.path.insert(0, str(V49))

import v50_common as common  # noqa: E402
from v50_env import V50SingleFishEnv, HOLD_ACTION  # noqa: E402
import single_fish_env as v49_env  # noqa: E402  (the v49 module)


def test_original_reward_bit_identical_to_v49():
    """`original` mode must reproduce the v49 wrapper's reward/obs/termination
    exactly, on the same seed and the same action sequence."""
    for seed in (111, 222, 333):
        a = v49_env.SingleFishControlEnv(include_neighbor_features=False)
        b = V50SingleFishEnv(include_neighbor_features=False, reward_mode="original")
        oa, ia = a.reset(seed=seed)
        ob, ib = b.reset(seed=seed)
        assert ia["focal_id"] == ib["focal_id"], "focal id differs"
        assert np.array_equal(oa, ob), "reset obs differs"
        rng = np.random.default_rng(seed + 99)
        for step in range(80):
            act = int(rng.integers(0, 5))
            oa, ra, ta, tra, _ = a.step(act)
            ob, rb, tb, trb, _ = b.step(act)
            assert ra == rb, f"reward differs at step {step}: {ra} vs {rb}"
            assert ta == tb and tra == trb, "termination/truncation differs"
            assert np.array_equal(oa, ob), "obs differs"
            if ta or tra:
                break
        a.close(); b.close()
    print("PASS test_original_reward_bit_identical_to_v49")


def test_survival_only_signal():
    """survival_only: +0.7 on alive steps, one-shot -50 on the focal death step,
    never repeated. Force death by teleporting the predator onto the focal fish."""
    env = V50SingleFishEnv(include_neighbor_features=False, reward_mode="survival_only")
    env.reset(seed=3003)
    fid = env.focal_id
    # alive step: HOLD far from predator -> +0.7
    for _ in range(5):
        _, r, term, trunc, _ = env.step(HOLD_ACTION)
        assert not term and not trunc
        assert abs(r - common.SURVIVAL_STEP_REWARD) < 1e-9, f"alive reward {r}"
    env.base_env.predator_pos = env.base_env.fish_positions[fid].copy()
    env.base_env.predator_vel = np.zeros(2, dtype=np.float32)
    _, r, term, trunc, info = env.step(HOLD_ACTION)
    assert term and not trunc and info["agent_death"]
    assert abs(r - common.DEATH_STEP_PENALTY) < 1e-9, f"death reward {r}"
    env.close()
    print("PASS test_survival_only_signal")


def test_reward_modes_same_world_trajectory():
    """Same seed + same action sequence -> identical physics/obs/termination in
    both modes; only the scalar reward differs."""
    a = V50SingleFishEnv(include_neighbor_features=False, reward_mode="original")
    b = V50SingleFishEnv(include_neighbor_features=False, reward_mode="survival_only")
    oa, ia = a.reset(seed=4242)
    ob, ib = b.reset(seed=4242)
    assert ia["focal_id"] == ib["focal_id"]
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(4242)
    n = 0
    for step in range(120):
        act = int(rng.integers(0, 5))
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert ta == tb and tra == trb, "termination diverged across reward modes"
        assert np.array_equal(oa, ob), "obs diverged across reward modes"
        assert np.array_equal(a.base_env.fish_positions, b.base_env.fish_positions)
        assert np.array_equal(a.base_env.fish_alive, b.base_env.fish_alive)
        n += 1
        if ta or tra:
            break
    assert n > 10
    a.close(); b.close()
    print(f"PASS test_reward_modes_same_world_trajectory (steps={n})")


def test_seed_banks_disjoint():
    def ep(rng_seed, n):
        rng = np.random.default_rng(rng_seed)
        return set(int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n))

    old = set()
    for s, n in [(481000, 2), (481001, 12), (481002, 40), (555001, 40),
                 (555002, 40), (482100, 2), (482101, 20), (482102, 40),
                 (500100, 2), (500101, 24), (500102, 40)]:
        old |= ep(s, n)
    old |= set(range(4901001, 4901009))
    sets = {name: set(common.resolve_seeds(name)) for name in ("smoke", "selection", "report")}
    sets["train_workers"] = set()
    for r in common.REPLICATES:
        sets["train_workers"] |= set(common.env_bank_worker_seeds(r))
    for name, s in sets.items():
        assert not (s & old), f"{name} overlaps legacy/v48/v49 bank"
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (sets[keys[i]] & sets[keys[j]]), f"{keys[i]} overlaps {keys[j]}"
    for r in common.REPLICATES:
        assert len(set(common.env_bank_worker_seeds(r))) == common.NUM_ENVS
    assert len(set(common.MODEL_SEEDS.values())) == len(common.REPLICATES)
    print("PASS test_seed_banks_disjoint")


def test_checkpoint_hook_counts_post_update():
    """The PPO subclass must record one snapshot per completed update and save a
    checkpoint whose label equals the number of completed updates."""
    import v50_train

    from stable_baselines3.common.vec_env import DummyVecEnv

    V50PPO = v50_train.make_ppo_subclass()
    vec = DummyVecEnv([lambda: V50SingleFishEnv(reward_mode="survival_only", worker_seed=999001)])
    model = V50PPO("MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1, seed=0,
                   device="cpu", policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))
    model._v50_updates = []
    import time as _t
    model._v50_t0 = _t.time()
    model._v50_checkpoint_iters = set()
    model._v50_checkpoint_dir = tempfile.gettempdir()
    model._v50_ckpt_meta = []
    model.learn(total_timesteps=24, progress_bar=False)
    assert len(model._v50_updates) == 3, f"expected 3 update snapshots, got {len(model._v50_updates)}"
    assert [u["update"] for u in model._v50_updates] == [1, 2, 3]
    assert model._n_updates == 3
    vec.close()
    print("PASS test_checkpoint_hook_counts_post_update")


def test_report_gate_requires_manifest():
    """v50_evaluate must refuse to run a report when the selection manifest is
    absent."""
    missing = Path(tempfile.gettempdir()) / "v50_definitely_absent_manifest.json"
    if missing.exists():
        missing.unlink()
    cmd = [
        str(V50.parents[1] / "experiments" / "v48" / ".venv" / "bin" / "python"),
        str(V50 / "v50_evaluate.py"),
        "--seeds", "smoke", "--arm", "rule_hold",
        "--out", str(Path(tempfile.gettempdir()) / "v50_gate_probe.jsonl"),
        "--require-manifest", str(missing),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode != 0, "evaluate ran despite missing manifest"
    assert "selection manifest" in (res.stderr + res.stdout), "gate message missing"
    print("PASS test_report_gate_requires_manifest")


def test_consecutive_resets_fresh_world():
    """P0 regression: on a single env instance, a later `reset()` (seed=None,
    as in SB3/SubprocVecEnv autoreset) must produce a FRESH world/focal, not
    replay the worker's initial one."""
    env = V50SingleFishEnv(reward_mode="survival_only", worker_seed=999777)
    o1, i1 = env.reset()
    p1 = env.base_env.fish_positions.copy()
    for _ in range(10):
        env.step(HOLD_ACTION)
    o2, i2 = env.reset()  # no seed -> must continue RNG, not re-seed
    p2 = env.base_env.fish_positions.copy()
    assert not np.array_equal(p1, p2), "consecutive reset replayed the same world (P0)"
    env.close()
    print("PASS test_consecutive_resets_fresh_world")


def test_explicit_seed_reproduces_episode():
    """An explicit `reset(seed=...)` must reproduce that exact episode, even
    after a later unseeded reset on the same instance."""
    a = V50SingleFishEnv(reward_mode="original")
    oa, ia = a.reset(seed=123456)
    for _ in range(20):
        a.step(HOLD_ACTION)
    oa2, ia2 = a.reset(seed=123456)
    assert ia2["focal_id"] == ia["focal_id"]
    assert np.array_equal(oa2, oa), "explicit seed did not reproduce the episode"
    a.close()
    print("PASS test_explicit_seed_reproduces_episode")


def test_same_seed_same_actions_reproducible():
    """Two envs with the same worker seed and the same action sequence must be
    bit-identical (paired control), for a full episode."""
    a = V50SingleFishEnv(reward_mode="survival_only", worker_seed=555111)
    b = V50SingleFishEnv(reward_mode="survival_only", worker_seed=555111)
    oa, _ = a.reset()
    ob, _ = b.reset()
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(7)
    for _ in range(80):
        act = int(rng.integers(0, 5))
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert ra == rb and ta == tb and tra == trb
        assert np.array_equal(oa, ob)
        if ta or tra:
            break
    a.close(); b.close()
    print("PASS test_same_seed_same_actions_reproducible")


def test_subproc_autoreset_fresh_episodes():
    """Real SubprocVecEnv: worker 0's pinned initial must be reproducible, the
    two workers must start from different worlds, and after the first episode
    auto-resets the next episode must be a FRESH world (not a replay)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def f0():
        import torch

        torch.set_num_threads(1)
        return V50SingleFishEnv(reward_mode="survival_only", worker_seed=6000011)

    def f1():
        import torch

        torch.set_num_threads(1)
        return V50SingleFishEnv(reward_mode="survival_only", worker_seed=6000012)

    vec = SubprocVecEnv([f0, f1])
    vec._seeds = [None, None]  # trainer policy: wrapper owns the per-worker seed
    obs = np.asarray(vec.reset())
    first = obs.copy()
    assert not np.allclose(first[0], first[1]), "workers share the same initial world"

    ref = V50SingleFishEnv(reward_mode="survival_only", worker_seed=6000011)
    o_ref, _ = ref.reset()
    ref.close()
    assert np.allclose(first[0], o_ref), "worker0 initial is not reproducible from its seed"

    done = False
    steps = 0
    while not done and steps < 600:
        obs, _rew, dones, _infos = vec.step(np.array([HOLD_ACTION, HOLD_ACTION]))
        steps += 1
        if dones[0]:
            done = True
    second0 = np.asarray(obs)[0]
    assert not np.allclose(second0, first[0]), "autoreset replayed the same initial world (P0)"
    vec.close()
    print(f"PASS test_subproc_autoreset_fresh_episodes (steps={steps})")


if __name__ == "__main__":
    test_original_reward_bit_identical_to_v49()
    test_survival_only_signal()
    test_reward_modes_same_world_trajectory()
    test_seed_banks_disjoint()
    test_checkpoint_hook_counts_post_update()
    test_report_gate_requires_manifest()
    test_consecutive_resets_fresh_world()
    test_explicit_seed_reproduces_episode()
    test_same_seed_same_actions_reproducible()
    test_subproc_autoreset_fresh_episodes()
    print("\nALL V50 TESTS PASSED")
