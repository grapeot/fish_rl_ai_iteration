#!/usr/bin/env python3
"""v51 meaning tests for the action-selection ablation.

These pin the RNG semantics the round depends on, using the real env, the real
frozen v49 checkpoints and a real random-init PPO (not hand-built arrays):

  1. same env(seed) + same action seed  -> identical stochastic action sequence
     and identical world trajectory,
  2. changing the action seed actually changes the sampled actions,
  3. the deterministic argmax is unchanged by the action seed / torch RNG state,
  4. the action RNG does not pollute the env initial state (world RNG is driven
     only by the scenario seed),
  5. the scenario-averaged pairing in v51_analyze matches an independent
     recomputation from the raw JSONL produced by a real evaluator run.

Run:
  experiments/v48/.venv/bin/python experiments/v51/tests/test_v51_semantics.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V51 = HERE.parent
ROOT = V51.parents[1]
for p in (str(ROOT), str(V51)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v51_common as common  # noqa: E402
from v51_eval import run_episode_policy, _make_base_env, _Track  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "v49"))
import common as common_v49  # noqa: E402  (v49 bank access for the overlap test)


def _load_model(name):
    from stable_baselines3 import PPO

    spec = common.MANIFEST_MODELS[name]
    return PPO.load(str(ROOT / spec["path"]), device="cpu")


def _random_policy(seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: _make_base_env(False)])
    model = PPO("MlpPolicy", vec, n_steps=8, batch_size=8, seed=seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    return model, vec


def _run_actions(model, env, scenario_seed, mode, action_seed, max_steps=60):
    """Return the full per-step action matrix (one list per step, alive fish)."""
    import torch
    obs, _info = env.reset(seed=scenario_seed)
    if mode == "stoch":
        torch.manual_seed(int(action_seed))
    seq = []
    steps = 0
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=(mode == "det"))
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        seq.append(actions)
        obs, _r, term, trunc, _info = env.step(actions)
        steps += 1
        if term or trunc or steps >= max_steps:
            break
    return seq


def test_stoch_same_seed_reproducible():
    model = _load_model("run1_final")
    env = _make_base_env(False)
    try:
        a = _run_actions(model, env, 510102, "stoch", 12345)
        b = _run_actions(model, env, 510102, "stoch", 12345)
    finally:
        env.close()
    assert a == b, "same scenario+action seed gave different stochastic actions"
    print(f"PASS test_stoch_same_seed_reproducible ({len(a)} steps identical)")


def test_action_seed_changes_actions():
    model, vec = _random_policy(7)
    env = _make_base_env(False)
    try:
        a = _run_actions(model, env, 510102, "stoch", 1, max_steps=40)
        b = _run_actions(model, env, 510102, "stoch", 2, max_steps=40)
    finally:
        env.close()
        vec.close()
    differs = any(x != y for x, y in zip(a, b))
    # a random-init categorical policy must react to the RNG seed
    variety = len({tuple(s) for s in a}) > 1
    assert variety, "random policy produced a constant action vector (unexpected)"
    assert differs, "changing the action seed did not change the sampled actions"
    print("PASS test_action_seed_changes_actions")


def test_det_independent_of_action_seed():
    import torch
    model = _load_model("run2_final")
    env = _make_base_env(False)
    try:
        torch.manual_seed(11)
        a = _run_actions(model, env, 510102, "det", None)
        torch.manual_seed(999999)
        [torch.rand(3) for _ in range(4)]  # perturb global RNG between runs
        b = _run_actions(model, env, 510102, "det", None)
    finally:
        env.close()
    assert a == b, "deterministic argmax changed when torch RNG state changed"
    print("PASS test_det_independent_of_action_seed")


def test_action_rng_does_not_pollute_env_reset():
    import torch
    seed = 510102
    env = _make_base_env(False)
    try:
        torch.manual_seed(1)
        env.reset(seed=seed)
        pos0 = env.fish_positions.copy()
        vel0 = env.fish_velocities.copy()
        # burn a lot of torch RNG, then reset the SAME scenario again
        torch.manual_seed(424242)
        [torch.rand(1000) for _ in range(3)]
        env.reset(seed=seed)
        pos1 = env.fish_positions.copy()
        vel1 = env.fish_velocities.copy()
        # also: run a stochastic episode, then reset the same scenario
        model = _load_model("run1_final")
        _run_actions(model, env, seed, "stoch", 777, max_steps=50)
        env.reset(seed=seed)
        pos2 = env.fish_positions.copy()
        vel2 = env.fish_velocities.copy()
    finally:
        env.close()
    assert np.array_equal(pos0, pos1) and np.array_equal(vel0, vel1), \
        "torch RNG state changed the env initial state"
    assert np.array_equal(pos0, pos2) and np.array_equal(vel0, vel2), \
        "a stochastic episode changed the next env reset's initial state"
    print("PASS test_action_rng_does_not_pollute_env_reset")


def test_action_stream_seeds_distinct():
    """The SeedSequence derivation gives distinct streams across episodes and
    replicates, and the fresh master differs from the debug master."""
    seeds = [common.action_stream_seed(e, r)
             for e in range(common.N_SCENARIOS) for r in common.REPLICATE_INDEX]
    assert len(seeds) == len(set(seeds)), "action-stream seeds collided"
    d0 = common.debug_action_stream_seed(0)
    assert d0 not in set(seeds), "debug stream collides with a fresh stream"
    # scenario seeds themselves must be distinct
    sc = common.episode_seeds(common.SCENARIO_RNG_SEED, common.N_SCENARIOS)
    assert len(sc) == len(set(sc)), "scenario seeds collided"
    # the fresh and v49 banks must not overlap
    v49 = set(common_v49.episode_seeds(482101, 20)) | set(common_v49.episode_seeds(482102, 40))
    assert not (set(sc) & v49), "v51 scenario seed overlaps a v49 bank"
    print(f"PASS test_action_stream_seeds_distinct ({len(seeds)} distinct streams)")


def test_det_and_stoch_actions_differ_on_trained_model():
    """The ablation must have power: on a trained model the argmax and the
    sampled actions are not the same sequence at the same scenario."""
    model = _load_model("run2_sel")
    env = _make_base_env(False)
    try:
        d = _run_actions(model, env, 510102, "det", None, max_steps=120)
        s = _run_actions(model, env, 510102, "stoch", 20240, max_steps=120)
    finally:
        env.close()
    assert d != s, "det and stoch produced identical action sequences (no power)"
    print(f"PASS test_det_and_stoch_actions_differ_on_trained_model ({len(d)} steps)")


def test_summary_paired_unit():
    """End-to-end: run the real evaluator on debug scenarios, run the real
    analyzer, and independently recompute the paired delta from the raw rows."""
    tmp = Path(tempfile.mkdtemp(prefix="v51_test_"))
    raw = tmp / "debug.jsonl"
    manifest = tmp / "manifest.json"
    analysis = tmp / "analysis.json"
    py = sys.executable
    cmd = [py, str(V51 / "v51_eval.py"), "--scenarios", "debug",
           "--models", "run1_final", "--arms", "det", "stoch",
           "--workers", "2", "--out", str(raw), "--manifest-out", str(manifest)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    cmd2 = [py, str(V51 / "v51_analyze.py"), "--raw", str(raw),
            "--manifest", str(manifest), "--out", str(analysis)]
    subprocess.run(cmd2, check=True, cwd=str(ROOT))

    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    det = {}
    stoch = {}
    for r in rows:
        if r.get("arm") == "det":
            det[r["scenario_index"]] = r["final_survival"]
        elif r.get("arm") == "stoch":
            stoch.setdefault(r["scenario_index"], {})[r["replicate"]] = r["final_survival"]
    # independent recomputation: average replicas within scenario, then pair
    diffs = []
    for si, reps in stoch.items():
        avg = float(np.mean(list(reps.values())))
        diffs.append(avg - det[si])
    expected = float(np.mean(diffs))
    got = json.loads(analysis.read_text())["models"]["run1_final"]["stoch_minus_det"]["mean_diff"]
    assert abs(expected - got) < 1e-12, f"paired unit mismatch: {expected} vs {got}"
    assert len(diffs) == len(det) == common.N_DEBUG_SCENARIOS
    print(f"PASS test_summary_paired_unit (paired delta {got:.6f} over {len(diffs)} scenarios)")


if __name__ == "__main__":
    test_stoch_same_seed_reproducible()
    test_action_seed_changes_actions()
    test_det_independent_of_action_seed()
    test_action_rng_does_not_pollute_env_reset()
    test_action_stream_seeds_distinct()
    test_det_and_stoch_actions_differ_on_trained_model()
    test_summary_paired_unit()
    print("\nALL V51 SEMANTICS TESTS PASSED")
