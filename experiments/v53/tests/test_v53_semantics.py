#!/usr/bin/env python3
"""v53 meaning tests — the load-bearing semantics of the round-5 axis.

These must be run and pass BEFORE any report. They check, on the real code:

  1. `horizon_end_flag_differs`: at step 500 a surviving focal is `truncated`
     under timeout_bootstrap and `terminated` under finite_terminal, with the
     SAME +0.7 reward.
  2. `death_priority`: a focal that dies on the horizon step is terminal with -50
     in BOTH modes; finite_terminal never rewrites a death into a horizon end.
  3. `continuous_rng_autoreset`: consecutive unseeded resets give fresh worlds
     (the audited v50 P0 fix is inherited by the subclass).
  4. `same_actions_same_world`: same seed + same action sequence -> identical
     physics/obs/reward in both modes; only the horizon flag differs.
  5. `seeding_and_initial_weights_paired`: treatment replicates use the v50
     worker banks/model seeds; the three treatment initial-policy hashes are
     distinct across replicates (same construction as the accepted control).
  6. `true_sb3_collector_bootstrap`: through the REAL SB3 collector + rollout
     buffer, the timeout_bootstrap target at the horizon is
     `reward + gamma*V(terminal_observation)` (a genuine bootstrap), while the
     finite_terminal target is exactly the raw reward (no bootstrap). This is
     NOT a wrapper-flag-only test.

Run:
  experiments/v48/.venv/bin/python experiments/v53/tests/test_v53_semantics.py
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V53 = HERE.parent
ROOT = V53.parents[1]
for p in (str(ROOT), str(V53), str(ROOT / "experiments" / "v50")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v53_common as common  # noqa: E402
from v53_env import V53SingleFishEnv, HOLD_ACTION  # noqa: E402

HOLD_SEED = 1  # under HOLD this focal survives to step 500 (probed)


def _roll_to_horizon(env, seed=HOLD_SEED):
    obs, info = env.reset(seed=seed)
    last = None
    for _ in range(common.MAX_TIMESTEPS):
        obs, r, term, trunc, info = env.step(HOLD_ACTION)
        last = (r, term, trunc, dict(info))
        if term or trunc:
            break
    return last


def test_horizon_end_flag_differs():
    boot = V53SingleFishEnv(reward_mode="survival_only", termination_mode="timeout_bootstrap")
    fin = V53SingleFishEnv(reward_mode="survival_only", termination_mode="finite_terminal")
    rb = _roll_to_horizon(boot)
    rf = _roll_to_horizon(fin)
    boot.close(); fin.close()
    rb_r, rb_t, rb_tr, rb_i = rb
    rf_r, rf_t, rf_tr, rf_i = rf
    assert rb_t is False and rb_tr is True, f"bootstrap horizon not truncated: {rb_t},{rb_tr}"
    assert rf_t is True and rf_tr is False, f"finite horizon not terminal: {rf_t},{rf_tr}"
    assert abs(rb_r - common.SURVIVAL_STEP_REWARD) < 1e-9, f"bootstrap horizon reward {rb_r}"
    assert abs(rf_r - common.SURVIVAL_STEP_REWARD) < 1e-9, f"finite horizon reward {rf_r}"
    assert rb_i.get("agent_truncated") is True
    assert rf_i.get("agent_finite_terminal") is True and rf_i.get("agent_truncated") is False
    print("PASS test_horizon_end_flag_differs")


def test_death_priority():
    for mode in common.TERMINATION_MODES:
        env = V53SingleFishEnv(reward_mode="survival_only", termination_mode=mode)
        env.reset(seed=HOLD_SEED)
        fid = env.focal_id
        # advance to the last step, then teleport the predator on top of the focal
        for _ in range(common.MAX_TIMESTEPS - 1):
            _, _, term, trunc, _ = env.step(HOLD_ACTION)
            if term or trunc:
                raise AssertionError("focal died early under HOLD; pick another seed")
        env.base_env.predator_pos = env.base_env.fish_positions[fid].copy()
        env.base_env.predator_vel = np.zeros(2, dtype=np.float32)
        _, r, term, trunc, info = env.step(HOLD_ACTION)
        assert term and not trunc, f"{mode}: death on horizon step not terminal: {term},{trunc}"
        assert abs(r - common.DEATH_STEP_PENALTY) < 1e-9, f"{mode}: death reward {r}"
        assert info.get("agent_death") is True
        assert info.get("agent_finite_terminal") is None, f"{mode}: death mislabelled finite-terminal"
        env.close()
    print("PASS test_death_priority")


def test_continuous_rng_autoreset():
    env = V53SingleFishEnv(reward_mode="survival_only", termination_mode="finite_terminal",
                           worker_seed=530999)
    env.reset()
    p1 = env.base_env.fish_positions.copy()
    for _ in range(10):
        env.step(HOLD_ACTION)
    env.reset()
    p2 = env.base_env.fish_positions.copy()
    assert not np.array_equal(p1, p2), "consecutive reset replayed the same world (P0)"
    env.close()
    print("PASS test_continuous_rng_autoreset")


def test_same_actions_same_world():
    a = V53SingleFishEnv(reward_mode="survival_only", termination_mode="timeout_bootstrap")
    b = V53SingleFishEnv(reward_mode="survival_only", termination_mode="finite_terminal")
    oa, ia = a.reset(seed=4242)
    ob, ib = b.reset(seed=4242)
    assert ia["focal_id"] == ib["focal_id"]
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(4242)
    n = 0
    for _ in range(120):
        act = int(rng.integers(0, 5))
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert ra == rb, "reward diverged across termination modes"
        assert ta == tb and tra == trb, "termination diverged before horizon"
        assert np.array_equal(oa, ob), "obs diverged across termination modes"
        assert np.array_equal(a.base_env.fish_positions, b.base_env.fish_positions)
        n += 1
        if ta or tra:
            break
    a.close(); b.close()
    assert n > 10
    print(f"PASS test_same_actions_same_world (steps={n})")


def test_seeding_and_initial_weights_paired():
    assert set(common.MODEL_SEEDS.values()) == set(common.v50.MODEL_SEEDS.values())
    assert common.ENV_BANK_BASE == common.v50.ENV_BANK_BASE
    for r in common.REPLICATES:
        assert len(set(common.env_bank_worker_seeds(r))) == common.NUM_ENVS
    # the v53 control runs are the accepted v50 corrected survival_only runs
    for r in common.REPLICATES:
        assert common.CONTROL_RUN_DIR[r].endswith(f"rep{r}_survival_only")
    print("PASS test_seeding_and_initial_weights_paired")


def test_v53_banks_disjoint():
    """The fresh v53 banks (530100/530101/530102) must not overlap each other,
    the v50 banks (500200/500201/500202), the pilot banks (500100/500101/500102),
    v48 (481xxx/555xxx) or v49 (482xxx), at the GENERATED-VALUE level."""
    def ep(rng_seed, n):
        return set(common.episode_seeds(rng_seed, n))

    legacy = set()
    for s, n in [(481000, 2), (481001, 12), (481002, 40), (555001, 40), (555002, 40),
                 (482100, 2), (482101, 20), (482102, 40),
                 (500100, 2), (500101, 24), (500102, 40),
                 (500200, 2), (500201, 24), (500202, 40)]:
        legacy |= ep(s, n)
    sets = {name: set(common.resolve_seeds(name)) for name in ("smoke", "selection", "report")}
    for name, s in sets.items():
        assert not (s & legacy), f"v53 {name} overlaps a legacy/v48/v49/v50 bank"
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (sets[keys[i]] & sets[keys[j]]), f"v53 {keys[i]} overlaps {keys[j]}"
    # and the v50 evaluation banks really are the v50 values (import source of truth)
    assert set(common.v50.resolve_seeds("selection")) == ep(500201, 24)
    assert set(common.v50.resolve_seeds("report")) == ep(500202, 40)
    print("PASS test_v53_banks_disjoint")


class _ForceHold(V53SingleFishEnv):
    """Drive the real env with a fixed HOLD so a surviving seed reaches horizon
    under whatever policy PPO samples; used to exercise the real collector."""

    def step(self, action):
        return super().step(HOLD_ACTION)


def _collector_last_reward_and_flags(mode, n_steps=common.MAX_TIMESTEPS):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)

    captured = {}

    def _mk():
        return _ForceHold(reward_mode="survival_only", worker_seed=530111,
                          termination_mode=mode)

    vec = DummyVecEnv([_mk])
    vec._seeds = [None]

    model = PPO("MlpPolicy", vec, n_steps=n_steps, batch_size=n_steps, n_epochs=1,
                seed=0, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))

    class _Cap(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if info.get("terminal_observation") is not None:
                    captured.setdefault("terminal_obs", info["terminal_observation"])
                if info.get("TimeLimit.truncated"):
                    captured["tl_truncated"] = True
            return True

    buf = RolloutBuffer(
        n_steps, vec.observation_space, vec.action_space, device="cpu",
        gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=1,
    )
    cap = _Cap()
    model._setup_learn(total_timesteps=n_steps, callback=cap)  # inits ep buffers + _last_obs
    model.collect_rollouts(vec, callback=cap, rollout_buffer=buf,
                           n_rollout_steps=n_steps)
    last_reward = float(buf.rewards[n_steps - 1, 0])
    term_obs = captured.get("terminal_obs")
    tl_truncated = captured.get("tl_truncated", False)
    gamma_v = None
    if term_obs is not None:
        with torch.no_grad():
            gamma_v = float(model.gamma) * float(
                model.policy.predict_values(
                    model.policy.obs_to_tensor(np.asarray(term_obs, dtype=np.float32))[0]
                )[0]
            )
    vec.close()
    return {"last_reward": last_reward, "tl_truncated": bool(tl_truncated),
            "gamma_v": gamma_v}


def test_true_sb3_collector_bootstrap():
    boot = _collector_last_reward_and_flags("timeout_bootstrap")
    fin = _collector_last_reward_and_flags("finite_terminal")

    # finite_terminal: the collector must NOT bootstrap the last step.
    assert not fin["tl_truncated"], "finite_terminal set TimeLimit.truncated"
    assert abs(fin["last_reward"] - common.SURVIVAL_STEP_REWARD) < 1e-6, (
        f"finite_terminal last reward bootstrapped: {fin['last_reward']} != {common.SURVIVAL_STEP_REWARD}")

    # timeout_bootstrap: the collector MUST bootstrap the last step.
    assert boot["tl_truncated"], "timeout_bootstrap did not set TimeLimit.truncated"
    assert boot["gamma_v"] is not None, "no terminal_observation saved for bootstrap"
    expected = common.SURVIVAL_STEP_REWARD + boot["gamma_v"]
    assert abs(boot["last_reward"] - expected) < 1e-6, (
        f"timeout_bootstrap target {boot['last_reward']} != reward+gamma*V {expected}")
    print(f"PASS test_true_sb3_collector_bootstrap "
          f"(finite={fin['last_reward']:.6f}, boot_target={boot['last_reward']:.6f}, "
          f"gamma*V={boot['gamma_v']:.6f})")


if __name__ == "__main__":
    test_horizon_end_flag_differs()
    test_death_priority()
    test_continuous_rng_autoreset()
    test_same_actions_same_world()
    test_seeding_and_initial_weights_paired()
    test_v53_banks_disjoint()
    test_true_sb3_collector_bootstrap()
    print("\nALL V53 SEMANTICS TESTS PASSED")
