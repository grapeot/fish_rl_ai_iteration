#!/usr/bin/env python3
"""v52 tests: input-mask semantics, production predict-call capture, env/RNG
immutability, seed-bank validation, hash gate, rule parity, manifest exclusivity.

Run:
  experiments/v48/.venv/bin/python experiments/v52/current_baseline/tests/test_v52_mask.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
V50 = ROOT / "experiments" / "v50"
for p in (str(BASE), str(V50), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v50_common as common  # noqa: E402
import v50_evaluate as v50_eval  # noqa: E402
import v52_eval as v52  # noqa: E402
import v52_analyze as an  # noqa: E402


def _load_analysis_and_records():
    recs = an.load_records(BASE / "artifacts" / "raw_episodes.jsonl")
    man = json.loads((BASE / "artifacts" / "v52_spec_manifest.json").read_text())
    return an, recs, man


# -- mask semantics ----------------------------------------------------------
def test_mask_predator_zeroes_5_11_only():
    obs = np.arange(11, dtype=np.float32) + 1.0
    out = v52.apply_mask(obs, "mask_predator")
    assert np.array_equal(out[0:5], obs[0:5])
    assert np.all(out[5:11] == 0.0)
    print("PASS test_mask_predator_zeroes_5_11_only")


def test_mask_velocity_only_zeroes_8_10_only():
    obs = np.arange(11, dtype=np.float32) + 1.0
    out = v52.apply_mask(obs, "mask_velocity_only")
    assert np.array_equal(out[0:8], obs[0:8])
    assert np.all(out[8:10] == 0.0)
    assert out[10] == obs[10]
    print("PASS test_mask_velocity_only_zeroes_8_10_only")


def test_full_mode_is_identity_copy():
    obs = np.arange(11, dtype=np.float32) + 1.0
    out = v52.apply_mask(obs, "full")
    assert np.array_equal(out, obs)
    print("PASS test_full_mode_is_identity_copy")


def test_apply_mask_does_not_mutate_original():
    obs = np.arange(11, dtype=np.float32) + 1.0
    snap = obs.copy()
    v52.apply_mask(obs, "mask_predator")
    v52.apply_mask(obs, "mask_velocity_only")
    assert np.array_equal(obs, snap)
    print("PASS test_apply_mask_does_not_mutate_original")


def test_mask_predator_matches_env_natural_encoding_when_invisible():
    env = common.make_base_env(include_neighbor_features=False)
    try:
        obs, _ = env.reset(seed=520202)
        for _ in range(60):
            actions = np.full(int(obs.shape[0]), 4, dtype=np.int64)
            obs, _, term, trunc, _ = env.step(actions)
            if np.all(obs[:, 5] == 0.0):
                break
            if term or trunc:
                break
        invisible_rows = obs[:, 5] == 0.0
        assert invisible_rows.any()
        masked = v52.apply_mask(obs, "mask_predator")
        assert np.array_equal(masked[invisible_rows, 5:11], obs[invisible_rows, 5:11])
        assert np.all(masked[invisible_rows, 5:11] == 0.0)
        assert np.array_equal(masked[:, 0:5], obs[:, 0:5])
    finally:
        env.close()
    print("PASS test_mask_predator_matches_env_natural_encoding_when_invisible")


# -- env / RNG immutability --------------------------------------------------
def test_masking_does_not_change_env_or_rng_state():
    env = common.make_base_env(include_neighbor_features=False)
    try:
        obs, _ = env.reset(seed=520202)
        pos = env.fish_positions.copy()
        vel = env.fish_velocities.copy()
        alive = env.fish_alive.copy()
        ppos = env.predator_pos.copy()
        pvel = env.predator_vel.copy()
        ts = env.timestep
        rng_state = env.np_random.bit_generator.state
        v52.apply_mask(obs, "mask_predator")
        v52.apply_mask(obs, "mask_velocity_only")
        assert np.array_equal(env.fish_positions, pos)
        assert np.array_equal(env.fish_velocities, vel)
        assert np.array_equal(env.fish_alive, alive)
        assert np.array_equal(env.predator_pos, ppos)
        assert np.array_equal(env.predator_vel, pvel)
        assert env.timestep == ts
        assert env.np_random.bit_generator.state == rng_state
    finally:
        env.close()
    print("PASS test_masking_does_not_change_env_or_rng_state")


# -- production predict call site --------------------------------------------
class _SpyModel:
    def __init__(self, real):
        self.real = real
        self.seen = []

    def predict(self, obs, deterministic=True):
        self.seen.append(np.array(obs, copy=True))
        return self.real.predict(obs, deterministic=deterministic)


def test_production_predict_receives_masked_array_rep0():
    """Drive the real ``run_episode`` production chain (3 modes, 2 steps each)
    with a spy at ``policy.model.predict``; assert the exact masked array is the
    one predict receives, and the original observation is retained (not zeroed
    in place)."""
    from stable_baselines3 import PPO

    path = ROOT / v52.PPO_MODEL["rep0_survival_only_final"]["checkpoint"]
    model = PPO.load(str(path), device="cpu")
    captured = []

    def capture(orig, masked):
        captured.append((np.array(orig, copy=True), np.array(masked, copy=True)))

    for mode, sl in (("mask_predator", slice(5, 11)),
                     ("mask_velocity_only", slice(8, 10)),
                     ("full", slice(5, 11))):
        captured.clear()
        env = common.make_base_env(include_neighbor_features=False)
        try:
            spy = _SpyModel(model)
            policy = v52.MaskedPolicy(spy, mode, controller_name="rep0_survival_only_final",
                                      capture=capture)
            rec = v52.run_episode(mode, env, 520202, 0, policy=policy, max_steps=2)
            assert rec["controller"] == "rep0_survival_only_final"
            assert rec["mode"] == mode
            assert len(captured) == 2
            for orig, masked in captured:
                assert masked is not None
                # predict received exactly this masked array (spy captures it)
                assert np.array_equal(spy.seen.pop(0), masked)
                # own channels 0:5 preserved on the real call
                assert np.array_equal(orig[:, 0:5], masked[:, 0:5])
                # masked slice zeroed; full leaves everything intact
                if mode != "full":
                    assert np.all(masked[:, sl] == 0.0)
                else:
                    assert np.array_equal(orig, masked)
                # the original observation was NOT zeroed in place: where the
                # predator was visible, its real (non-zero) channels survive
                vis = orig[:, 5] > 0.5
                if vis.any() and mode != "full":
                    assert not np.array_equal(orig[vis][:, sl], masked[vis][:, sl])
        finally:
            env.close()
    print("PASS test_production_predict_receives_masked_array_rep0")


def test_mask_velocity_only_preserves_visible_channels():
    obs = np.zeros((4, 11), dtype=np.float32)
    obs[:, 5] = 1.0
    obs[:, 6:8] = 0.3
    obs[:, 8:10] = 0.7
    obs[:, 10] = 0.5
    out = v52.apply_mask(obs, "mask_velocity_only")
    assert np.array_equal(out[:, 5:8], obs[:, 5:8])
    assert np.array_equal(out[:, 10], obs[:, 10])
    assert np.all(out[:, 8:10] == 0.0)
    print("PASS test_mask_velocity_only_preserves_visible_channels")


# -- seed bank validation ----------------------------------------------------
def test_scenario_bank_frozen_and_disjoint():
    seeds = v52.scenario_seeds(v52.SEED_RNG, v52.N_SCENARIOS)
    assert len(seeds) == 40
    assert len(set(seeds)) == 40
    sel = set(common.resolve_seeds("selection"))
    rep = set(common.resolve_seeds("report"))
    assert not (set(seeds) & (sel | rep))
    debug = v52.scenario_seeds(v52.DEBUG_SEED_RNG, v52.DEBUG_N_SCENARIOS)
    assert len(debug) == 3 and len(set(debug)) == 3
    assert not (set(debug) & set(seeds))
    print(f"PASS test_scenario_bank_frozen_and_disjoint first={seeds[0]}")


def test_duplicate_seed_rejected():
    try:
        v52.validate_scenarios([1, 2, 2, 3])
    except ValueError:
        print("PASS test_duplicate_seed_rejected")
        return
    raise AssertionError("duplicate seeds were not rejected")


def test_empty_seed_rejected():
    try:
        v52.validate_scenarios([])
    except ValueError:
        print("PASS test_empty_seed_rejected")
        return
    raise AssertionError("empty seed bank was not rejected")


def test_wrong_hash_rejected():
    real = v52.PPO_MODEL["rep0_survival_only_final"]["checkpoint"]
    try:
        v52.check_model_hashes({"rep0_survival_only_final": real},
                               {"rep0_survival_only_final": "0" * 64})
    except ValueError:
        print("PASS test_wrong_hash_rejected")
        return
    raise AssertionError("wrong hash was not rejected")


def test_real_hashes_match_frozen():
    got = v52.check_model_hashes(
        {n: v52.PPO_MODEL[n]["checkpoint"] for n in v52.PPO_CONTROLLERS},
        {n: v52.PPO_MODEL[n]["final_policy_tensor_sha256"] for n in v52.PPO_CONTROLLERS},
    )
    assert set(got) == set(v52.PPO_CONTROLLERS)
    print("PASS test_real_hashes_match_frozen")


# -- rule parity with v50 ----------------------------------------------------
def test_rule_flee_lead_matches_v50_logic():
    rng50 = np.random.default_rng(7)
    rng_used = np.random.default_rng(7)
    for _ in range(200):
        obs = rng50.uniform(-1, 1, size=11)
        obs[5] = 1.0 if rng50.random() > 0.5 else 0.0
        expect = v50_eval.rule_action("flee_lead", obs, rng_used)
        assert v52.rule_flee_lead(obs) == expect
    print("PASS test_rule_flee_lead_matches_v50_logic")


def test_rule_mask_positive_control_behaviour():
    obs = np.zeros(11, dtype=np.float32)
    obs[4] = 0.5  # away from boundary so the predator branch governs
    obs[5] = 1.0
    obs[6], obs[7] = 0.2, 0.0
    obs[8], obs[9] = 0.1, 0.0
    assert v52.rule_flee_lead(obs) != 4
    masked = v52.apply_mask(obs, "mask_predator")
    assert v52.rule_flee_lead(masked) == 4
    print("PASS test_rule_mask_positive_control_behaviour")


# -- manifest exclusivity ----------------------------------------------------
def test_manifest_not_overwritable():
    seeds = v52.scenario_seeds(v52.DEBUG_SEED_RNG, v52.DEBUG_N_SCENARIOS)
    m = v52.build_manifest(seeds, debug=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "spec.json"
        v52.write_manifest_exclusive(p, m)
        assert json.loads(p.read_text())["kind"] == "v52_spec_manifest"
        try:
            v52.write_manifest_exclusive(p, m)
        except FileExistsError:
            print("PASS test_manifest_not_overwritable")
            return
        raise AssertionError("manifest was overwritten")


# -- analysis strict gate ----------------------------------------------------
def test_strict_gate_rejects_missing_and_duplicate(bundle):
    an_mod, records, manifest = bundle
    an_mod.validate_records_strict(records, manifest)  # valid set passes
    dup = records[:-1] + [records[0]]
    try:
        an_mod.validate_records_strict(dup, manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate not rejected")
    try:
        an_mod.validate_records_strict(records[:-1], manifest)
    except ValueError:
        print("PASS test_strict_gate_rejects_missing_and_duplicate")
        return
    raise AssertionError("missing not rejected")


# -- skip-manifest read-back guard -------------------------------------------
def test_skip_manifest_readback_guard():
    seeds = v52.scenario_seeds(v52.DEBUG_SEED_RNG, v52.DEBUG_N_SCENARIOS)
    m = v52.build_manifest(seeds, debug=True)
    rep = v52.verify_existing_manifest(m, v52.build_manifest(seeds, debug=True))
    assert rep["script_sha256_matches"]
    other = v52.build_manifest(v52.scenario_seeds(v52.DEBUG_SEED_RNG, 4), debug=True)
    try:
        v52.verify_existing_manifest(other, m)
    except ValueError:
        print("PASS test_skip_manifest_readback_guard")
        return
    raise AssertionError("differing spec was not rejected")


# -- matched-observation diagnostic ------------------------------------------
def _matched_obs_diagnostic_impl(seeds, models):
    """Reference reimplementation used to cross-check the module output."""
    from stable_baselines3 import PPO

    chunks = []
    for s in seeds:
        e = common.make_base_env(include_neighbor_features=False)
        try:
            o, _ = e.reset(seed=int(s))
            chunks.append(np.asarray(o, dtype=np.float32))
        finally:
            e.close()
    base = np.concatenate(chunks, axis=0)
    out = {}
    for name, path in models.items():
        model = PPO.load(str(path), device="cpu")
        ba = np.asarray(model.predict(base, deterministic=True)[0])
        ent = {}
        for mode in ("mask_predator", "mask_velocity_only"):
            a = np.asarray(model.predict(v52.apply_mask(base, mode), deterministic=True)[0])
            ent[mode] = int((a != ba).sum())
        out[name] = ent
    return base, out


def test_matched_obs_diagnostic_matches_reference():
    man = json.loads((BASE / "artifacts" / "v52_spec_manifest.json").read_text())
    seeds = man["scenarios"]
    models = {n: ROOT / man["models"][n]["checkpoint"] for n in v52.PPO_CONTROLLERS}
    base, ref = _matched_obs_diagnostic_impl(seeds, models)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "diag.json"
        diag = an.matched_observation_diagnostic(man, p)
    assert diag["n_observations"] == base.shape[0]
    assert diag["n_visible"] == int((base[:, 5] > 0.5).sum())
    for name in v52.PPO_CONTROLLERS:
        for mode in ("mask_predator", "mask_velocity_only"):
            assert diag["models"][name][mode]["changes"] == ref[name][mode], (name, mode)
    print(f"PASS test_matched_obs_diagnostic_matches_reference rows={diag['n_observations']} "
          f"visible={diag['n_visible']}")


if __name__ == "__main__":
    test_mask_predator_zeroes_5_11_only()
    test_mask_velocity_only_zeroes_8_10_only()
    test_full_mode_is_identity_copy()
    test_apply_mask_does_not_mutate_original()
    test_mask_predator_matches_env_natural_encoding_when_invisible()
    test_masking_does_not_change_env_or_rng_state()
    test_mask_velocity_only_preserves_visible_channels()
    test_scenario_bank_frozen_and_disjoint()
    test_duplicate_seed_rejected()
    test_empty_seed_rejected()
    test_wrong_hash_rejected()
    test_real_hashes_match_frozen()
    test_rule_flee_lead_matches_v50_logic()
    test_rule_mask_positive_control_behaviour()
    test_manifest_not_overwritable()
    test_skip_manifest_readback_guard()
    test_production_predict_receives_masked_array_rep0()
    _an, _recs, _man = _load_analysis_and_records()
    test_strict_gate_rejects_missing_and_duplicate((_an, _recs, _man))
    test_matched_obs_diagnostic_matches_reference()
    print("\nALL V52 TESTS PASSED")
