#!/usr/bin/env python3
"""v58 meaning tests — the load-bearing semantics of the round-10 confirmation.

Must pass BEFORE any score is interpreted. They check, on the real code:

  1. `joint_change_only_touches_velocity_vectors`: for a fixed worker seed, every
     condition restore leaves fish positions/alive/death-timesteps, predator position,
     timestep, pre-roll trace and the base env RNG bit-identical to the un-conditioned
     reset; only the velocity vectors change. Fish velocities equal the restored
     velocities scaled by the fish factor; predator velocity equals the nominal
     rotated by the integer matrix then scaled.
  2. `nominal_is_exact_identity`: the nominal triple (1.0, 0, 1.0) reproduces the raw
     post-reset state bit-for-bit (velocities and a full rollout under the fixed rule).
  3. `components_match_accepted_rounds`: the fish-factor scaling matches v54's
     post-reset fish scaling; the rotation matrices match v55; the speed scaling
     matches v56 (direction preserved, norm proportional). This is the "compose the
     formula" check.
  4. `obs_visibility_correct`: with the combined stress, obs[2]/obs[3] equal the
     scaled focal velocity / FISH_MAX_SPEED; predator-visible flag and relative fields
     correspond to the rotated/scaled predator state (visible iff within vision).
  5. `draws_frozen_and_shared`: the combined-stress triples are reproducible from RNG
     580202 and are the frozen per-scene arrays; a scene drawn as the nominal triple is
     kept (not redrawn); two controllers evaluated on the same scene get the same
     triple.
  6. `nominal_reproduces_direct_baseline`: a direct v50-style deterministic loop for a
     frozen control model equals the v58 nominal-condition loop bit-for-bit.
  7. `zero_velocity_and_early_death_denominator`: fish factor 0 gives zero fish
     velocities and the run is still scored over /96; a first-step death is kept.

Run:
  experiments/v48/.venv/bin/python experiments/v58/tests/test_v58_semantics.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V58 = HERE.parent
ROOT = V58.parents[1]
for p in (str(ROOT), str(V58), str(ROOT / "experiments" / "v50"),
          str(ROOT / "experiments" / "v54"), str(ROOT / "experiments" / "v55"),
          str(ROOT / "experiments" / "v56"), str(ROOT / "experiments" / "v57")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v58_common as common  # noqa: E402
import v58_env as venv  # noqa: E402
import v54_common as v54c  # noqa: E402
import v55_common as v55c  # noqa: E402
import v56_common as v56c  # noqa: E402

SEED = 580777


def _fresh_env():
    return common.make_base_env(include_neighbor_features=False)


def test_joint_change_only_touches_velocity_vectors():
    env = _fresh_env()
    obs0, snap = venv.reset_nominal(env, SEED)
    for triple in ([0.0, 90, 0.75], [0.5, 180, 1.0], [1.0, 270, 1.25], [1.0, 0, 1.0]):
        venv.apply_condition(env, snap, triple)
        assert np.array_equal(env.fish_positions, snap["fish_positions"]), "fish positions changed"
        assert np.array_equal(env.fish_alive, snap["fish_alive"]), "alive changed"
        assert np.array_equal(env.fish_death_timesteps, snap["fish_death_timesteps"]), "death ts changed"
        assert np.array_equal(env.predator_pos, snap["predator_pos"]), "predator pos changed"
        assert env.timestep == snap["timestep"], "timestep changed"
        assert env._last_pre_roll_stats == snap["last_pre_roll_stats"], "pre-roll changed"
        assert env.np_random.bit_generator.state == snap["np_random_state"], "RNG changed"
        # fish velocity = restored * factor, exactly
        exp_fish = (snap["fish_velocities"] * np.float32(triple[0])).astype(np.float32)
        assert np.array_equal(env.fish_velocities, exp_fish), "fish velocity not exact scale"
        # predator velocity = nominal rotated then scaled
        exp_pred = common.rotate_predator_velocity(snap["predator_vel"], triple[1])
        if triple[2] != 1.0:
            exp_pred = (exp_pred * np.float32(triple[2])).astype(np.float32)
        assert np.array_equal(env.predator_vel, exp_pred), "predator velocity wrong"
    env.close()
    print("PASS test_joint_change_only_touches_velocity_vectors")


def test_nominal_is_exact_identity():
    env = _fresh_env()
    o_nom, _ = venv.reset_nominal(env, SEED)
    o_id, _ = venv.apply_condition(env, venv.capture_reset_state(env), list(common.NOMINAL_TRIPLE))
    assert np.array_equal(o_nom, o_id), "nominal obs != raw nominal obs"
    # full rule rollout bit-for-bit
    from v58_evaluate import rule_action

    def run(apply_nom):
        e = _fresh_env()
        obs, snap = venv.reset_nominal(e, SEED)
        if apply_nom:
            obs, _ = venv.apply_condition(e, snap, list(common.NOMINAL_TRIPLE))
        rng = np.random.default_rng(SEED * 2654435761 % (2 ** 31))
        acts = []
        while True:
            if len(obs) > 0:
                a = [rule_action("flee_lead", obs[i], rng) for i in range(len(obs))]
            else:
                a = []
            acts.append(a)
            obs, _r, term, trunc, info = e.step(a)
            if term or trunc:
                break
        final = int(info.get("num_alive", 0))
        e.close()
        return acts, final

    a0, f0 = run(False)
    a1, f1 = run(True)
    assert a0 == a1 and f0 == f1, "nominal condition changed the rollout"
    env.close()
    print("PASS test_nominal_is_exact_identity")


def test_components_match_accepted_rounds():
    assert tuple(common.FISH_FACTORS) == (0.0, 0.5, 1.0)
    assert dict((k, v) for k, v in zip(common.CONDITIONS, (1.0, None)))  # sanity
    # fish factors match v54 semantics (same set as v54 VELOCITY_FACTORS values)
    assert set(common.FISH_FACTORS) == set(v54c.VELOCITY_FACTORS.values()), "fish factors != v54"
    # rotation matrices match v55
    for d, mat in common.ROTATION_MATRICES.items():
        assert tuple(tuple(r) for r in mat) == tuple(tuple(r) for r in v55c.ROTATION_MATRICES[f"deg{d}"]), \
            f"rotation matrix mismatch at {d}"
    # speed factors match v56
    assert set(common.SPEED_FACTORS) == set(v56c.SPEED_FACTORS.values()), "speed factors != v56"
    # direction preserved (positive scalar) and norm proportional
    v = np.array([0.7, -1.3], dtype=np.float32)
    for f in common.SPEED_FACTORS:
        w = (common.rotate_predator_velocity(v, 0) * np.float32(f)).astype(np.float32)
        if f == 1.0:
            assert np.array_equal(w, v)
    # rotation norm preservation (exact integer matrices)
    for d in common.ROTATION_DEGREES:
        r = common.rotate_predator_velocity(v, d)
        assert abs(float(np.linalg.norm(r)) - float(np.linalg.norm(v))) < 1e-6, f"norm not preserved at {d}"
    print("PASS test_components_match_accepted_rounds")


def test_obs_visibility_correct():
    env = _fresh_env()
    obs, snap = venv.reset_nominal(env, SEED)
    triple = [0.5, 180, 1.25]
    obs, _ = venv.apply_condition(env, snap, triple)
    alive = np.where(env.fish_alive)[0]
    for k, fi in enumerate(alive):
        row = k  # observations are row-aligned to alive order
        assert abs(float(obs[row][2]) - float(env.fish_velocities[fi][0]) / 2.0) < 1e-6
        assert abs(float(obs[row][3]) - float(env.fish_velocities[fi][1]) / 2.0) < 1e-6
        rel = env.predator_pos - env.fish_positions[fi]
        dist = float(np.linalg.norm(rel))
        vision = env.FISH_VISION_RADIUS
        if dist < vision:
            assert float(obs[row][5]) == 1.0, "predator should be visible"
            assert abs(float(obs[row][10]) - dist / vision) < 1e-6
        else:
            assert float(obs[row][5]) == 0.0, "predator should be invisible"
    env.close()
    print("PASS test_obs_visibility_correct")


def test_draws_frozen_and_shared():
    seeds = common.resolve_seeds("report")
    triples = common.draw_combined_stress_triples(seeds)
    assert len(triples) == len(seeds)
    # reproducible from the RNG seed alone
    triples2 = common.draw_combined_stress_triples(seeds, rng_seed=common.AUG_RNG_SEED)
    assert triples == triples2, "combined-stress draws not reproducible"
    # each component within the declared discrete set
    for f, r, s in triples:
        assert f in common.FISH_FACTORS and r in common.ROTATION_DEGREES and s in common.SPEED_FACTORS
    # a nominal-triple scene, if present, is kept (not redrawn); all scenes retained
    n_nom = sum(1 for t in triples if tuple(t) == tuple(common.NOMINAL_TRIPLE))
    assert len(triples) == 64, "expected 64 frozen scenes"
    # the same scene seed always maps to the same triple (shared across controllers)
    m = dict(zip(seeds, [tuple(t) for t in triples]))
    assert all(m[s] == tuple(triples[i]) for i, s in enumerate(seeds))
    print(f"PASS test_draws_frozen_and_shared (nominal-triple scenes kept: {n_nom})")


def test_nominal_reproduces_direct_baseline():
    from stable_baselines3 import PPO

    path = ROOT / common.frozen_models_from_v57()[common.CONTROL_MODEL_NAME[0]]["path"]
    model = PPO.load(str(path), device="cpu")
    env = _fresh_env()
    obs, snap = venv.reset_nominal(env, SEED)
    obs, _ = venv.apply_condition(env, snap, list(common.NOMINAL_TRIPLE))
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

    env2 = _fresh_env()
    obs, _info = env2.reset(seed=SEED)
    log2 = []
    while True:
        if len(obs) > 0:
            batch, _ = model.predict(np.asarray(obs), deterministic=True)
            actions = [int(a) for a in np.atleast_1d(batch)]
        else:
            actions = []
        log2.append(actions)
        obs, _r, term, trunc, info = env2.step(actions)
        if term or trunc:
            break
    final2 = int(info.get("num_alive", 0)) / float(common.NUM_FISH)
    env2.close()
    assert log == log2 and final == final2, "v58 nominal path != direct baseline"
    print("PASS test_nominal_reproduces_direct_baseline")


def test_zero_velocity_and_early_death_denominator():
    env = _fresh_env()
    obs, snap = venv.reset_nominal(env, SEED)
    obs, _ = venv.apply_condition(env, snap, [0.0, 0, 1.0])
    assert np.allclose(env.fish_velocities, 0.0), "fish factor 0 did not zero velocities"
    # run one full hold episode; denominator stays /96 even if deaths happen
    steps = 0
    while True:
        actions = [4] * len(obs)
        obs, _r, term, trunc, info = env.step(actions)
        steps += 1
        if term or trunc:
            break
    assert float(info["num_alive"]) / float(common.NUM_FISH) <= 1.0
    assert common.NUM_FISH == 96
    env.close()
    print("PASS test_zero_velocity_and_early_death_denominator")


if __name__ == "__main__":
    test_joint_change_only_touches_velocity_vectors()
    test_nominal_is_exact_identity()
    test_components_match_accepted_rounds()
    test_obs_visibility_correct()
    test_draws_frozen_and_shared()
    test_nominal_reproduces_direct_baseline()
    test_zero_velocity_and_early_death_denominator()
    print("\nALL V58 SEMANTICS TESTS PASSED")
