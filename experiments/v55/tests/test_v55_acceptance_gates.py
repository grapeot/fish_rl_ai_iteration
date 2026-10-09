#!/usr/bin/env python3
"""v55 acceptance-gate tests — negative/binding checks for the frozen controller
manifest and the strict paired-record validation.

These are the "could the round pass without a real binding?" guards:

  1. missing frozen controller arm  -> gate refuses
  2. unknown controller name        -> gate refuses
  3. wrong checkpoint path          -> gate refuses
  4. wrong model identity hash      -> gate refuses
  5. wrong env config               -> gate refuses
  6. wrong seed bank                -> gate refuses
  7. condition (rotation angle) set mismatch -> gate refuses
  8. record with unknown controller -> validation refuses
  9. record with unknown condition  -> validation refuses
 10. duplicate (controller, seed, condition) -> validation refuses
 11. missing scenario for a controller         -> validation refuses
 12. missing (dropped) angle cell for a scenario -> validation refuses
 13. a real-manifest positive binding passes

Run:
  experiments/v48/.venv/bin/python experiments/v55/tests/test_v55_acceptance_gates.py
"""

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V55 = HERE.parent
ROOT = V55.parents[1]
for p in (str(ROOT), str(V55)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v55_common as common  # noqa: E402
import v55_verify as verify  # noqa: E402

MANIFEST = verify.load_manifest(ROOT / common.FROZEN_MANIFEST_RELPATH)


def _expect_manifest_error(fn, label):
    try:
        fn()
    except verify.ManifestError:
        print(f"PASS {label}")
        return
    raise AssertionError(f"{label}: expected ManifestError, none raised")


def _controller_paths_and_hashes():
    paths = {name: str(ROOT / info["checkpoint"]) for name, info in MANIFEST["controllers"].items()}
    hashes = {name: info["policy_tensor_sha256"] for name, info in MANIFEST["controllers"].items()}
    return paths, hashes


def test_missing_controller_arm():
    paths, hashes = _controller_paths_and_hashes()
    paths.pop("rep1_ppo")
    hashes.pop("rep1_ppo")
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_missing_controller_arm")


def test_unknown_controller():
    paths, hashes = _controller_paths_and_hashes()
    paths["rep9_ppo"] = paths["rep0_ppo"]
    hashes["rep9_ppo"] = hashes["rep0_ppo"]
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_unknown_controller")


def test_wrong_checkpoint_path():
    paths, hashes = _controller_paths_and_hashes()
    paths["rep1_ppo"] = paths["rep0_ppo"]
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_wrong_checkpoint_path")


def test_wrong_model_hash():
    paths, hashes = _controller_paths_and_hashes()
    hashes["rep2_ppo"] = hashes["rep0_ppo"]
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_wrong_model_hash")


def test_wrong_env_config():
    paths, hashes = _controller_paths_and_hashes()
    bad = common.env_config(False)
    bad["escape_boost_speed"] = 0.5
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=bad, conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_wrong_env_config")


def test_wrong_heading_bias():
    paths, hashes = _controller_paths_and_hashes()
    bad = common.env_config(False)
    bad["predator_heading_bias"] = []
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=bad, conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_wrong_heading_bias")


def test_wrong_seed_bank():
    paths, hashes = _controller_paths_and_hashes()
    bad = common.resolve_seeds("debug")
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=bad),
        "test_wrong_seed_bank")


def test_condition_set_mismatch():
    paths, hashes = _controller_paths_and_hashes()
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=("deg0", "deg90"),
            seeds_name="report", actual_seeds=common.resolve_seeds("report")),
        "test_condition_set_mismatch")


def _good_records():
    seeds = common.resolve_seeds("report")[:4]  # subset is fine for record-structure tests
    recs = []
    for ctrl in list(MANIFEST["controllers"]) + list(MANIFEST["rule_controllers"]):
        for s in seeds:
            for cond in common.CONDITIONS:
                recs.append({"controller": ctrl, "seed": s, "condition": cond,
                             "final_survival": 0.9})
    return recs


def _subset_manifest(seeds):
    m = copy.deepcopy(MANIFEST)
    m["seed_banks"]["report"]["seeds"] = list(seeds)
    return m


def test_record_unknown_controller():
    recs = _good_records()
    recs[0] = dict(recs[0], controller="mystery_ctrl")
    seeds = common.resolve_seeds("report")[:4]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_record_unknown_controller")


def test_record_unknown_condition():
    recs = _good_records()
    recs[0] = dict(recs[0], condition="deg123")
    seeds = common.resolve_seeds("report")[:4]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_record_unknown_condition")


def test_duplicate_cell():
    recs = _good_records()
    recs.append(dict(recs[0]))
    seeds = common.resolve_seeds("report")[:4]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_duplicate_cell")


def test_missing_scenario():
    recs = _good_records()
    seeds = common.resolve_seeds("report")[:4]
    drop_seed = seeds[-1]
    recs = [r for r in recs if not (r["controller"] == "rep0_ppo" and r["seed"] == drop_seed)]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_missing_scenario")


def test_missing_angle_cell():
    recs = _good_records()
    recs = [r for r in recs if not (r["controller"] == "rep0_ppo" and r["condition"] == "deg270")]
    seeds = common.resolve_seeds("report")[:4]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_missing_angle_cell")


def test_real_manifest_positive():
    paths, hashes = _controller_paths_and_hashes()
    verify.validate_controller_binding(
        MANIFEST, controller_paths=paths, controller_hashes=hashes,
        actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
        seeds_name="report", actual_seeds=common.resolve_seeds("report"))
    verify.validate_controller_binding(
        MANIFEST, controller_paths=paths, controller_hashes=hashes,
        actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
        seeds_name="debug", actual_seeds=common.resolve_seeds("debug"))
    print("PASS test_real_manifest_positive")


# --- cheap synthetic checks for the post-hoc relative-to-HOLD aggregation / DiD ---

def _synthetic_grid():
    """A tiny hand-checkable grid: 2 seeds x 4 conditions, values chosen so the
    PPO-HOLD average and the DiD are exactly known by hand."""
    ppo = list(common.PPO_CONTROLLER_NAMES)
    ppo_vals = {
        # deg0: ppo mean 0.4 -> G = 0.4 - 0.2 = 0.2
        ("deg0", 1): {ppo[0]: 0.3, ppo[1]: 0.4, ppo[2]: 0.5, "rule_hold": 0.2},
        ("deg0", 2): {ppo[0]: 0.6, ppo[1]: 0.5, ppo[2]: 0.4, "rule_hold": 0.3},
        # deg180: ppo mean 0.2 -> G = 0.2 - 0.2 = 0.0  -> DiD = -0.2
        ("deg180", 1): {ppo[0]: 0.1, ppo[1]: 0.2, ppo[2]: 0.3, "rule_hold": 0.2},
        ("deg180", 2): {ppo[0]: 0.3, ppo[1]: 0.2, ppo[2]: 0.1, "rule_hold": 0.2},
        # deg90 / deg270 present so the grid is complete
        ("deg90", 1): {ppo[0]: 0.3, ppo[1]: 0.4, ppo[2]: 0.5, "rule_hold": 0.2},
        ("deg90", 2): {ppo[0]: 0.6, ppo[1]: 0.5, ppo[2]: 0.4, "rule_hold": 0.3},
        ("deg270", 1): {ppo[0]: 0.3, ppo[1]: 0.4, ppo[2]: 0.5, "rule_hold": 0.2},
        ("deg270", 2): {ppo[0]: 0.6, ppo[1]: 0.5, ppo[2]: 0.4, "rule_hold": 0.3},
    }
    grid = {}
    for (cond, seed), vals in ppo_vals.items():
        for ctrl, v in vals.items():
            grid.setdefault((ctrl, seed), {})[cond] = v
    return grid


def test_synthetic_relative_aggregation():
    """Hand-checked: ppo_minus_hold averages the 3 PPO first, then subtracts HOLD."""
    import v55_analyze as an

    grid = _synthetic_grid()
    vals, seeds = an.ppo_minus_hold(grid, "deg0", common.PPO_CONTROLLER_NAMES)
    # seed1: mean(0.3,0.4,0.5)=0.4 - 0.2 = 0.2 ; seed2: mean(0.6,0.5,0.4)=0.5 - 0.3 = 0.2
    assert seeds == [1, 2], seeds
    assert np.allclose(vals, [0.2, 0.2]), vals
    deg180, _ = an.ppo_minus_hold(grid, "deg180", common.PPO_CONTROLLER_NAMES)
    assert np.allclose(deg180, [0.0, 0.0]), deg180
    did, _ = an.did_advantage(grid, "deg180", "deg0", common.PPO_CONTROLLER_NAMES)
    assert np.allclose(did, [-0.2, -0.2]), did
    print("PASS test_synthetic_relative_aggregation")


def test_relative_analysis_rejects_missing_grid():
    """ppo_minus_hold / did_advantage must refuse an incomplete grid, not silently drop."""
    import v55_analyze as an

    grid = _synthetic_grid()
    # drop one whole PPO cell (a scenario) so the grid is incomplete
    del grid[(common.PPO_CONTROLLER_NAMES[2], 2)]
    try:
        an.ppo_minus_hold(grid, "deg180", common.PPO_CONTROLLER_NAMES)
    except SystemExit:
        print("PASS test_relative_analysis_rejects_missing_grid")
        return
    raise AssertionError("missing-grid: expected SystemExit, none raised")


if __name__ == "__main__":
    test_missing_controller_arm()
    test_unknown_controller()
    test_wrong_checkpoint_path()
    test_wrong_model_hash()
    test_wrong_env_config()
    test_wrong_heading_bias()
    test_wrong_seed_bank()
    test_condition_set_mismatch()
    test_record_unknown_controller()
    test_record_unknown_condition()
    test_duplicate_cell()
    test_missing_scenario()
    test_missing_angle_cell()
    test_real_manifest_positive()
    test_synthetic_relative_aggregation()
    test_relative_analysis_rejects_missing_grid()
    print("\nALL V55 ACCEPTANCE GATE TESTS PASSED")
