#!/usr/bin/env python3
"""v54 acceptance-gate tests — negative/binding checks for the frozen controller
manifest and the strict paired-record validation.

These are the "could the round pass without a real binding?" guards:

  1. missing frozen controller arm  -> gate refuses
  2. unknown controller name        -> gate refuses
  3. wrong checkpoint path          -> gate refuses
  4. wrong model identity hash      -> gate refuses
  5. wrong env config               -> gate refuses
  6. wrong seed bank                -> gate refuses
  7. condition set mismatch         -> gate refuses
  8. record with unknown controller -> validation refuses
  9. record with unknown condition  -> validation refuses
 10. duplicate (controller, seed, condition) -> validation refuses
 11. missing scenario for a controller         -> validation refuses
 12. missing condition cell for a scenario     -> validation refuses
 13. a real-manifest positive binding passes

Run:
  experiments/v48/.venv/bin/python experiments/v54/tests/test_v54_acceptance_gates.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V54 = HERE.parent
ROOT = V54.parents[1]
for p in (str(ROOT), str(V54)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v54_common as common  # noqa: E402
import v54_verify as verify  # noqa: E402

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
            actual_env_config=common.env_config(False), conditions=("nominal", "half"),
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
    recs[0] = dict(recs[0], condition="quadruple")
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
    # drop all rows for one controller at the last scenario
    seeds = common.resolve_seeds("report")[:4]
    drop_seed = seeds[-1]
    recs = [r for r in recs if not (r["controller"] == "rep0_ppo" and r["seed"] == drop_seed)]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_missing_scenario")


def test_missing_condition_cell():
    recs = _good_records()
    recs = [r for r in recs if not (r["controller"] == "rep0_ppo" and r["condition"] == "zero")]
    seeds = common.resolve_seeds("report")[:4]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_missing_condition_cell")


def test_wrong_velocity_factor_map():
    paths, hashes = _controller_paths_and_hashes()
    bad = dict(common.VELOCITY_FACTORS)
    bad["zero"] = 0.1
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report"),
            velocity_factors=bad),
        "test_wrong_velocity_factor_map")


def test_real_manifest_positive():
    paths, hashes = _controller_paths_and_hashes()
    # the frozen manifest itself must pass a full binding
    verify.validate_controller_binding(
        MANIFEST, controller_paths=paths, controller_hashes=hashes,
        actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
        seeds_name="report", actual_seeds=common.resolve_seeds("report"),
        velocity_factors=common.VELOCITY_FACTORS)
    verify.validate_controller_binding(
        MANIFEST, controller_paths=paths, controller_hashes=hashes,
        actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
        seeds_name="debug", actual_seeds=common.resolve_seeds("debug"),
        velocity_factors=common.VELOCITY_FACTORS)
    print("PASS test_real_manifest_positive")


def _synthetic_grid(scenarios=(101, 102), drop_hold=False, drop_ppo_seed=None):
    grid = {}
    for s in scenarios:
        for i, name in enumerate(common.PPO_CONTROLLER_NAMES):
            if drop_ppo_seed is not None and name == common.PPO_CONTROLLER_NAMES[0] and s == drop_ppo_seed:
                continue
            grid[(name, s)] = {"nominal": 0.9 - 0.1 * i, "zero": 0.8 - 0.1 * i}
        if not drop_hold:
            grid[("rule_hold", s)] = {"nominal": 0.5, "zero": 0.5}
    return grid


def test_summary_raw_guard_detects_tamper():
    """The cheap consistency guard must reject a summary mean that disagrees with
    the raw recomputation."""
    import v54_analyze as an

    grid = _synthetic_grid()
    good_summary = {"controllers": {
        "rep0_ppo": {"nominal": {"mean_final_survival": float(np.mean([0.9, 0.9]))}}}}
    assert an.check_summary_matches_raw(good_summary, grid) is None
    bad_summary = {"controllers": {
        "rep0_ppo": {"nominal": {"mean_final_survival": 0.123}}}}
    assert an.check_summary_matches_raw(bad_summary, grid) is not None
    print("PASS test_summary_raw_guard_detects_tamper")


def test_relative_hold_rejects_incomplete_grid():
    """The raw-derived relative-HOLD computation must reject a grid missing HOLD
    or missing a PPO policy at a scenario."""
    import v54_analyze as an

    an.relative_hold_contrasts(_synthetic_grid(), n_boot=100)
    for kwargs in ({"drop_hold": True}, {"drop_ppo_seed": 101}):
        try:
            an.relative_hold_contrasts(_synthetic_grid(**kwargs), n_boot=100)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"relative-HOLD accepted incomplete grid: {kwargs}")
    # DiD must be the difference of the two paired advantages, not two CIs subtracted
    rel = an.relative_hold_contrasts(_synthetic_grid(), n_boot=100)
    did = rel["difference_in_differences"]["mean"]
    lhs = rel["zero_ppo_minus_hold"]["mean"] - rel["nominal_ppo_minus_hold"]["mean"]
    assert abs(did - lhs) < 1e-12, "DiD is not the paired advantage difference"
    print("PASS test_relative_hold_rejects_incomplete_grid")


if __name__ == "__main__":
    test_missing_controller_arm()
    test_unknown_controller()
    test_wrong_checkpoint_path()
    test_wrong_model_hash()
    test_wrong_env_config()
    test_wrong_seed_bank()
    test_condition_set_mismatch()
    test_wrong_velocity_factor_map()
    test_record_unknown_controller()
    test_record_unknown_condition()
    test_duplicate_cell()
    test_missing_scenario()
    test_missing_condition_cell()
    test_real_manifest_positive()
    test_summary_raw_guard_detects_tamper()
    test_relative_hold_rejects_incomplete_grid()
    print("\nALL V54 ACCEPTANCE GATE TESTS PASSED")
