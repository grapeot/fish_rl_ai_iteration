#!/usr/bin/env python3
"""v56 acceptance-gate tests — negative/binding checks for the frozen controller
manifest, the speed-factor map and the strict paired-record validation.

These are the "could the round pass without a real binding?" guards:

   1. missing frozen controller arm  -> gate refuses
   2. unknown controller name        -> gate refuses
   3. wrong checkpoint path          -> gate refuses
   4. wrong model identity hash      -> gate refuses
   5. wrong env config               -> gate refuses
   6. wrong heading bias             -> gate refuses
   7. wrong seed bank                -> gate refuses
   8. condition (speed scale) set mismatch -> gate refuses
   9. wrong speed-factor map         -> gate refuses
  10. record with unknown controller -> validation refuses
  11. record with unknown condition  -> validation refuses
  12. duplicate (controller, seed, condition) -> validation refuses
  13. missing scenario for a controller         -> validation refuses
  14. missing (dropped) scale cell for a scenario -> validation refuses
  15. a real-manifest positive binding passes

Run:
  experiments/v48/.venv/bin/python experiments/v56/tests/test_v56_acceptance_gates.py
"""

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V56 = HERE.parent
ROOT = V56.parents[1]
for p in (str(ROOT), str(V56)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v56_common as common  # noqa: E402
import v56_verify as verify  # noqa: E402

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
            seeds_name="report", actual_seeds=common.resolve_seeds("report"),
            speed_factors=common.SPEED_FACTORS),
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
            actual_env_config=common.env_config(False),
            conditions=("scale0p75", "scale1p00"),
            seeds_name="report", actual_seeds=common.resolve_seeds("report"),
            speed_factors=common.SPEED_FACTORS),
        "test_condition_set_mismatch")


def test_wrong_speed_factor_map():
    paths, hashes = _controller_paths_and_hashes()
    bad_factors = dict(common.SPEED_FACTORS)
    bad_factors["scale1p25"] = 1.5
    _expect_manifest_error(
        lambda: verify.validate_controller_binding(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
            seeds_name="report", actual_seeds=common.resolve_seeds("report"),
            speed_factors=bad_factors),
        "test_wrong_speed_factor_map")


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
    recs[0] = dict(recs[0], condition="scale9p99")
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


def test_missing_scale_cell():
    recs = _good_records()
    recs = [r for r in recs if not (r["controller"] == "rep0_ppo" and r["condition"] == "scale1p25")]
    seeds = common.resolve_seeds("report")[:4]
    _expect_manifest_error(
        lambda: verify.validate_paired_records(recs, _subset_manifest(seeds), conditions=common.CONDITIONS),
        "test_missing_scale_cell")
    # supplementary cheap check: the raw-derived HOLD baseline contrast must refuse an
    # incomplete grid (missing HOLD cell or missing PPO arm), no evaluation involved.
    import v56_analyze as an

    def _synthetic(missing_hold_seed=None, drop_arm=None):
        g = {}
        for s in (101, 102, 103):
            for arm, base in (("rep0_ppo", 0.9), ("rep1_ppo", 0.8), ("rep2_ppo", 0.7)):
                if arm == drop_arm:
                    continue
                g[(arm, s)] = {"scale0p75": base - 0.2, "scale1p00": base, "scale1p25": base + 0.1}
            if s != missing_hold_seed:
                g[("rule_hold", s)] = {"scale0p75": 0.5, "scale1p00": 0.5, "scale1p25": 0.5}
        return g

    for label, kwargs in (("missing_hold_cell", {"missing_hold_seed": 101}),
                          ("missing_ppo_arm", {"drop_arm": "rep1_ppo"})):
        try:
            an.relative_hold_contrasts(_synthetic(**kwargs), n_boot=50)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"relative_hold_contrasts accepted an incomplete grid: {label}")
    print("PASS test_missing_scale_cell (+ relative-hold grid guards)")


def test_real_manifest_positive():
    paths, hashes = _controller_paths_and_hashes()
    verify.validate_controller_binding(
        MANIFEST, controller_paths=paths, controller_hashes=hashes,
        actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
        seeds_name="report", actual_seeds=common.resolve_seeds("report"),
        speed_factors=common.SPEED_FACTORS)
    verify.validate_controller_binding(
        MANIFEST, controller_paths=paths, controller_hashes=hashes,
        actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
        seeds_name="debug", actual_seeds=common.resolve_seeds("debug"))
    print("PASS test_real_manifest_positive")


if __name__ == "__main__":
    test_missing_controller_arm()
    test_unknown_controller()
    test_wrong_checkpoint_path()
    test_wrong_model_hash()
    test_wrong_env_config()
    test_wrong_heading_bias()
    test_wrong_seed_bank()
    test_condition_set_mismatch()
    test_wrong_speed_factor_map()
    test_record_unknown_controller()
    test_record_unknown_condition()
    test_duplicate_cell()
    test_missing_scenario()
    test_missing_scale_cell()
    test_real_manifest_positive()
    print("\nALL V56 ACCEPTANCE GATE TESTS PASSED")
