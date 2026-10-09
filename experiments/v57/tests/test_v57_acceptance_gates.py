#!/usr/bin/env python3
"""v57 acceptance-gate tests — negative/binding checks for the frozen manifest and
the strict paired-record validation.

Guards:

  1. missing frozen selected arm        -> gate refuses
  2. unknown policy arm name            -> gate refuses
  3. wrong checkpoint path              -> gate refuses
  4. wrong model identity hash          -> gate refuses
  5. wrong env config                   -> gate refuses
  6. wrong report seed bank             -> gate refuses
  7. condition set mismatch             -> gate refuses
  8. record with unknown controller     -> validation refuses
  9. record with unknown condition      -> validation refuses
 10. duplicate (controller, seed, condition) -> validation refuses
 11. missing scenario for a controller         -> validation refuses
 12. missing condition cell for a scenario     -> validation refuses
 13. a real-manifest positive binding passes
 14. analysis raw-vs-summary tamper guard detects a mismatch
 15. the overall balanced CI is the paired scenario mean, not two CIs subtracted

Tests 14-15 use synthetic grids and do not require an on-disk manifest; tests
1-13 need a frozen manifest (skipped with a clear notice if not present).

Run:
  experiments/v48/.venv/bin/python experiments/v57/tests/test_v57_acceptance_gates.py \
      --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V57 = HERE.parent
ROOT = V57.parents[1]
for p in (str(ROOT), str(V57)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v57_common as common  # noqa: E402
import v57_verify as verify  # noqa: E402


def _expect_manifest_error(fn, label):
    try:
        fn()
    except verify.ManifestError:
        print(f"PASS {label}")
        return
    raise AssertionError(f"{label}: expected ManifestError, none raised")


def _paths_and_hashes(manifest):
    paths, hashes = {}, {}
    for run_key, info in manifest["runs"].items():
        sel = info["selected"]
        paths[f"{run_key}_selected"] = str(ROOT / sel["checkpoint"])
        hashes[f"{run_key}_selected"] = sel["policy_tensor_sha256"]
    return paths, hashes


def run_manifest_gate_tests(MANIFEST):
    def gate(paths, hashes, env=None, conds=None, seeds="report", factors=None):
        verify.validate_manifest_gate(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=env if env is not None else common.env_config(False),
            conditions=conds if conds is not None else common.CONDITIONS,
            actual_velocity_factors=factors if factors is not None else common.VELOCITY_FACTORS,
            seeds_name=seeds, actual_seeds=common.resolve_seeds(seeds))

    def test_missing_selected_arm():
        paths, hashes = _paths_and_hashes(MANIFEST)
        k = sorted(hashes)[0]
        paths.pop(k); hashes.pop(k)
        _expect_manifest_error(lambda: gate(paths, hashes), "test_missing_selected_arm")

    def test_unknown_arm():
        paths, hashes = _paths_and_hashes(MANIFEST)
        paths["rep9_mystery_selected"] = paths[sorted(paths)[0]]
        hashes["rep9_mystery_selected"] = hashes[sorted(hashes)[0]]
        _expect_manifest_error(lambda: gate(paths, hashes), "test_unknown_arm")

    def test_wrong_checkpoint_path():
        paths, hashes = _paths_and_hashes(MANIFEST)
        keys = sorted(paths)
        paths[keys[1]] = paths[keys[0]]
        _expect_manifest_error(lambda: gate(paths, hashes), "test_wrong_checkpoint_path")

    def test_wrong_model_hash():
        paths, hashes = _paths_and_hashes(MANIFEST)
        keys = sorted(hashes)
        hashes[keys[1]] = hashes[keys[0]]
        _expect_manifest_error(lambda: gate(paths, hashes), "test_wrong_model_hash")

    def test_wrong_env_config():
        paths, hashes = _paths_and_hashes(MANIFEST)
        bad = common.env_config(False); bad["escape_boost_speed"] = 0.5
        _expect_manifest_error(lambda: gate(paths, hashes, env=bad), "test_wrong_env_config")

    def test_wrong_seed_bank():
        paths, hashes = _paths_and_hashes(MANIFEST)
        # 'smoke' is a frozen bank but NOT the bank this gate call is checked
        # against (we pass a wrong actual seed list for the declared name).
        bad_seeds = common.resolve_seeds("smoke")
        _expect_manifest_error(
            lambda: verify.validate_manifest_gate(
                MANIFEST, controller_paths=paths, controller_hashes=hashes,
                actual_env_config=common.env_config(False), conditions=common.CONDITIONS,
                actual_velocity_factors=common.VELOCITY_FACTORS,
                seeds_name="selection", actual_seeds=bad_seeds),
            "test_wrong_seed_bank")

    def test_wrong_velocity_factor():
        # The P1 gate gap: mutating a condition factor must be refused. The real
        # code/report/manifest all use {1.0,0.5,0.0}; a mutated half=0.25 must fail.
        paths, hashes = _paths_and_hashes(MANIFEST)
        bad = dict(common.VELOCITY_FACTORS); bad["half"] = 0.25
        _expect_manifest_error(
            lambda: gate(paths, hashes, factors=bad), "test_wrong_velocity_factor")

    def test_wrong_velocity_factor_zero():
        paths, hashes = _paths_and_hashes(MANIFEST)
        bad = dict(common.VELOCITY_FACTORS); bad["zero"] = 0.1
        _expect_manifest_error(
            lambda: gate(paths, hashes, factors=bad), "test_wrong_velocity_factor_zero")

    def test_condition_set_mismatch():
        paths, hashes = _paths_and_hashes(MANIFEST)
        _expect_manifest_error(
            lambda: gate(paths, hashes, conds=("nominal", "half")), "test_condition_set_mismatch")

    def test_real_manifest_positive():
        paths, hashes = _paths_and_hashes(MANIFEST)
        gate(paths, hashes)
        print("PASS test_real_manifest_positive")

    test_missing_selected_arm()
    test_unknown_arm()
    test_wrong_checkpoint_path()
    test_wrong_model_hash()
    test_wrong_env_config()
    test_wrong_seed_bank()
    test_wrong_velocity_factor()
    test_wrong_velocity_factor_zero()
    test_condition_set_mismatch()
    test_real_manifest_positive()


def _good_records(MANIFEST, n=4):
    seeds = common.resolve_seeds("report")[:n]
    recs = []
    for run_key in MANIFEST["runs"]:
        for s in seeds:
            for cond in common.CONDITIONS:
                recs.append({"controller": f"{run_key}_selected", "seed": s,
                             "condition": cond, "final_survival": 0.9})
    for rule in MANIFEST["rule_controllers"]:
        for s in seeds:
            for cond in common.CONDITIONS:
                recs.append({"controller": rule, "seed": s, "condition": cond,
                             "final_survival": 0.5})
    return recs


def _subset_manifest(MANIFEST, seeds):
    m = copy.deepcopy(MANIFEST)
    m["report_seeds"] = list(seeds)
    m["seed_banks"] = dict(m.get("seed_banks", {}))
    m["seed_banks"]["report"] = {"seeds": list(seeds)}
    return m


def run_record_validation_tests(MANIFEST):
    seeds = common.resolve_seeds("report")[:4]

    def test_unknown_controller():
        recs = _good_records(MANIFEST); recs[0] = dict(recs[0], controller="mystery")
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_record_unknown_controller")

    def test_unknown_condition():
        recs = _good_records(MANIFEST); recs[0] = dict(recs[0], condition="quadruple")
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_record_unknown_condition")

    def test_duplicate_cell():
        recs = _good_records(MANIFEST); recs.append(dict(recs[0]))
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_duplicate_cell")

    def test_missing_scenario():
        recs = _good_records(MANIFEST)
        ctrl0 = f"{sorted(MANIFEST['runs'])[0]}_selected"
        drop = seeds[-1]
        recs = [r for r in recs if not (r["controller"] == ctrl0 and r["seed"] == drop)]
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_missing_scenario")

    def test_missing_condition_cell():
        recs = _good_records(MANIFEST)
        ctrl0 = f"{sorted(MANIFEST['runs'])[0]}_selected"
        recs = [r for r in recs if not (r["controller"] == ctrl0 and r["condition"] == "zero")]
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_missing_condition_cell")

    test_unknown_controller()
    test_unknown_condition()
    test_duplicate_cell()
    test_missing_scenario()
    test_missing_condition_cell()


def run_synthetic_analysis_tests():
    import v57_analyze as an

    # synthetic paired grid for 3 scenarios
    grid = {}
    for s in (101, 102, 103):
        grid[("rep0_treatment_selected", s)] = {"nominal": 0.90, "half": 0.85, "zero": 0.80}
        grid[("rep1_treatment_selected", s)] = {"nominal": 0.88, "half": 0.84, "zero": 0.79}
        grid[("rep2_treatment_selected", s)] = {"nominal": 0.86, "half": 0.83, "zero": 0.78}
        grid[("rep0_control_selected", s)] = {"nominal": 0.93, "half": 0.92, "zero": 0.88}
        grid[("rep1_control_selected", s)] = {"nominal": 0.93, "half": 0.93, "zero": 0.92}
        grid[("rep2_control_selected", s)] = {"nominal": 0.92, "half": 0.90, "zero": 0.87}

    treat = [f"rep{r}_treatment_selected" for r in (0, 1, 2)]
    ctrl = [f"rep{r}_control_selected" for r in (0, 1, 2)]
    _scen, diffs = an.condition_diffs_by_scenario(grid, treat, ctrl, ("nominal", "half", "zero"))
    # nominal diff = mean(0.90,0.88,0.86) - mean(0.93,0.93,0.92) = 0.88 - 0.92667
    assert abs(diffs["nominal"].mean() - (0.88 - 0.9266666667)) < 1e-6
    blk = an.block_from_scenario_diffs(diffs, ("nominal", "half", "zero"), n_boot=200)
    # overall must equal the per-scenario equal-weight mean of the three diffs
    expect_overall = float(np.mean([diffs[c].mean() for c in ("nominal", "half", "zero")]))
    assert abs(blk["overall_equal_weight"]["mean_diff"] - expect_overall) < 1e-9
    # worst per scenario is strictly <= overall per scenario
    assert blk["worst_condition"]["mean_diff"] <= blk["overall_equal_weight"]["mean_diff"] + 1e-12
    print("PASS test_overall_is_paired_scenario_mean")

    # raw-vs-summary tamper guard
    summary = {"controllers": {"rep0_treatment_selected": {
        "nominal": {"mean_final_survival": 0.90},
        "half": {"mean_final_survival": 0.85},
        "zero": {"mean_final_survival": 0.80}}}}
    good = an.check_summary_matches_raw(summary, grid, None, ("nominal", "half", "zero"))
    assert good is None, "guard rejected a truthful summary"
    bad_summary = {"controllers": {"rep0_treatment_selected": {
        "nominal": {"mean_final_survival": 0.123}}}}
    assert an.check_summary_matches_raw(bad_summary, grid, None, ("nominal", "half", "zero")) is not None
    print("PASS test_raw_summary_guard_detects_tamper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    run_synthetic_analysis_tests()
    if args.manifest and Path(args.manifest).exists():
        MANIFEST = verify.load_manifest(args.manifest)
        run_manifest_gate_tests(MANIFEST)
        run_record_validation_tests(MANIFEST)
    else:
        print("[skip] manifest-binding/record tests: no frozen manifest available yet")
    print("\nALL V57 ACCEPTANCE GATE TESTS PASSED")


if __name__ == "__main__":
    main()
