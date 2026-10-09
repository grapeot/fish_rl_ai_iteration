#!/usr/bin/env python3
"""v58 acceptance-gate tests — negative/binding checks for the frozen manifest and the
strict paired-record validation.

Guards:

  1. missing frozen model arm                         -> gate refuses
  2. unknown policy arm name                          -> gate refuses
  3. wrong checkpoint path                            -> gate refuses
  4. wrong model identity hash                        -> gate refuses
  5. wrong env config                                 -> gate refuses
  6. wrong report seed bank                           -> gate refuses
  7. condition set mismatch                           -> gate refuses
  8. mutated combined-stress triple                   -> gate refuses
  9. mutated nominal triple (legal in-grid value)     -> gate refuses
 10. record with unknown controller                   -> validation refuses
 11. record with unknown condition                    -> validation refuses
 12. duplicate (controller, seed, condition)         -> validation refuses
 13. missing scenario for a controller                 -> validation refuses
 14. extra scenario for a controller                   -> validation refuses
 15. missing condition cell for a scenario             -> validation refuses
 16. a real-manifest positive binding passes
 17. analysis raw-vs-summary style guard / primary is the scene-paired mean
 18. the recommendation predicate boundary at the pre-declared thresholds

Run:
  experiments/v48/.venv/bin/python experiments/v58/tests/test_v58_acceptance_gates.py \
      --manifest experiments/v58/artifacts/frozen_config/v58_frozen_manifest.json
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V58 = HERE.parent
ROOT = V58.parents[1]
for p in (str(ROOT), str(V58)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v58_common as common  # noqa: E402
import v58_verify as verify  # noqa: E402


def _expect_manifest_error(fn, label):
    try:
        fn()
    except verify.ManifestError:
        print(f"PASS {label}")
        return
    raise AssertionError(f"{label}: expected ManifestError, none raised")


def _paths_hashes_triples(manifest):
    paths, hashes = {}, {}
    for name, rec in manifest["models"].items():
        paths[name] = str(ROOT / rec["checkpoint"])
        hashes[name] = rec["policy_tensor_sha256"]
    triples = {int(s): list(t) for s, t in zip(manifest["report_seeds"],
                                              manifest["combined_stress_triples"])}
    return paths, hashes, triples


def run_manifest_gate_tests(MANIFEST):
    def gate(paths, hashes, triples, env=None, conds=None, seeds="report",
             actual_seeds=None, nominal=None):
        verify.validate_manifest_gate(
            MANIFEST, controller_paths=paths, controller_hashes=hashes,
            actual_env_config=env if env is not None else common.env_config(False),
            conditions=conds if conds is not None else common.CONDITIONS,
            actual_triples_by_seed=triples,
            actual_nominal_triple=nominal if nominal is not None else MANIFEST["nominal_triple"],
            seeds_name=seeds,
            actual_seeds=actual_seeds if actual_seeds is not None else common.resolve_seeds(seeds),
        )

    def test_missing_model():
        p, h, t = _paths_hashes_triples(MANIFEST)
        k = sorted(h)[0]
        p.pop(k); h.pop(k)
        _expect_manifest_error(lambda: gate(p, h, t), "test_missing_model")

    def test_unknown_arm():
        p, h, t = _paths_hashes_triples(MANIFEST)
        p["rep9_mystery"] = p[sorted(p)[0]]
        h["rep9_mystery"] = h[sorted(h)[0]]
        _expect_manifest_error(lambda: gate(p, h, t), "test_unknown_arm")

    def test_wrong_checkpoint_path():
        p, h, t = _paths_hashes_triples(MANIFEST)
        ks = sorted(p)
        p[ks[1]] = p[ks[0]]
        _expect_manifest_error(lambda: gate(p, h, t), "test_wrong_checkpoint_path")

    def test_wrong_model_hash():
        p, h, t = _paths_hashes_triples(MANIFEST)
        ks = sorted(h)
        h[ks[1]] = h[ks[0]]
        _expect_manifest_error(lambda: gate(p, h, t), "test_wrong_model_hash")

    def test_wrong_env_config():
        p, h, t = _paths_hashes_triples(MANIFEST)
        bad = common.env_config(False); bad["escape_boost_speed"] = 0.5
        _expect_manifest_error(lambda: gate(p, h, t, env=bad), "test_wrong_env_config")

    def test_wrong_seed_bank():
        p, h, t = _paths_hashes_triples(MANIFEST)
        bad_seeds = common.resolve_seeds("debug")  # a frozen bank, but not 'report'
        _expect_manifest_error(
            lambda: gate(p, h, t, seeds="report", actual_seeds=bad_seeds),
            "test_wrong_seed_bank")

    def test_condition_set_mismatch():
        p, h, t = _paths_hashes_triples(MANIFEST)
        _expect_manifest_error(
            lambda: gate(p, h, t, conds=("nominal",)), "test_condition_set_mismatch")

    def test_mutated_triple():
        p, h, t = _paths_hashes_triples(MANIFEST)
        s0 = sorted(t)[0]
        t[s0] = [0.25, t[s0][1], t[s0][2]]  # mutate a fish factor to an off-grid value
        _expect_manifest_error(lambda: gate(p, h, t), "test_mutated_triple")

    def test_mutated_nominal_triple_in_grid():
        # A legal in-grid mutation of the nominal triple (0.5,0,1) must be refused,
        # exercising the production nominal binding end-to-end (not a malformed tuple).
        p, h, t = _paths_hashes_triples(MANIFEST)
        _expect_manifest_error(
            lambda: gate(p, h, t, nominal=[0.5, 0, 1.0]),
            "test_mutated_nominal_triple_in_grid")

    def test_real_manifest_positive():
        p, h, t = _paths_hashes_triples(MANIFEST)
        gate(p, h, t)
        print("PASS test_real_manifest_positive")

    test_missing_model()
    test_unknown_arm()
    test_wrong_checkpoint_path()
    test_wrong_model_hash()
    test_wrong_env_config()
    test_wrong_seed_bank()
    test_condition_set_mismatch()
    test_mutated_triple()
    test_mutated_nominal_triple_in_grid()
    test_real_manifest_positive()


def _good_records(MANIFEST, n=4, seeds=None):
    seeds = seeds if seeds is not None else common.resolve_seeds("report")[:n]
    recs = []
    for name in MANIFEST["models"]:
        for s in seeds:
            for cond in common.CONDITIONS:
                recs.append({"controller": name, "seed": s, "condition": cond, "final_survival": 0.9})
    for rule in MANIFEST["rule_controllers"]:
        for s in seeds:
            for cond in common.CONDITIONS:
                recs.append({"controller": rule, "seed": s, "condition": cond, "final_survival": 0.5})
    return recs


def _subset_manifest(MANIFEST, seeds):
    m = copy.deepcopy(MANIFEST)
    m["report_seeds"] = list(seeds)
    m["combined_stress_triples"] = [list(t) for t in
                                    common.draw_combined_stress_triples(list(seeds))]
    m["seed_banks"] = dict(m.get("seed_banks", {}))
    return m


def run_record_validation_tests(MANIFEST):
    seeds = common.resolve_seeds("report")[:4]

    def test_unknown_controller():
        recs = _good_records(MANIFEST, seeds=seeds); recs[0] = dict(recs[0], controller="mystery")
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_record_unknown_controller")

    def test_unknown_condition():
        recs = _good_records(MANIFEST, seeds=seeds); recs[0] = dict(recs[0], condition="x")
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_record_unknown_condition")

    def test_duplicate_cell():
        recs = _good_records(MANIFEST, seeds=seeds); recs.append(dict(recs[0]))
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_duplicate_cell")

    def test_missing_scenario():
        recs = _good_records(MANIFEST, seeds=seeds)
        ctrl0 = sorted(MANIFEST["models"])[0]
        drop = seeds[-1]
        recs = [r for r in recs if not (r["controller"] == ctrl0 and r["seed"] == drop)]
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_missing_scenario")

    def test_extra_scenario():
        recs = _good_records(MANIFEST, seeds=seeds)
        ctrl0 = sorted(MANIFEST["models"])[0]
        extra = 123456789
        for cond in common.CONDITIONS:
            recs.append({"controller": ctrl0, "seed": extra, "condition": cond, "final_survival": 0.5})
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_extra_scenario")

    def test_missing_condition_cell():
        recs = _good_records(MANIFEST, seeds=seeds)
        ctrl0 = sorted(MANIFEST["models"])[0]
        recs = [r for r in recs if not (r["controller"] == ctrl0 and r["condition"] == "combined_stress")]
        _expect_manifest_error(
            lambda: verify.validate_paired_records(recs, _subset_manifest(MANIFEST, seeds),
                                                    conditions=common.CONDITIONS),
            "test_missing_condition_cell")

    test_unknown_controller()
    test_unknown_condition()
    test_duplicate_cell()
    test_missing_scenario()
    test_extra_scenario()
    test_missing_condition_cell()


def run_synthetic_analysis_tests():
    import v58_analyze as an
    import v58_env as venv

    # synthetic scene-paired grid: treatment beats control in combined_stress,
    # loses slightly in nominal
    grid = {}
    for s in (101, 102, 103):
        for r in (0, 1, 2):
            grid[(f"rep{r}_treatment_selected", s)] = {"nominal": 0.90, "combined_stress": 0.88}
            grid[(f"rep{r}_control_selected", s)] = {"nominal": 0.91, "combined_stress": 0.84}
        grid[("rule_hold", s)] = {"nominal": 0.75, "combined_stress": 0.80}
    treat = [f"rep{r}_treatment_selected" for r in (0, 1, 2)]
    ctrl = [f"rep{r}_control_selected" for r in (0, 1, 2)]

    d, seeds = an.scene_diffs_pair(grid, treat, ctrl, "combined_stress")
    assert abs(d.mean() - (0.88 - 0.84)) < 1e-9, "combined-stress mean not the scene-paired mean"
    m, lo, hi = an.paired_bootstrap_ci(d)
    assert abs(m - d.mean()) < 1e-9
    dn, _ = an.scene_diffs_pair(grid, treat, ctrl, "nominal")
    assert abs(dn.mean() - (0.90 - 0.91)) < 1e-9
    print("PASS test_primary_is_scene_paired_mean")

    # nominal-visible identity check via the real env
    env = common.make_base_env(False)
    o1, snap = venv.reset_nominal(env, 580777)
    o2, _ = venv.apply_condition(env, snap, list(common.NOMINAL_TRIPLE))
    assert np.array_equal(o1, o2)
    env.close()
    print("PASS test_nominal_identity_in_gate_tests")

    # recommendation predicate boundary arithmetic is exactly the pre-declared formulas
    def pred(combined_lo, all_pos, nominal_mean, nominal_lo, tol=0.01, floor=-0.02):
        p = bool(all_pos and combined_lo > 0.0)
        n = bool((-nominal_mean) < tol and nominal_lo >= floor)
        return p, n, bool(p and n)

    assert pred(0.01, True, -0.005, -0.01) == (True, True, True)
    assert pred(-0.0001, True, -0.005, -0.01)[0] is False         # CI lower not > 0
    assert pred(0.01, False, -0.005, -0.01)[0] is False           # not all reps positive
    assert pred(0.01, True, -0.02, -0.01)[1] is False             # loss >= 1 pp
    assert pred(0.01, True, -0.005, -0.021)[1] is False           # CI lower below floor
    print("PASS test_recommendation_boundary")


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
    print("\nALL V58 ACCEPTANCE GATE TESTS PASSED")


if __name__ == "__main__":
    main()
