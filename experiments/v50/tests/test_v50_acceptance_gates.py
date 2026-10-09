#!/usr/bin/env python3
"""v50 acceptance-gap tests: the manifest gate and the analyzer must reject
tampering, not merely check that a file exists.

Run:
  experiments/v48/.venv/bin/python experiments/v50/tests/test_v50_acceptance_gates.py
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V50 = HERE.parent
ROOT = V50.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V50))

import v50_common as common  # noqa: E402
import v50_verify as verify  # noqa: E402


def _fake_manifest(seeds_sel=(1, 2, 3), seeds_rep=(11, 12, 13, 14)):
    runs = [f"rep{r}_{arm}" for r in (0, 1, 2) for arm in ("original", "survival_only")]
    sel = {}
    for rk in runs:
        sel[rk] = {
            "selected_arm": f"{rk}_u50",
            "selected_update": 50,
            "checkpoint": f"fixture/{rk}/checkpoints/model_updates_50.zip",
            "selection_mean_final_survival": 0.9,
            "selection_n": len(seeds_sel),
            "policy_tensor_sha256": f"hash_sel_{rk}",
            "final_checkpoint": f"fixture/{rk}/checkpoints/model_final.zip",
            "final_policy_tensor_sha256": f"hash_fin_{rk}",
            "selected_is_final": False,
            "candidates": [],
        }
    return {
        "kind": verify.MANIFEST_KIND,
        "version": verify.MANIFEST_VERSION,
        "hash_scheme": verify.HASH_SCHEME,
        "created_utc": "2026-10-09T00:00:00+00:00",
        "selection_seed_bank": {"rng_seed": 500201, "n": len(seeds_sel)},
        "selection_seeds": list(seeds_sel),
        "report_seed_bank": {"rng_seed": 500202, "n": len(seeds_rep)},
        "report_seeds": list(seeds_rep),
        "env": {"neighbor": False, "env_config": {"num_fish": 96}},
        "selected": sel,
    }


def _arm_specs_and_hashes(manifest):
    specs, hashes = [], {}
    for rk, info in manifest["selected"].items():
        specs.append({"name": f"{rk}_selected", "kind": "policy", "path": info["checkpoint"]})
        hashes[f"{rk}_selected"] = info["policy_tensor_sha256"]
        if not info["selected_is_final"]:
            specs.append({"name": f"{rk}_final", "kind": "policy", "path": info["final_checkpoint"]})
            hashes[f"{rk}_final"] = info["final_policy_tensor_sha256"]
    return specs, hashes


def test_gate_accepts_correct_binding():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    verify.validate_manifest_gate(
        m, seeds_name="report", actual_seeds=m["report_seeds"], neighbor=False,
        actual_env_config=m["env"]["env_config"], arm_specs=specs, arm_hashes=hashes)
    print("PASS test_gate_accepts_correct_binding")


def test_gate_rejects_swapped_model():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    hashes["rep0_original_selected"] = "TAMPERED"  # wrong model identity
    _expect_fail(m, specs, hashes, "model identity hash differs")
    print("PASS test_gate_rejects_swapped_model")


def test_gate_rejects_different_checkpoint_path():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    for s in specs:
        if s["name"] == "rep1_survival_only_selected":
            s["path"] = "fixture/other/checkpoints/model_updates_100.zip"
    _expect_fail(m, specs, hashes, "not the frozen selection")
    print("PASS test_gate_rejects_different_checkpoint_path")


def test_gate_rejects_wrong_seed_bank():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    _expect_fail(m, specs, hashes, "report seed bank",
                 actual_seeds=[999, 998, 997, 996])
    print("PASS test_gate_rejects_wrong_seed_bank")


def test_gate_rejects_env_mismatch():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    _expect_fail(m, specs, hashes, "env evaluation config differs",
                 actual_env_config={"num_fish": 250})
    print("PASS test_gate_rejects_env_mismatch")


def test_gate_rejects_missing_final_arm():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    specs = [s for s in specs if s["name"] != "rep2_original_final"]
    hashes.pop("rep2_original_final", None)
    _expect_fail(m, specs, hashes, "missing the final arm")
    print("PASS test_gate_rejects_missing_final_arm")


def _expect_fail(m, specs, hashes, needle, **overrides):
    kwargs = dict(seeds_name="report", actual_seeds=m["report_seeds"], neighbor=False,
                  actual_env_config=m["env"]["env_config"], arm_specs=specs, arm_hashes=hashes)
    kwargs.update(overrides)
    try:
        verify.validate_manifest_gate(m, **kwargs)
    except verify.ManifestError as exc:
        assert needle in str(exc), f"wrong failure: {exc}"
        return
    raise AssertionError(f"gate accepted when it should have failed ({needle})")


def _records_from_manifest(m, drop=None):
    recs = []
    for rk in m["selected"]:
        for stage in ("selected", "final"):
            if m["selected"][rk]["selected_is_final"] and stage == "final":
                continue
            name = f"{rk}_{stage}"
            for s in m["report_seeds"]:
                if drop and (name, s) == drop:
                    continue
                recs.append({"control": name, "seed": s, "final_survival": 0.9})
    return recs


def test_analyzer_accepts_complete_records():
    m = _fake_manifest()
    recs = _records_from_manifest(m)
    resolved = verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    assert f"rep0_original_selected" in resolved
    assert f"rep0_original_final" in resolved
    print("PASS test_analyzer_accepts_complete_records")


def test_analyzer_rejects_missing_episode():
    m = _fake_manifest()
    recs = _records_from_manifest(m, drop=("rep1_original_selected", m["report_seeds"][0]))
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "seed set mismatch" in str(exc)
        print("PASS test_analyzer_rejects_missing_episode")
        return
    raise AssertionError("analyzer accepted a missing episode")


def test_analyzer_rejects_duplicate_episode():
    m = _fake_manifest()
    recs = _records_from_manifest(m)
    recs.append(dict(recs[0]))
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "duplicate episode" in str(exc)
        print("PASS test_analyzer_rejects_duplicate_episode")
        return
    raise AssertionError("analyzer accepted a duplicate episode")


def test_analyzer_rejects_extra_episode():
    m = _fake_manifest()
    recs = _records_from_manifest(m)
    recs.append({"control": "rep0_original_selected", "seed": 123456789, "final_survival": 0.9})
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "seed set mismatch" in str(exc)
        print("PASS test_analyzer_rejects_extra_episode")
        return
    raise AssertionError("analyzer accepted an extra episode")


def test_analyzer_rejects_missing_final_without_flag():
    m = _fake_manifest()  # selected_is_final is False for all runs
    recs = [r for r in _records_from_manifest(m) if not r["control"].endswith("_final")]
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "does not mark selected_is_final" in str(exc)
        print("PASS test_analyzer_rejects_missing_final_without_flag")
        return
    raise AssertionError("analyzer silently backfilled final when the manifest forbade it")


def test_analyzer_reuses_selected_when_manifest_says_final():
    m = _fake_manifest()
    m["selected"]["rep0_original"]["selected_is_final"] = True
    # a legitimate reuse claim carries identical identity hashes
    m["selected"]["rep0_original"]["final_policy_tensor_sha256"] = (
        m["selected"]["rep0_original"]["policy_tensor_sha256"]
    )
    recs = _records_from_manifest(m)  # produces no _final for rep0_original
    resolved = verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    assert "rep0_original_final" in resolved
    assert resolved["rep0_original_final"] == resolved["rep0_original_selected"]
    print("PASS test_analyzer_reuses_selected_when_manifest_says_final")


def test_analyzer_rejects_flag_true_with_mismatched_hash():
    """A manifest that flags selected_is_final but whose selected/final hashes
    differ must not let the analyzer silently reuse the selected records."""
    m = _fake_manifest()
    m["selected"]["rep0_original"]["selected_is_final"] = True
    m["selected"]["rep0_original"]["final_policy_tensor_sha256"] = "different_hash"
    recs = _records_from_manifest(m)  # no _final for rep0_original
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "policy hashes differ" in str(exc)
        print("PASS test_analyzer_rejects_flag_true_with_mismatched_hash")
        return
    raise AssertionError("analyzer reused selected for final despite mismatched hashes")


def test_gate_rejects_flag_true_with_mismatched_hash():
    m = _fake_manifest()
    m["selected"]["rep1_original"]["selected_is_final"] = True
    m["selected"]["rep1_original"]["final_policy_tensor_sha256"] = "different_hash"
    specs, hashes = _arm_specs_and_hashes(m)  # no _final for rep1_original
    _expect_fail(m, specs, hashes, "selected/final policy hashes differ")
    print("PASS test_gate_rejects_flag_true_with_mismatched_hash")


def test_gate_rejects_unknown_selected_arm():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    specs.append({"name": "rep9_ghost_selected", "kind": "policy", "path": "fixture/x.zip"})
    hashes["rep9_ghost_selected"] = "x"
    _expect_fail(m, specs, hashes, "unknown run")
    print("PASS test_gate_rejects_unknown_selected_arm")


def test_gate_binds_real_manifest_if_present():
    """If the corrected manifest exists on disk, load it and check the real
    selected/final hashes it recorded are internally consistent."""
    real = V50 / "artifacts" / "corrected_streams" / "results" / "selection_manifest.json"
    if not real.exists():
        print("SKIP test_gate_binds_real_manifest_if_present (no corrected manifest yet)")
        return
    m = verify.load_manifest(real)
    for rk, info in m["selected"].items():
        assert info["policy_tensor_sha256"] == info["policy_state_hash"]
        assert Path(info["checkpoint"]).exists()
        assert Path(info["final_checkpoint"]).exists()
        if info["selected_is_final"]:
            assert info["policy_tensor_sha256"] == info["final_policy_tensor_sha256"]
    print("PASS test_gate_binds_real_manifest_if_present")


if __name__ == "__main__":
    test_gate_accepts_correct_binding()
    test_gate_rejects_swapped_model()
    test_gate_rejects_different_checkpoint_path()
    test_gate_rejects_wrong_seed_bank()
    test_gate_rejects_env_mismatch()
    test_gate_rejects_missing_final_arm()
    test_gate_rejects_unknown_selected_arm()
    test_analyzer_accepts_complete_records()
    test_analyzer_rejects_missing_episode()
    test_analyzer_rejects_duplicate_episode()
    test_analyzer_rejects_extra_episode()
    test_analyzer_rejects_missing_final_without_flag()
    test_analyzer_reuses_selected_when_manifest_says_final()
    test_analyzer_rejects_flag_true_with_mismatched_hash()
    test_gate_rejects_flag_true_with_mismatched_hash()
    test_gate_binds_real_manifest_if_present()
    print("\nALL V50 ACCEPTANCE-GATE TESTS PASSED")
