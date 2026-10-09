#!/usr/bin/env python3
"""v53 acceptance-gate tests: the manifest gate and analyzer must reject tampering
(not merely check existence), and must close the two v50 binding gaps.

Gap 1 (closed): the frozen env binding includes `predator_heading_bias`, so a
report run whose heading config differs is rejected (test_gate_rejects_heading_bias_mismatch).
Gap 2 (closed): `resolve_report_arms` reuses selected-as-final only when the
manifest flag AND the recorded hashes agree; a flag with mismatched hashes is
rejected (test_analyzer_rejects_flag_with_mismatched_hashes).

Run:
  experiments/v48/.venv/bin/python experiments/v53/tests/test_v53_acceptance_gates.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V53 = HERE.parent
ROOT = V53.parents[1]
for p in (str(ROOT), str(V53)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v53_common as common  # noqa: E402
import v53_verify as verify  # noqa: E402


def _fake_env_config():
    cfg = common.env_config(False)   # includes predator_heading_bias
    return cfg


def _fake_manifest(seeds_sel=(1, 2, 3), seeds_rep=(11, 12, 13, 14)):
    keys = [f"rep{r}_{arm}" for r in common.REPLICATES for arm in common.ARMS]
    sel = {}
    for rk in keys:
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
        "selection_seed_bank": {"rng_seed": 530101, "n": len(seeds_sel)},
        "selection_seeds": list(seeds_sel),
        "report_seed_bank": {"rng_seed": 530102, "n": len(seeds_rep)},
        "report_seeds": list(seeds_rep),
        "env": {"neighbor": False, "env_config": _fake_env_config()},
        "selected": sel,
        "selected_only": {},
    }


def _arm_specs_and_hashes(m):
    specs, hashes = [], {}
    for rk, info in m["selected"].items():
        specs.append({"name": f"{rk}_selected", "kind": "policy", "path": info["checkpoint"]})
        hashes[f"{rk}_selected"] = info["policy_tensor_sha256"]
        if not info["selected_is_final"]:
            specs.append({"name": f"{rk}_final", "kind": "policy", "path": info["final_checkpoint"]})
            hashes[f"{rk}_final"] = info["final_policy_tensor_sha256"]
    return specs, hashes


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
    hashes["rep0_control_selected"] = "TAMPERED"
    _expect_fail(m, specs, hashes, "model identity hash differs")
    print("PASS test_gate_rejects_swapped_model")


def test_gate_rejects_different_checkpoint_path():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    for s in specs:
        if s["name"] == "rep1_treatment_selected":
            s["path"] = "fixture/other/checkpoints/model_updates_100.zip"
    _expect_fail(m, specs, hashes, "not the frozen selection")
    print("PASS test_gate_rejects_different_checkpoint_path")


def test_gate_rejects_wrong_seed_bank():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    _expect_fail(m, specs, hashes, "report seed bank", actual_seeds=[999, 998, 997, 996])
    print("PASS test_gate_rejects_wrong_seed_bank")


def test_gate_rejects_env_mismatch():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    bad = dict(m["env"]["env_config"])
    bad["num_fish"] = 250
    _expect_fail(m, specs, hashes, "env evaluation config differs", actual_env_config=bad)
    print("PASS test_gate_rejects_env_mismatch")


def test_gate_rejects_heading_bias_mismatch():
    """Gap 1: heading bias is bound. A report whose heading config differs must fail."""
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    bad = dict(m["env"]["env_config"])
    tampered_heading = [dict(h) for h in bad["predator_heading_bias"]]
    tampered_heading[0]["weight"] = 999.0
    bad["predator_heading_bias"] = tampered_heading
    _expect_fail(m, specs, hashes, "env evaluation config differs", actual_env_config=bad)
    print("PASS test_gate_rejects_heading_bias_mismatch")


def test_gate_rejects_missing_final_arm():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    specs = [s for s in specs if s["name"] != "rep2_control_final"]
    hashes.pop("rep2_control_final", None)
    _expect_fail(m, specs, hashes, "missing the final arm")
    print("PASS test_gate_rejects_missing_final_arm")


def test_gate_rejects_unknown_selected_arm():
    m = _fake_manifest()
    specs, hashes = _arm_specs_and_hashes(m)
    specs.append({"name": "rep9_ghost_selected", "kind": "policy", "path": "fixture/x.zip"})
    hashes["rep9_ghost_selected"] = "x"
    _expect_fail(m, specs, hashes, "unknown run")
    print("PASS test_gate_rejects_unknown_selected_arm")


def test_analyzer_accepts_complete_records():
    m = _fake_manifest()
    resolved = verify.resolve_report_arms(_records_from_manifest(m), m,
                                          expected_n=len(m["report_seeds"]))
    assert "rep0_control_selected" in resolved
    assert "rep0_control_final" in resolved
    print("PASS test_analyzer_accepts_complete_records")


def test_analyzer_rejects_missing_episode():
    m = _fake_manifest()
    recs = _records_from_manifest(m, drop=("rep1_control_selected", m["report_seeds"][0]))
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
    recs.append({"control": "rep0_control_selected", "seed": 987654321, "final_survival": 0.9})
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "seed set mismatch" in str(exc)
        print("PASS test_analyzer_rejects_extra_episode")
        return
    raise AssertionError("analyzer accepted an extra episode")


def test_analyzer_rejects_missing_final_without_flag():
    m = _fake_manifest()
    recs = [r for r in _records_from_manifest(m) if not r["control"].endswith("_final")]
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "does not mark selected_is_final" in str(exc)
        print("PASS test_analyzer_rejects_missing_final_without_flag")
        return
    raise AssertionError("analyzer silently backfilled final")


def test_analyzer_reuses_selected_when_flag_and_hashes_agree():
    m = _fake_manifest()
    m["selected"]["rep0_control"]["selected_is_final"] = True
    m["selected"]["rep0_control"]["final_policy_tensor_sha256"] = \
        m["selected"]["rep0_control"]["policy_tensor_sha256"]
    recs = _records_from_manifest(m)   # no _final generated for rep0_control
    resolved = verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    assert resolved["rep0_control_final"] == resolved["rep0_control_selected"]
    print("PASS test_analyzer_reuses_selected_when_flag_and_hashes_agree")


def test_analyzer_rejects_flag_with_mismatched_hashes():
    """Gap 2: a selected_is_final flag with differing selected/final hashes fails."""
    m = _fake_manifest()
    m["selected"]["rep0_control"]["selected_is_final"] = True
    # hashes intentionally differ (final hash != selected hash)
    recs = [r for r in _records_from_manifest(m)
            if r["control"] != "rep0_control_final"]
    try:
        verify.resolve_report_arms(recs, m, expected_n=len(m["report_seeds"]))
    except verify.ManifestError as exc:
        assert "hashes differ" in str(exc)
        print("PASS test_analyzer_rejects_flag_with_mismatched_hashes")
        return
    raise AssertionError("analyzer accepted a flag with mismatched selected/final hashes")


if __name__ == "__main__":
    test_gate_accepts_correct_binding()
    test_gate_rejects_swapped_model()
    test_gate_rejects_different_checkpoint_path()
    test_gate_rejects_wrong_seed_bank()
    test_gate_rejects_env_mismatch()
    test_gate_rejects_heading_bias_mismatch()
    test_gate_rejects_missing_final_arm()
    test_gate_rejects_unknown_selected_arm()
    test_analyzer_accepts_complete_records()
    test_analyzer_rejects_missing_episode()
    test_analyzer_rejects_duplicate_episode()
    test_analyzer_rejects_extra_episode()
    test_analyzer_rejects_missing_final_without_flag()
    test_analyzer_reuses_selected_when_flag_and_hashes_agree()
    test_analyzer_rejects_flag_with_mismatched_hashes()
    print("\nALL V53 ACCEPTANCE-GATE TESTS PASSED")
