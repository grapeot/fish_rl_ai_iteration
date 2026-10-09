"""v53 verification helpers: manifest binding + strict report-record validation.

This is a narrow, self-contained fork of `v50_verify` with two acceptance gaps
closed, as required for round 5:

  1. **Full env binding.** The frozen manifest binds the *complete* env config,
     including `predator_heading_bias` (which v50 filtered out of both the
     manifest and the evaluator's `actual_env_config`). A report run whose env
     parameters differ in ANY field, heading bias included, is rejected.
  2. **No flag-only final reuse.** `resolve_report_arms` may reuse the selected
     records as the final arm ONLY when the manifest records
     `selected_is_final=True` AND the recorded selected/final policy-tensor
     hashes are identical. A flag that disagrees with the hashes is rejected.

Model identity is the policy-tensor SHA-256 (`model.policy.state_dict()`), stable
across re-saves, chosen over the zip byte hash (SB3 zips embed timestamps).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

HASH_SCHEME = "policy_tensor_sha256"
MANIFEST_KIND = "v53_selection_manifest"
MANIFEST_VERSION = 1

ROOT = Path(__file__).resolve().parents[2]


class ManifestError(RuntimeError):
    """Raised when the frozen v53 selection manifest does not match the report run."""


def policy_tensor_sha256(model) -> str:
    h = hashlib.sha256()
    state = model.policy.state_dict()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def policy_tensor_sha256_from_zip(path) -> str:
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    return policy_tensor_sha256(model)


def load_manifest(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"selection manifest does not exist: {p}")
    manifest = json.loads(p.read_text())
    if manifest.get("kind") != MANIFEST_KIND:
        raise ManifestError(f"not a v53 selection manifest: {manifest.get('kind')!r}")
    if int(manifest.get("version", 0)) != MANIFEST_VERSION:
        raise ManifestError(f"manifest version mismatch: {manifest.get('version')!r}")
    if manifest.get("hash_scheme") != HASH_SCHEME:
        raise ManifestError(f"unexpected hash scheme: {manifest.get('hash_scheme')!r}")
    return manifest


def _norm(p) -> str:
    """Normalize to an absolute resolved path; repo-relative paths resolve against
    the repo root so manifest paths (repo-relative) and CLI spec paths (absolute)
    compare equal."""
    pp = Path(p)
    if not pp.is_absolute():
        pp = ROOT / pp
    return str(pp.resolve())


def validate_manifest_gate(
    manifest,
    *,
    seeds_name,
    actual_seeds,
    neighbor,
    actual_env_config,
    arm_specs,
    arm_hashes,
):
    """Raise ManifestError unless the report run is bound to the frozen selection.

    arm_specs: list of dicts {name, kind ('policy'|'rule'|'untrained'), path}.
    arm_hashes: mapping name -> policy_tensor_sha256 for policy arms.
    """
    # --- seed bank binding --------------------------------------------------
    report_seeds = list(manifest["report_seeds"])
    if seeds_name == "report":
        if sorted(int(s) for s in actual_seeds) != sorted(int(s) for s in report_seeds):
            raise ManifestError("report run does not use the frozen report seed bank")
    else:
        sel = set(int(s) for s in manifest["selection_seeds"])
        rep = set(int(s) for s in report_seeds)
        if set(int(s) for s in actual_seeds) & (sel | rep):
            raise ManifestError(f"seed set {seeds_name} overlaps the frozen selection/report banks")

    # --- FULL env parameter binding (heading bias included) -----------------
    frozen_env = manifest["env"]
    if bool(frozen_env["neighbor"]) != bool(neighbor):
        raise ManifestError("neighbor feature flag differs from the frozen manifest")
    if frozen_env["env_config"] != actual_env_config:
        raise ManifestError("env evaluation config differs from the frozen manifest")

    # --- arm <-> checkpoint <-> hash binding --------------------------------
    expected = manifest["selected"]
    seen_selected, seen_final = set(), set()
    for spec in arm_specs:
        if spec["kind"] != "policy":
            continue
        name = spec["name"]
        path = _norm(spec["path"])
        if name.endswith("_selected"):
            rk = name[: -len("_selected")]
            if rk not in expected:
                raise ManifestError(f"selected arm {name} refers to unknown run {rk}")
            info = expected[rk]
            if path != _norm(info["checkpoint"]):
                raise ManifestError(f"{name} checkpoint is not the frozen selection: {path}")
            if arm_hashes.get(name) != info["policy_tensor_sha256"]:
                raise ManifestError(f"{name} model identity hash differs from the frozen manifest")
            seen_selected.add(rk)
        elif name.endswith("_final"):
            rk = name[: -len("_final")]
            if rk not in expected:
                raise ManifestError(f"final arm {name} refers to unknown run {rk}")
            info = expected[rk]
            if path != _norm(info["final_checkpoint"]):
                raise ManifestError(f"{name} checkpoint is not this run's frozen final: {path}")
            if arm_hashes.get(name) != info["final_policy_tensor_sha256"]:
                raise ManifestError(f"{name} model identity hash differs from the frozen manifest")
            seen_final.add(rk)
        else:
            raise ManifestError(f"policy arm {name} is neither a selected nor a final arm")

    if seen_selected != set(expected):
        missing = set(expected) - seen_selected
        raise ManifestError(f"report is missing selected arms for runs: {sorted(missing)}")
    for rk, info in expected.items():
        if not info["selected_is_final"] and rk not in seen_final:
            raise ManifestError(f"report is missing the final arm for run {rk}")

    names = [s["name"] for s in arm_specs]
    if len(names) != len(set(names)):
        raise ManifestError("duplicate arm names in report")


def resolve_report_arms(records, manifest, expected_n):
    """Strict per-arm seed-set check and manifest-bound final/selected resolution.

    Requires, per arm, exactly the frozen report seed set. Resolves the final arm
    only when a real final record exists, or when the manifest records
    `selected_is_final=True` AND the recorded selected/final hashes agree (a real
    hash-verified reuse). A flag without matching hashes is rejected.
    """
    report_seeds = set(int(s) for s in manifest["report_seeds"])
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["control"], {})
        s = int(r["seed"])
        if s in by_arm[r["control"]]:
            raise ManifestError(f"duplicate episode (arm={r['control']}, seed={s})")
        by_arm[r["control"]][s] = float(r["final_survival"])

    for arm, per_seed in by_arm.items():
        got = set(per_seed)
        if got != report_seeds:
            missing = sorted(report_seeds - got)[:5]
            extra = sorted(got - report_seeds)[:5]
            raise ManifestError(
                f"arm {arm} seed set mismatch: n={len(got)} (expected {expected_n}); "
                f"missing={missing} extra={extra}"
            )

    resolved = {}
    for rk, info in manifest["selected"].items():
        sel_name = f"{rk}_selected"
        if sel_name not in by_arm:
            raise ManifestError(f"missing selected arm {sel_name}")
        resolved[sel_name] = by_arm[sel_name]
        final_name = f"{rk}_final"
        if final_name in by_arm:
            resolved[final_name] = by_arm[final_name]
        elif info.get("selected_is_final"):
            if info.get("policy_tensor_sha256") != info.get("final_policy_tensor_sha256"):
                raise ManifestError(
                    f"run {rk}: selected_is_final=True but selected/final hashes differ"
                )
            resolved[final_name] = by_arm[sel_name]
        else:
            raise ManifestError(
                f"missing final arm {final_name} and the manifest does not mark selected_is_final"
            )
    resolved["_by_arm"] = by_arm
    return resolved
