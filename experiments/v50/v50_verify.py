"""v50 verification helpers: manifest binding checks + report record strictness.

These turn the selection manifest from a mere "file exists" gate into an actual
binding: the report run must prove that (a) it uses the frozen report seed bank,
(b) its env evaluation parameters match the frozen ones, and (c) every policy arm
is exactly the checkpoint the frozen selection picked, by loaded-model identity —
not by a filename someone typed in.

Model identity is the POLICY TENSoR SHA-256 (`policy_tensor_sha256`), computed
from `model.policy.state_dict()` (sorted keys, contiguous float bytes). It is
explicitly chosen over the zip byte-hash because SB3 zips embed timestamps and
re-saving the same weights yields a different byte-hash; the tensor hash is stable
and is what the pilot and corrected manifests both use.

The analyzer uses `validate_report_records` to require, per arm, exactly the
frozen report seed set (no missing, no duplicate, no extra) and to resolve the
"final" stage only when the manifest explicitly records `selected_is_final`.
"""

import hashlib
from pathlib import Path

import numpy as np

HASH_SCHEME = "policy_tensor_sha256"
MANIFEST_KIND = "v50_selection_manifest"
MANIFEST_VERSION = 2


class ManifestError(RuntimeError):
    """Raised when the frozen selection manifest does not match the report run."""


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
    import json

    p = Path(path)
    if not p.exists():
        raise ManifestError(f"selection manifest does not exist: {p}")
    manifest = json.loads(p.read_text())
    if manifest.get("kind") != MANIFEST_KIND:
        raise ManifestError(f"not a v50 selection manifest: {manifest.get('kind')!r}")
    if int(manifest.get("version", 0)) != MANIFEST_VERSION:
        raise ManifestError(f"manifest version mismatch: {manifest.get('version')!r}")
    if manifest.get("hash_scheme") != HASH_SCHEME:
        raise ManifestError(f"unexpected hash scheme: {manifest.get('hash_scheme')!r}")
    return manifest


def _norm(p) -> str:
    return str(Path(p).resolve())


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

    arm_specs: list of dicts with keys name, kind ('policy'|'rule'|'untrained'),
               path (for policy/untrained).
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

    # --- env parameter binding ----------------------------------------------
    frozen_env = manifest["env"]
    if bool(frozen_env["neighbor"]) != bool(neighbor):
        raise ManifestError("neighbor feature flag differs from the frozen manifest")
    if frozen_env["env_config"] != actual_env_config:
        raise ManifestError("env evaluation config differs from the frozen manifest")

    # --- arm <-> checkpoint <-> hash binding --------------------------------
    expected = manifest["selected"]
    seen_selected, seen_final = set(), set()
    for spec in arm_specs:
        if spec["kind"] not in ("policy",):
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
        if info["selected_is_final"]:
            # A reuse claim must be backed by identical model identity; a manifest
            # that flags selected_is_final with differing hashes is inconsistent.
            if info["policy_tensor_sha256"] != info["final_policy_tensor_sha256"]:
                raise ManifestError(
                    f"{rk} marks selected_is_final but its selected/final policy hashes differ"
                )
        elif rk not in seen_final:
            raise ManifestError(f"report is missing the final arm for run {rk}")

    # --- rule/untrained arms are allowed but must not duplicate ----
    names = [s["name"] for s in arm_specs]
    if len(names) != len(set(names)):
        raise ManifestError("duplicate arm names in report")


def resolve_report_arms(records, manifest, expected_n):
    """Strict per-arm seed-set check and final/selected resolution.

    records: iterable of report records (dicts with control, seed, final_survival).
    Returns: dict arm -> {seed: final_survival} after validation. Raises
    ManifestError on any duplicate/missing/extra seed, or when a needed final arm
    is absent and the manifest does not record selected_is_final for that run.
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

    # final/selected resolution bound to the manifest
    resolved = {}
    for rk, info in manifest["selected"].items():
        sel_name = f"{rk}_selected"
        if sel_name not in by_arm:
            raise ManifestError(f"missing selected arm {sel_name}")
        resolved[sel_name] = by_arm[sel_name]
        final_name = f"{rk}_final"
        if final_name in by_arm:
            resolved[final_name] = by_arm[final_name]
        elif info["selected_is_final"]:
            # Only reuse the selected records when the manifest itself proves the
            # two stages are the same model (identical policy tensor hash). A flag
            # alone is not sufficient.
            if info["policy_tensor_sha256"] != info["final_policy_tensor_sha256"]:
                raise ManifestError(
                    f"cannot reuse selected as final for {rk}: "
                    f"selected and final policy hashes differ"
                )
            resolved[final_name] = by_arm[sel_name]
        else:
            raise ManifestError(
                f"missing final arm {final_name} and the manifest does not mark selected_is_final"
            )
    resolved["_by_arm"] = by_arm
    return resolved
