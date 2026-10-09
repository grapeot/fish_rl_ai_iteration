"""v57 verification helpers: frozen-manifest binding + strict paired-record
validation.

Two jobs:

1. **Frozen identity.** The manifest records, for each run key
   (`rep{r}_control`, `rep{r}_treatment`), the selection-selected checkpoint and
   the final checkpoint — each with path + policy-tensor SHA-256
   (`model.policy.state_dict()`, stable across re-saves). The evaluation gate
   requires every supplied policy arm to be exactly one of these frozen
   (path, hash) pairs, and requires all *selected* arms to be present; the report
   deliberately runs only the selected checkpoints, so un-selected finals are
   optional. It also binds the full env config (heading bias included), the
   velocity-factor map, and the v54-matched condition set.

2. **Paired-record strictness.** The analysis requires, per (controller, scenario,
   condition), exactly one episode, and exactly the frozen set of conditions and
   scenarios. Missing/duplicate condition, missing/duplicate scenario, unknown
   controller, and any missing (controller, condition) cell all fail.

Model identity is the policy-tensor SHA-256, chosen over the zip byte hash (SB3
zips embed timestamps).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

HASH_SCHEME = "policy_tensor_sha256"
MANIFEST_KIND = "v57_frozen_manifest"
MANIFEST_VERSION = 1

ROOT = Path(__file__).resolve().parents[2]


class ManifestError(RuntimeError):
    """Raised when the frozen v57 manifest or the records do not match."""


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
        raise ManifestError(f"frozen manifest does not exist: {p}")
    manifest = json.loads(p.read_text())
    if manifest.get("kind") != MANIFEST_KIND:
        raise ManifestError(f"not a v57 frozen manifest: {manifest.get('kind')!r}")
    if int(manifest.get("version", 0)) != MANIFEST_VERSION:
        raise ManifestError(f"manifest version mismatch: {manifest.get('version')!r}")
    if manifest.get("hash_scheme") != HASH_SCHEME:
        raise ManifestError(f"unexpected hash scheme: {manifest.get('hash_scheme')!r}")
    return manifest


def _norm(p) -> str:
    """Absolute resolved path; repo-relative paths resolve against the repo root so
    manifest paths and CLI spec paths compare equal."""
    pp = Path(p)
    if not pp.is_absolute():
        pp = ROOT / pp
    return str(pp.resolve())


def _frozen_cells(manifest):
    """Expand run keys into the frozen policy-arm cells: name -> (checkpoint, hash).

    Selected cell name is `{run}_selected`; final cell name is `{run}_final`.
    """
    cells = {}
    for run_key, info in manifest["runs"].items():
        cells[f"{run_key}_selected"] = (info["selected"]["checkpoint"],
                                        info["selected"]["policy_tensor_sha256"])
        cells[f"{run_key}_final"] = (info["final"]["checkpoint"],
                                     info["final"]["policy_tensor_sha256"])
    return cells


def validate_manifest_gate(
    manifest,
    *,
    controller_paths,
    controller_hashes,
    actual_env_config,
    conditions,
    actual_velocity_factors,
    seeds_name,
    actual_seeds,
):
    """Raise ManifestError unless the run is bound to the frozen manifest.

    controller_paths: mapping controller name -> checkpoint path for policy arms.
    controller_hashes: mapping controller name -> policy_tensor_sha256.
    conditions: the evaluation condition names in order.
    actual_velocity_factors: the condition -> factor map actually used by the
        evaluation (the v54-matched factor values). It is a hard binding: a
        report whose half factor differs from the frozen 0.5 is rejected.
    """
    cells = _frozen_cells(manifest)

    # Any supplied policy arm must be a frozen (path, hash) pair.
    for name, path in controller_paths.items():
        if name not in cells:
            raise ManifestError(f"unknown policy arm: {name}")
        frozen_path, frozen_hash = cells[name]
        if _norm(path) != _norm(frozen_path):
            raise ManifestError(f"{name} checkpoint is not the frozen one: {path}")
        if controller_hashes.get(name) != frozen_hash:
            raise ManifestError(f"{name} model identity hash differs from the frozen manifest")

    # All SELECTED arms are required (the report set); finals are optional —
    # the report deliberately reports only the selected checkpoints and never
    # runs un-selected finals, so it must be allowed to omit them.
    missing = {f"{rk}_selected" for rk in manifest["runs"]} - set(controller_paths)
    if missing:
        raise ManifestError(f"missing frozen selected policy arms: {sorted(missing)}")

    if list(manifest["conditions"]) != list(conditions):
        raise ManifestError("condition set differs from the frozen manifest")
    frozen_factors = manifest.get("velocity_factors")
    if frozen_factors is None:
        raise ManifestError("manifest does not freeze a velocity_factors map")
    if dict(actual_velocity_factors) != dict(frozen_factors):
        raise ManifestError(
            f"velocity-factor map differs from the frozen manifest: "
            f"{dict(actual_velocity_factors)} != {dict(frozen_factors)}")
    if manifest["env"]["env_config"] != actual_env_config:
        raise ManifestError("env config differs from the frozen manifest")

    if seeds_name == "report":
        expected = manifest["report_seeds"]
        if sorted(int(s) for s in actual_seeds) != sorted(int(s) for s in expected):
            raise ManifestError("report run does not use the frozen report seed bank")
    else:
        bank = manifest.get("seed_banks", {}).get(seeds_name)
        if bank is None:
            raise ManifestError(f"unknown seed set: {seeds_name!r}")
        if sorted(int(s) for s in actual_seeds) != sorted(int(s) for s in bank["seeds"]):
            raise ManifestError(f"{seeds_name} run does not use the frozen seed bank")


def validate_paired_records(records, manifest, *, conditions):
    """Validate the paired grid (controller x scenario x condition).

    Returns {(controller, seed): {condition: final_survival}} after checking:
      * every record has a known controller (frozen arm or a declared rule);
      * every record's condition is in the frozen set;
      * no duplicate (controller, seed, condition);
      * every controller present covers exactly the frozen scenario set and every
        frozen condition;
      * the required controller set (all frozen *_selected arms plus the rule
        anchors) is present. Un-selected finals may be absent: the report
        deliberately does not run them.
    """
    frozen_ctrl = set(_frozen_cells(manifest)) | set(manifest.get("rule_controllers", []))
    required = ({f"{rk}_selected" for rk in manifest["runs"]}
                | set(manifest.get("rule_controllers", [])))
    report_seeds = set(int(s) for s in manifest["report_seeds"])
    conds = set(conditions)

    grid = {}
    for r in records:
        ctrl = r["controller"]
        cond = r["condition"]
        seed = int(r["seed"])
        if ctrl not in frozen_ctrl:
            raise ManifestError(f"unknown controller in records: {ctrl!r}")
        if cond not in conds:
            raise ManifestError(f"unknown condition in records: {cond!r}")
        cell = grid.setdefault((ctrl, seed), {})
        if cond in cell:
            raise ManifestError(f"duplicate episode (controller={ctrl}, seed={seed}, condition={cond})")
        cell[cond] = float(r["final_survival"])

    present = sorted({c for (c, _s) in grid})
    missing_required = required - set(present)
    if missing_required:
        raise ManifestError(f"missing required controllers: {sorted(missing_required)}")

    for ctrl in present:
        scenarios = sorted({s for (c, s) in grid if c == ctrl})
        if set(scenarios) != report_seeds:
            missing = sorted(report_seeds - set(scenarios))[:5]
            extra = sorted(set(scenarios) - report_seeds)[:5]
            raise ManifestError(
                f"controller {ctrl} scenario set mismatch: n={len(scenarios)} "
                f"(expected {len(report_seeds)}); missing={missing} extra={extra}"
            )
        for seed in report_seeds:
            cell = grid.get((ctrl, seed), {})
            got = set(cell)
            if got != conds:
                raise ManifestError(
                    f"controller {ctrl} seed {seed}: conditions {sorted(got)} "
                    f"!= frozen {sorted(conds)}"
                )
    return grid
