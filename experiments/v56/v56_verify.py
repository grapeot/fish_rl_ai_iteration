"""v56 verification helpers: frozen-controller binding + strict paired-record
validation.

Two jobs:

1. **Frozen-controller identity.** The evaluation must use exactly the three
   accepted v50 corrected `survival_only` FINAL checkpoints, by loaded policy
   tensor SHA-256 (`model.policy.state_dict()`), stable across re-saves (chosen
   over the zip byte hash, which embeds timestamps). The frozen manifest records
   each PPO controller's path + tensor hash and is written before any result.

2. **Paired-record strictness.** The analysis requires, per (controller, scenario,
   condition), exactly one episode, and exactly the frozen set of conditions and
   scenarios. Missing/duplicate condition, missing/duplicate scenario, unknown
   controller, and any missing (controller, condition) cell all fail.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

HASH_SCHEME = "policy_tensor_sha256"
MANIFEST_KIND = "v56_frozen_controller_manifest"
MANIFEST_VERSION = 1


class ManifestError(RuntimeError):
    """Raised when the frozen v56 controller manifest or the records do not match."""


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
        raise ManifestError(f"frozen controller manifest does not exist: {p}")
    manifest = json.loads(p.read_text())
    if manifest.get("kind") != MANIFEST_KIND:
        raise ManifestError(f"not a v56 frozen manifest: {manifest.get('kind')!r}")
    if int(manifest.get("version", 0)) != MANIFEST_VERSION:
        raise ManifestError(f"manifest version mismatch: {manifest.get('version')!r}")
    if manifest.get("hash_scheme") != HASH_SCHEME:
        raise ManifestError(f"unexpected hash scheme: {manifest.get('hash_scheme')!r}")
    return manifest


def _norm(p) -> str:
    return str(Path(p).resolve())


def validate_controller_binding(
    manifest,
    *,
    controller_paths,
    controller_hashes,
    actual_env_config,
    conditions,
    seeds_name,
    actual_seeds,
    speed_factors=None,
):
    """Raise ManifestError unless every PPO controller and the design match.

    controller_paths: mapping controller name -> checkpoint path for PPO arms.
    controller_hashes: mapping controller name -> policy_tensor_sha256.
    speed_factors: optional mapping condition -> factor; when given, it must match
      the frozen manifest's factor map.
    """
    frozen = manifest["controllers"]
    # every supplied PPO controller must be one of the frozen three, path+hash equal
    for name, path in controller_paths.items():
        if name not in frozen:
            raise ManifestError(f"unknown PPO controller: {name}")
        info = frozen[name]
        if _norm(path) != _norm(info["checkpoint"]):
            raise ManifestError(f"{name} checkpoint is not the frozen one: {path}")
        if controller_hashes.get(name) != info["policy_tensor_sha256"]:
            raise ManifestError(f"{name} model identity hash differs from the frozen manifest")
    # and the frozen three must all be present (no silent subset)
    missing = set(frozen) - set(controller_paths)
    if missing:
        raise ManifestError(f"missing frozen PPO controllers: {sorted(missing)}")

    if manifest["conditions"] != list(conditions):
        raise ManifestError("condition set differs from the frozen manifest")
    if speed_factors is not None:
        frozen_factors = manifest.get("speed_factors")
        if frozen_factors != dict(speed_factors):
            raise ManifestError("speed factor map differs from the frozen manifest")
    if manifest["env"]["env_config"] != actual_env_config:
        raise ManifestError("env config differs from the frozen manifest")

    bank = manifest["seed_banks"][seeds_name]
    if sorted(int(s) for s in actual_seeds) != sorted(int(s) for s in bank["seeds"]):
        raise ManifestError(f"{seeds_name} run does not use the frozen seed bank")


def validate_paired_records(records, manifest, *, conditions):
    """Validate the 3-D paired grid (controller x scenario x condition).

    Returns {(controller, seed): {condition: final_survival}} after checking:
      * every record has a known controller (frozen PPO or a declared rule);
      * every record's condition is in the frozen set;
      * no duplicate (controller, seed, condition);
      * every (controller, condition) cell has exactly the frozen scenario set.
    """
    frozen_ctrl = (set(manifest["controllers"]) | set(manifest["rule_controllers"]))
    report_seeds = set(int(s) for s in manifest["seed_banks"]["report"]["seeds"])
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

    # every controller must cover every scenario and every condition
    controllers = sorted(frozen_ctrl)
    for ctrl in controllers:
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
