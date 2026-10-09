"""v58 verification helpers: frozen-model binding, frozen-draw binding, and strict
paired-record validation.

Three jobs:

1. **Frozen-model identity.** The evaluation must use exactly the six frozen models
   recorded in the accepted v57 manifest (three reused nominal controls and three
   speed-randomized treatments). Each is bound by its checkpoint path plus loaded
   policy-tensor SHA-256 (`model.policy.state_dict()`), stable across re-saves.
   Every supplied policy arm must be one of those six; all six must be present.

2. **Frozen-draw binding.** The combined-stress triples, the nominal triple and the
   seed banks are frozen into the v58 manifest BEFORE any score. The gate rejects any
   run whose actual nominal triple, actual condition triples or seed set differ from
   the frozen arrays (binding the exact per-scene values, not only the condition
   names).

3. **Paired-record strictness.** The analysis requires, per (controller, scenario,
   condition), exactly one episode, and exactly the frozen set of conditions and
   scenarios; missing/duplicate/extra scenario, controller or condition all fail.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

HASH_SCHEME = "policy_tensor_sha256"
MANIFEST_KIND = "v58_frozen_manifest"
MANIFEST_VERSION = 1

ROOT = Path(__file__).resolve().parents[2]


class ManifestError(RuntimeError):
    """Raised when the frozen v58 manifest or the records do not match."""


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
        raise ManifestError(f"not a v58 frozen manifest: {manifest.get('kind')!r}")
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


def _frozen_models(manifest):
    """name -> (checkpoint, policy_tensor_sha256) from the frozen manifest."""
    return {
        name: (rec["checkpoint"], rec["policy_tensor_sha256"])
        for name, rec in manifest["models"].items()
    }


def validate_manifest_gate(
    manifest,
    *,
    controller_paths,
    controller_hashes,
    actual_env_config,
    conditions,
    actual_triples_by_seed,
    actual_nominal_triple,
    seeds_name,
    actual_seeds,
):
    """Raise ManifestError unless the run is bound to the frozen manifest.

    controller_paths: name -> checkpoint path for policy arms.
    controller_hashes: name -> policy_tensor_sha256.
    actual_triples_by_seed: {seed: [fish, rot, speed]} actually used; a hard
        binding: a scene whose triple differs from the frozen array is rejected.
    actual_nominal_triple: the [fish, rot, speed] the run will actually apply for the
        nominal condition; it must equal the frozen `nominal_triple` exactly.
    """
    frozen = _frozen_models(manifest)

    for name, path in controller_paths.items():
        if name not in frozen:
            raise ManifestError(f"unknown policy arm: {name}")
        frozen_path, frozen_hash = frozen[name]
        if _norm(path) != _norm(frozen_path):
            raise ManifestError(f"{name} checkpoint is not the frozen one: {path}")
        if controller_hashes.get(name) != frozen_hash:
            raise ManifestError(f"{name} model identity hash differs from the frozen manifest")

    missing = set(frozen) - set(controller_paths)
    if missing:
        raise ManifestError(f"missing frozen policy arms: {sorted(missing)}")

    if list(manifest["conditions"]) != list(conditions):
        raise ManifestError("condition set differs from the frozen manifest")

    if manifest["env"]["env_config"] != actual_env_config:
        raise ManifestError("env config differs from the frozen manifest")

    # Nominal-triple binding: the run's actual nominal triple must equal the frozen
    # one exactly (so a mutated nominal cannot silently bind).
    frozen_nominal = [float(manifest["nominal_triple"][0]),
                      int(manifest["nominal_triple"][1]),
                      float(manifest["nominal_triple"][2])]
    got_nominal = [float(actual_nominal_triple[0]), int(actual_nominal_triple[1]),
                   float(actual_nominal_triple[2])]
    if got_nominal != frozen_nominal:
        raise ManifestError(
            f"nominal condition triple differs from the frozen manifest: "
            f"{got_nominal} != {frozen_nominal}")

    # Frozen-draw binding: every scene's combined-stress triple must equal the
    # frozen array (binding the exact values, not only the condition names).
    frozen_triples = {int(s): list(t) for s, t in zip(manifest["report_seeds"],
                                                     manifest["combined_stress_triples"])}
    if seeds_name != "report":
        bank = manifest.get("seed_banks", {}).get(seeds_name)
        if bank is None:
            raise ManifestError(f"unknown seed set: {seeds_name!r}")
        frozen_triples = {int(s): list(t) for s, t in zip(bank["seeds"], bank["combined_stress_triples"])}

    if sorted(int(s) for s in actual_seeds) != sorted(frozen_triples):
        raise ManifestError(f"{seeds_name} run does not use the frozen seed bank")
    for s in actual_seeds:
        got = [float(actual_triples_by_seed[int(s)][0]),
               int(actual_triples_by_seed[int(s)][1]),
               float(actual_triples_by_seed[int(s)][2])]
        want = [float(frozen_triples[int(s)][0]), int(frozen_triples[int(s)][1]),
                float(frozen_triples[int(s)][2])]
        if got != want:
            raise ManifestError(
                f"combined-stress triple for seed {s} differs from the frozen draw: {got} != {want}")


def validate_paired_records(records, manifest, *, conditions):
    """Validate the paired grid (controller x scenario x condition).

    Returns {(controller, seed): {condition: final_survival}} after checking:
      * every record has a known controller (frozen model or a declared rule);
      * every record's condition is in the frozen set;
      * no duplicate (controller, seed, condition);
      * every controller present covers exactly the frozen scenario set and every
        frozen condition.
    """
    frozen_ctrl = set(_frozen_models(manifest)) | set(manifest.get("rule_controllers", []))
    required = set(_frozen_models(manifest)) | set(manifest.get("rule_controllers", []))
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
