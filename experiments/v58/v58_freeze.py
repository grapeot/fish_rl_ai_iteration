#!/usr/bin/env python3
"""v58 freeze: record the six selected models (path + policy-tensor hash) from the
v57 manifest, the world config, the fresh v58 seed banks and the frozen
combined-stress triples into a manifest, BEFORE any v58 score is computed.

The manifest is written only if absent unless `--allow-overwrite` is passed, so a
casual rerun does not silently replace it; the policy-tensor hash comparison is an
identity check, not a cryptographic tamper-proof guarantee. The six models are read
from the v57 frozen manifest and their current on-disk policy-tensor hash is
recomputed and stored. The combined-stress draws use the independent RNG 580202 and
are frozen here for every controller.

Usage:
  experiments/v48/.venv/bin/python experiments/v58/v58_freeze.py \
      --out experiments/v58/artifacts/frozen_config/v58_frozen_manifest.json
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v58_common as common  # noqa: E402
import v58_verify as verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"frozen manifest already exists: {out_path} (use --allow-overwrite)")
    created_utc = None
    if out_path.exists():
        try:
            created_utc = json.loads(out_path.read_text()).get("created_utc")
        except Exception:
            created_utc = None

    v57_manifest = common.load_v57_manifest()
    v57_path = ROOT / common.V57_MANIFEST_RELPATH

    models = {}
    for r in common.REPLICATES:
        for run_key, arm, name in (
            (common.CONTROL_RUN_KEY[r], "control", common.CONTROL_MODEL_NAME[r]),
            (common.TREATMENT_RUN_KEY[r], "treatment", common.TREATMENT_MODEL_NAME[r]),
        ):
            rec = v57_manifest["runs"][run_key]["selected"]
            path = ROOT / rec["checkpoint"]
            if not path.exists():
                raise SystemExit(f"missing frozen checkpoint: {path}")
            h = verify.policy_tensor_sha256_from_zip(path)
            if h != rec["policy_tensor_sha256"]:
                raise SystemExit(
                    f"policy-tensor hash disagrees with the accepted v57 manifest for {run_key}")
            models[name] = {
                "run_key": run_key,
                "arm": arm,
                "replicate": r,
                "update": rec["update"],
                "checkpoint": rec["checkpoint"],
                "policy_tensor_sha256": h,
                "source_manifest": common.V57_MANIFEST_RELPATH,
            }

    # Fresh banks + frozen combined-stress triples (drawn before any score).
    banks = {}
    for name, spec in common.STAGE_SEEDS.items():
        seeds = common.resolve_seeds(name)
        banks[name] = {
            "rng_seed": spec["rng_seed"],
            "n": spec["n"],
            "seeds": seeds,
            "combined_stress_triples": common.draw_combined_stress_triples(seeds),
        }
    report_seeds = banks["report"]["seeds"]
    report_triples = banks["report"]["combined_stress_triples"]

    manifest = {
        "kind": verify.MANIFEST_KIND,
        "version": verify.MANIFEST_VERSION,
        "hash_scheme": verify.HASH_SCHEME,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "role": "v58 round 10/10: independent confirmation of the training-time "
                "initial-velocity randomization under a pre-specified joint stress",
        "depends_on": {
            "v57_manifest": common.V57_MANIFEST_RELPATH,
            "v57_manifest_sha256": common.sha256_file(v57_path),
            "note": "accepted corrected survival_only controls + v57 augmented-velocity "
                    "treatments (v57 round 9/10). No retraining, no reselection on v58.",
        },
        "conditions": list(common.CONDITIONS),
        "fish_factors": list(common.FISH_FACTORS),
        "rotation_degrees": list(common.ROTATION_DEGREES),
        "speed_factors": list(common.SPEED_FACTORS),
        "nominal_triple": list(common.NOMINAL_TRIPLE),
        "aug_rng_seed": common.AUG_RNG_SEED,
        "intervention": (
            "post-reset joint change of ONLY the velocity vectors on the nominal world "
            "(full heading/speed bias and pre-roll): fish initial velocities scaled by a "
            "factor in {0.0,0.5,1.0}; predator initial velocity rotated by an exact "
            "integer matrix of 0/90/180/270 deg then scaled by a positive factor in "
            "{0.75,1.0,1.25}. The nominal condition is the exact identity. All other state "
            "(positions, alive, death timesteps, predator position, timestep, pre-roll "
            "trace) and the env RNG are restored bit-identically; subsequent gravity/bounce "
            "physics are unchanged. The combined-stress triple is one finite discrete draw "
            "per scene, shared by every controller, frozen before any score. This is NOT a "
            "natural deployment distribution and NOT full continuous-domain coverage."
        ),
        "models": models,
        "rule_controllers": list(common.RULE_CONTROLLERS),
        "seed_banks": banks,
        "report_seeds": report_seeds,
        "combined_stress_triples": report_triples,
        "env": {
            "neighbor": False,
            "env_config": common.env_config(False),
            "num_fish": common.NUM_FISH,
            "max_timesteps": common.MAX_TIMESTEPS,
        },
        "statistic": "final_survival = num_alive / 96, per episode (scene)",
        "recommendation_rule": {
            "nominal_loss_tolerance": common.NOMINAL_LOSS_TOLERANCE,
            "nominal_ci_lower_floor": common.NOMINAL_CI_LOWER_FLOOR,
            "note": "pre-declared engineering tolerances, NOT discovered statistical constants; "
                    "never changed after seeing v57/v58 outcomes",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[written] {out_path}")
    print(json.dumps({k: v["policy_tensor_sha256"][:16] for k, v in models.items()}, indent=2))


if __name__ == "__main__":
    main()
