#!/usr/bin/env python3
"""v55 freeze: record the three accepted v50 corrected `survival_only` FINAL
checkpoints (path + policy-tensor hash), the world config and the seed banks
into a non-overwritable manifest, BEFORE any v55 result is produced.

The three checkpoints are read from the merged v50 corrected selection manifest
(PR#11) and their current on-disk policy-tensor hash is recomputed and stored, so
the v55 result can never be silently re-pointed at a different model.

Usage:
  experiments/v48/.venv/bin/python experiments/v55/v55_freeze.py \
      --out experiments/v55/artifacts/frozen_config/v55_frozen_manifest.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v55_common as common  # noqa: E402
import v55_verify as verify  # noqa: E402

V50_MANIFEST_RELPATH = "experiments/v50/artifacts/corrected_streams/results/selection_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"frozen manifest already exists: {out_path} (use --allow-overwrite)")

    v50_manifest = json.loads((ROOT / V50_MANIFEST_RELPATH).read_text())

    controllers = {}
    for r in (0, 1, 2):
        rel = common.PPO_CHECKPOINT_RELPATH[r]
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing frozen checkpoint: {path}")
        # cross-check against the merged v50 manifest record, if present
        v50_key = f"rep{r}_survival_only"
        v50_rec = v50_manifest["selected"].get(v50_key)
        if v50_rec is None:
            raise SystemExit(f"v50 manifest has no record for {v50_key}")
        if Path(v50_rec["final_checkpoint"]).resolve() != path.resolve():
            raise SystemExit(f"checkpoint path disagrees with the v50 manifest for {v50_key}")
        h = verify.policy_tensor_sha256_from_zip(path)
        if h != v50_rec["final_policy_tensor_sha256"]:
            raise SystemExit(f"policy-tensor hash disagrees with the v50 manifest for {v50_key}")
        controllers[f"rep{r}_ppo"] = {
            "source_run": v50_key,
            "stage": "final",
            "updates": 200,
            "total_steps": 614400,
            "checkpoint": rel,
            "policy_tensor_sha256": h,
        }

    manifest = {
        "kind": verify.MANIFEST_KIND,
        "version": verify.MANIFEST_VERSION,
        "hash_scheme": verify.HASH_SCHEME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "v55 round 7/10: post-reset predator-velocity rotation one-axis generalization probe",
        "depends_on": {
            "v50_manifest": V50_MANIFEST_RELPATH,
            "note": "accepted corrected survival_only FINAL models (PR#11 merged, 7e682e7). "
                    "No dependence on the unaccepted v54 or on v53.",
        },
        "conditions": list(common.CONDITIONS),
        "rotation_degrees": common.ROTATION_DEGREES,
        "rotation_matrices": {k: [list(r) for r in m] for k, m in common.ROTATION_MATRICES.items()},
        "intervention": (
            "post-reset deterministic rotation of ONLY predator_vel by 0/90/180/270 deg via "
            "exact integer matrices (norm preserved exactly), on the nominal world (full "
            "heading/speed bias and pre-roll). NOT a resample of the natural initial-heading "
            "distribution and NOT a gravity turn; fish pos/vel/alive/timestep/pre-roll/RNG held "
            "identical. deg0 is the exact identity and must reproduce the nominal."
        ),
        "controllers": controllers,
        "rule_controllers": list(common.RULE_CONTROLLERS),
        "seed_banks": {
            name: {"rng_seed": spec["rng_seed"], "n": spec["n"],
                   "seeds": common.resolve_seeds(name)}
            for name, spec in common.STAGE_SEEDS.items()
        },
        "env": {
            "neighbor": False,
            "env_config": common.env_config(False),
            "num_fish": common.NUM_FISH,
            "max_timesteps": common.MAX_TIMESTEPS,
        },
        "statistic": "final_survival = num_alive / 96, per episode (scenario)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[written] {out_path}")
    print(json.dumps({k: v["policy_tensor_sha256"][:16] for k, v in controllers.items()}, indent=2))


if __name__ == "__main__":
    main()
