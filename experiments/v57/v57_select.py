#!/usr/bin/env python3
"""v57 checkpoint selection + pre-registration freeze.

Pre-registration (fixed before any report data): for EVERY run key
(`rep{r}_control`, `rep{r}_treatment`) only the u100 and u200 checkpoints are
considered (u200 == the run's final). Each is evaluated on the fresh,
pre-registered selection bank 57010112 (24 paired scenarios, all three velocity
conditions), and the stage with the highest SPEED-BALANCED mean final survival is
selected — the equal-weight mean of the nominal/half/zero condition means; ties
are broken toward the earlier update. The rule is identical for both arms.

The control checkpoints are the REUSED accepted v50 corrected `survival_only`
runs; the treatment checkpoints are the new v57 augmented-velocity runs. The
control selection is cross-checked against the merged v50 selection manifest
(PR#11) path + final policy-tensor hash before anything is written.

The non-overwritable manifest records, per run, the selected and final checkpoint
paths + policy-tensor hashes, the FULL env config (heading bias included), both
seed banks, and the fixed selection rule. It is written only if absent (or with
--allow-overwrite) and is never regenerated after report data exists.

Usage:
  experiments/v48/.venv/bin/python experiments/v57/v57_select.py \
      --out-jsonl experiments/v57/artifacts/results/selection.jsonl \
      --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE), str(ROOT / "experiments" / "v50")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import v57_common as common  # noqa: E402
import v57_verify as verify  # noqa: E402

V50_MANIFEST_RELPATH = "experiments/v50/artifacts/corrected_streams/results/selection_manifest.json"


def _rel(p) -> str:
    """Repo-relative path (public-safe; no absolute home path in any artifact)."""
    return str(Path(p).resolve().relative_to(ROOT.resolve()))


def _abs(p) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp)


def checkpoint_paths(run_dir: Path):
    """The pre-registered candidate stages: u100 and u200(final)."""
    ck = run_dir / "checkpoints"
    out = {}
    u100 = ck / "model_updates_100.zip"
    if u100.exists():
        out[100] = u100
    final = ck / "model_final.zip"
    if final.exists():
        out[200] = final
    return out


def run_dirs():
    out = {}
    for r in common.REPLICATES:
        out[f"rep{r}_control"] = ROOT / common.CONTROL_RUN_DIR[r]
        out[f"rep{r}_treatment"] = ROOT / common.TREATMENT_RUN_DIR[r]
    for key, d in out.items():
        if not d.is_dir():
            raise SystemExit(f"missing run dir for {key}: {d}")
    return out


def check_control_identity():
    """Cross-check the reused control runs against the merged v50 manifest."""
    v50_manifest = json.loads((ROOT / V50_MANIFEST_RELPATH).read_text())
    for r in common.REPLICATES:
        key = f"rep{r}_survival_only"
        rec = v50_manifest["selected"].get(key)
        if rec is None:
            raise SystemExit(f"v50 manifest has no record for {key}")
        final_path = _abs(common.CONTROL_RUN_DIR[r]) / "checkpoints" / "model_final.zip"
        if Path(rec["final_checkpoint"]).resolve() != final_path.resolve():
            raise SystemExit(f"control {key} final path disagrees with the v50 manifest")
        h = verify.policy_tensor_sha256_from_zip(final_path)
        if h != rec["final_policy_tensor_sha256"]:
            raise SystemExit(f"control {key} policy hash disagrees with the v50 manifest")
    return v50_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="reuse a frozen selection summary instead of re-evaluating")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"frozen manifest already exists: {manifest_path} (use --allow-overwrite)")

    created_utc = None
    if manifest_path.exists():
        try:
            created_utc = json.loads(manifest_path.read_text()).get("created_utc")
        except Exception:
            created_utc = None

    check_control_identity()
    runs = run_dirs()

    arms = []
    candidates = {}
    for key, run_dir in runs.items():
        ckpts = checkpoint_paths(run_dir)
        if not ckpts:
            raise SystemExit(f"no pre-registered checkpoints under {run_dir}")
        candidates[key] = []
        for stage in sorted(ckpts):
            name = f"{key}_u{stage}"
            arms.append(f"{name}={ckpts[stage]}")
            candidates[key].append({"name": name, "stage": stage, "path": _rel(ckpts[stage])})

    seeds = common.resolve_seeds("selection")
    summary_path = Path(args.out_jsonl).with_suffix(".summary.json")
    reuse = args.reuse_existing and summary_path.exists()
    if reuse:
        print(f"[reuse] using frozen selection summary at {summary_path}; not re-evaluating")
        summary = json.loads(summary_path.read_text())
    else:
        cmd = [
            sys.executable, str(HERE / "v57_evaluate.py"),
            "--seeds", "selection", "--workers", str(args.workers), "--out", args.out_jsonl,
        ]
        for a in arms:
            cmd += ["--arm", a]
        print("running selection eval:\n", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout[-2000:])
        if res.returncode != 0:
            print(res.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"selection eval failed ({res.returncode})")
        summary = json.loads(summary_path.read_text())
    arm_stats = summary["arms"] if "arms" in summary else summary["controllers"]

    def balanced_mean(stats):
        """Pre-registered selection objective: equal-weight mean of the 3
        condition mean final survivals for this arm."""
        if isinstance(stats, dict) and all(c in stats for c in common.CONDITIONS):
            return float(np.mean([stats[c]["mean_final_survival"] for c in common.CONDITIONS]))
        return stats["mean_final_survival"]

    def n_of(stats):
        if isinstance(stats, dict) and "nominal" in stats:
            return stats["nominal"]["n"]
        return stats["n"]

    selected = {}
    for key, cands in candidates.items():
        best = None
        for c in cands:
            stats = arm_stats[c["name"]]
            mean = balanced_mean(stats)
            if best is None or mean > best["selection_speed_balanced_mean"] + 1e-12:
                best = {
                    "selected_arm": c["name"],
                    "selected_update": c["stage"],
                    "checkpoint": c["path"],
                    "selection_speed_balanced_mean": mean,
                    "selection_n": n_of(stats),
                }
            elif abs(mean - best["selection_speed_balanced_mean"]) <= 1e-12 and c["stage"] < best["selected_update"]:
                best.update({
                    "selected_arm": c["name"],
                    "selected_update": c["stage"],
                    "checkpoint": c["path"],
                    "selection_speed_balanced_mean": mean,
                    "selection_n": n_of(stats),
                })
        best["policy_tensor_sha256"] = verify.policy_tensor_sha256_from_zip(_abs(best["checkpoint"]))
        final_path = runs[key] / "checkpoints" / "model_final.zip"
        if not final_path.exists():
            raise SystemExit(f"missing final checkpoint for {key}: {final_path}")
        best["final_checkpoint"] = _rel(final_path)
        best["final_policy_tensor_sha256"] = verify.policy_tensor_sha256_from_zip(final_path)
        best["selected_is_final"] = (
            _abs(best["checkpoint"]).resolve() == final_path.resolve()
            and best["policy_tensor_sha256"] == best["final_policy_tensor_sha256"]
        )
        best["candidates"] = [
            {
                "arm": c["name"],
                "update": c["stage"],
                "mean_final_survival_speed_balanced": balanced_mean(arm_stats[c["name"]]),
                "mean_final_survival_nominal": (
                    arm_stats[c["name"]]["nominal"]["mean_final_survival"]
                    if isinstance(arm_stats[c["name"]], dict) and "nominal" in arm_stats[c["name"]]
                    else None
                ),
            }
            for c in sorted(cands, key=lambda x: x["stage"])
        ]
        selected[key] = best

    runs_block = {}
    for key, best in selected.items():
        runs_block[key] = {
            "arm": "control" if key.endswith("control") else "treatment",
            "selected": {
                "update": best["selected_update"],
                "checkpoint": best["checkpoint"],
                "policy_tensor_sha256": best["policy_tensor_sha256"],
                "selection_speed_balanced_mean": best["selection_speed_balanced_mean"],
                "selection_n": best["selection_n"],
            },
            "final": {
                "update": 200,
                "checkpoint": best["final_checkpoint"],
                "policy_tensor_sha256": best["final_policy_tensor_sha256"],
            },
            "selected_is_final": best["selected_is_final"],
            "candidates": best["candidates"],
        }

    manifest = {
        "kind": verify.MANIFEST_KIND,
        "version": verify.MANIFEST_VERSION,
        "hash_scheme": verify.HASH_SCHEME,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "role": "v57 round 9/10: train-time per-episode initial-velocity randomization",
        "depends_on": {
            "control": "reused accepted v50 corrected survival_only (PR#11, 7e682e7)",
            "evaluation_conditions": "v54 post-reset fish-velocity factors (540102)",
        },
        "selection_seed_bank": common.STAGE_SEEDS["selection"],
        "selection_seeds": seeds,
        "report_seed_bank": common.STAGE_SEEDS["report"],
        "report_seeds": common.resolve_seeds("report"),
        "seed_banks": {
            name: {"rng_seed": spec["rng_seed"], "n": spec["n"], "seeds": common.resolve_seeds(name)}
            for name, spec in common.STAGE_SEEDS.items()
        },
        "rule_controllers": list(common.RULE_CONTROLLERS),
        "conditions": list(common.CONDITIONS),
        "velocity_factors": common.VELOCITY_FACTORS,
        "env": {
            "neighbor": False,
            "env_config": common.env_config(False),   # FULL config, heading bias included
            "num_fish": common.NUM_FISH,
            "max_timesteps": common.MAX_TIMESTEPS,
        },
        "treatment_intervention": {
            "augment_velocity": True,
            "aug_factors": list(common.AUG_FACTORS),
            "aug_seed_offset": common.AUG_SEED_OFFSET,
            "note": "per-episode reset scales ALL fish initial velocities by a factor "
                    "drawn uniformly from {1.0,0.5,0.0} on an augmentation RNG "
                    "independent of the base world RNG; focal obs recomputed.",
    },
        "rule": (
            "Pre-registered: for each run key consider only u100 and u200(final); pick the "
            "stage with the highest SPEED-BALANCED mean final survival — the equal-weight mean "
            "of the nominal/half/zero condition means — over the frozen selection bank "
            "57010112 (24 eps); ties broken toward the earliest update. Identical rule for "
            "both arms. Computed and frozen before any report."
        ),
        "statistic": "equal-weight mean over {nominal,half,zero} of mean final_survival, "
                     "per selection episode",
        "runs": runs_block,
        "selection_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "selected_only": {k: v["selected_arm"] for k, v in selected.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[written] {args.out_jsonl}")
    print(f"[written] {summary_path}")
    print(f"[written] {manifest_path}")
    print(json.dumps({k: (v["selected_arm"], round(v["selection_speed_balanced_mean"], 6))
                      for k, v in selected.items()}, indent=2))


if __name__ == "__main__":
    main()
