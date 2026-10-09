#!/usr/bin/env python3
"""v50 checkpoint selection on the frozen selection seed bank, then freeze a
non-overwritable selection manifest BEFORE any report data is produced.

Selection rule (frozen, written into the manifest):
  * for each (replicate, arm), evaluate every saved stage checkpoint on the
    selection bank (500201, 24 episodes, paired seeds);
  * pick the stage with the highest mean final survival;
  * ties broken toward the EARLIEST update.

The manifest records: the exact rule, the selected checkpoint per (replicate,
arm) with its policy-tensor hash, the run's final checkpoint and hash, the report
seed bank, the full env config (including predator heading/speed bias), and a UTC
timestamp. It is written only if absent (`--allow-overwrite` to replace);
`v50_evaluate.py`'s report gate refuses to run report data without it.

For a report repro, see the command block in `experiments/v50/dev_v50.md`; the
canonical output dir is `experiments/v50/artifacts/corrected_streams/`.

Usage:
  experiments/v48/.venv/bin/python experiments/v50/v50_select.py \
      --runs-dir experiments/v50/artifacts/corrected_streams/runs \
      --out-jsonl experiments/v50/artifacts/corrected_streams/results/selection.jsonl \
      --manifest experiments/v50/artifacts/corrected_streams/results/selection_manifest.json
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
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import v50_common as common  # noqa: E402
import v50_verify as verify  # noqa: E402

STAGES = [50, 100, 150, 200]


def policy_state_hash_from_zip(path: Path) -> str:
    """Load a PPO checkpoint and hash its policy tensors (portable, timestamp-free)."""
    return verify.policy_tensor_sha256_from_zip(path)


def run_dirs(runs_dir: Path):
    return sorted([d for d in runs_dir.glob("rep*") if d.is_dir()])


def run_key(run_dir: Path):
    # rep{r}_{arm}
    stem = run_dir.name
    rep = int(stem.split("_")[0][3:])
    arm = stem.split("_", 1)[1]
    return rep, arm


def checkpoint_paths(run_dir: Path):
    ck = run_dir / "checkpoints"
    out = {}
    for stage in STAGES[:-1]:
        p = ck / f"model_updates_{stage}.zip"
        if p.exists():
            out[stage] = p
    final = ck / "model_final.zip"
    if final.exists():
        out[STAGES[-1]] = final
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="reuse an existing frozen selection summary instead of re-evaluating")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"selection manifest already exists: {manifest_path} (use --allow-overwrite)")

    # Preserve the original freeze time when upgrading an existing manifest to a
    # new format, so the recorded selection instant is not falsified.
    created_utc = None
    if manifest_path.exists():
        try:
            created_utc = json.loads(manifest_path.read_text()).get("created_utc")
        except Exception:
            created_utc = None

    runs = run_dirs(Path(args.runs_dir))
    if not runs:
        raise SystemExit("no run directories found")

    # Build the candidate checkpoint list. If a prior selection summary already
    # exists on disk (the frozen selection records), reuse it instead of re-running
    # the selection eval; a manifest-only upgrade must not touch the frozen numbers.
    summary_path = Path(args.out_jsonl).with_suffix(".summary.json")
    reuse = args.reuse_existing and summary_path.exists()
    if reuse:
        print(f"[reuse] using frozen selection summary at {summary_path}; not re-evaluating")
        summary = json.loads(summary_path.read_text())
        arm_stats = summary["arms"]

    arms = []
    candidates = {}
    runs_by_key = {}
    for run_dir in runs:
        rep, arm = run_key(run_dir)
        ckpts = checkpoint_paths(run_dir)
        if not ckpts:
            raise SystemExit(f"no checkpoints under {run_dir}")
        runs_by_key[f"rep{rep}_{arm}"] = run_dir
        candidates[f"rep{rep}_{arm}"] = []
        for stage in sorted(ckpts):
            name = f"rep{rep}_{arm}_u{stage}"
            arms.append(f"{name}={ckpts[stage]}")
            candidates[f"rep{rep}_{arm}"].append({"name": name, "stage": stage, "path": str(ckpts[stage])})

    seeds = common.resolve_seeds("selection")
    if not reuse:
        cmd = [
            sys.executable, str(HERE / "v50_evaluate.py"),
            "--seeds", "selection", "--neighbor", "off",
            "--workers", str(args.workers),
            "--out", args.out_jsonl,
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
        arm_stats = summary["arms"]

    selected = {}
    for key, cands in candidates.items():
        best = None
        for c in cands:
            stats = arm_stats[c["name"]]
            mean = stats["mean_final_survival"]
            if best is None or mean > best["selection_mean_final_survival"] + 1e-12:
                best = {
                    "selected_arm": c["name"],
                    "selected_update": c["stage"],
                    "checkpoint": c["path"],
                    "selection_mean_final_survival": mean,
                    "selection_n": stats["n"],
                }
            elif abs(mean - best["selection_mean_final_survival"]) <= 1e-12 and c["stage"] < best["selected_update"]:
                best.update({
                    "selected_arm": c["name"],
                    "selected_update": c["stage"],
                    "checkpoint": c["path"],
                    "selection_mean_final_survival": mean,
                    "selection_n": stats["n"],
                })
        best["policy_state_hash"] = policy_state_hash_from_zip(Path(best["checkpoint"]))
        # policy-tensor identity under the explicit chosen scheme
        best["policy_tensor_sha256"] = best["policy_state_hash"]
        # the run's final checkpoint is part of the frozen record even when it is
        # not the selected stage, so the report can bind its final arm to it.
        final_path = Path(runs_by_key[key]) / "checkpoints" / "model_final.zip"
        if not final_path.exists():
            raise SystemExit(f"missing final checkpoint for {key}: {final_path}")
        best["final_checkpoint"] = str(final_path)
        best["final_policy_tensor_sha256"] = policy_state_hash_from_zip(final_path)
        best["selected_is_final"] = (
            Path(best["checkpoint"]).resolve() == final_path.resolve()
            and best["policy_tensor_sha256"] == best["final_policy_tensor_sha256"]
        )
        # full candidate table for auditability
        best["candidates"] = [
            {
                "arm": c["name"],
                "update": c["stage"],
                "mean_final_survival": arm_stats[c["name"]]["mean_final_survival"],
                "std": arm_stats[c["name"]]["std"],
            }
            for c in sorted(cands, key=lambda x: x["stage"])
        ]
        selected[key] = best

    manifest = {
        "kind": verify.MANIFEST_KIND,
        "version": verify.MANIFEST_VERSION,
        "hash_scheme": verify.HASH_SCHEME,
        # created_utc = when the selection mapping was first frozen. The v2 binding
        # fields below were enriched AFTER the first report run; they are a later
        # schema addition, not part of the original freeze instant.
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "bindings_enriched_utc": datetime.now(timezone.utc).isoformat(),
        "bindings_note": (
            "created_utc is the original selection freeze time. report_seeds, env "
            "config (incl. heading/speed bias) and the final-checkpoint identity "
            "fields are a v2 metadata enrichment added after the first report; they "
            "did not all exist at created_utc."
        ),
        "selection_seed_bank": common.STAGE_SEEDS["selection"],
        "selection_seeds": seeds,
        "report_seed_bank": common.STAGE_SEEDS["report"],
        "report_seeds": common.resolve_seeds("report"),
        "env": {
            "neighbor": False,
            "env_config": common.env_config(False),
        },
        "rule": (
            "For each (replicate, arm) pick the stage with the highest mean final "
            "survival over the frozen selection bank; ties broken toward the "
            "earliest update. Computed and frozen before any report evaluation."
        ),
        "statistic": "mean final_survival over selection episodes",
        "selected": selected,
        "selection_summary_sha256": hashlib.sha256(
            Path(args.out_jsonl).with_suffix(".summary.json").read_bytes()
        ).hexdigest(),
        "selected_only": {k: v["selected_arm"] for k, v in selected.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[written] {args.out_jsonl}")
    print(f"[written] {args.out_jsonl.replace('.jsonl', '.summary.json')}")
    print(f"[written] {manifest_path}")
    print(json.dumps({k: (v["selected_arm"], round(v["selection_mean_final_survival"], 6))
                      for k, v in selected.items()}, indent=2))


if __name__ == "__main__":
    main()
