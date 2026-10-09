#!/usr/bin/env python3
"""v53 checkpoint selection on the fresh selection bank, then freeze a
non-overwritable manifest BEFORE any report data is produced.

For each run key in {rep{r}_control, rep{r}_treatment}:
  * evaluate every saved stage checkpoint (u50/u100/u150/u200) on 530101 (24 eps,
    paired seeds);
  * pick the stage with the highest mean final survival; ties -> EARLIEST update.

The control checkpoints are the REUSED accepted v50 corrected survival_only runs;
the treatment checkpoints are the new v53 finite_terminal runs. The manifest
records per run: the selected checkpoint path + policy-tensor hash, the run's
final checkpoint path + final hash, `selected_is_final` (path AND hash), and the
full candidate table. It binds the COMPLETE env config (heading bias included)
and both frozen seed banks, and is written only if absent.

Usage:
  experiments/v48/.venv/bin/python experiments/v53/v53_select.py \
      --out-jsonl experiments/v53/artifacts/results/selection.jsonl \
      --manifest experiments/v53/artifacts/results/selection_manifest.json
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

import v53_common as common  # noqa: E402
import v53_verify as verify  # noqa: E402

STAGES = [50, 100, 150, 200]


def _rel(p) -> str:
    """Repo-relative path (public-safe; no absolute home path in any artifact)."""
    return str(Path(p).resolve().relative_to(ROOT.resolve()))


def _abs(p) -> Path:
    """Repo-relative path -> absolute, resolved against the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp)


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


def run_dirs():
    """Ordered mapping run_key -> checkpoint dir."""
    out = {}
    for r in common.REPLICATES:
        out[f"rep{r}_control"] = ROOT / common.CONTROL_RUN_DIR[r]
        out[f"rep{r}_treatment"] = ROOT / common.TREATMENT_RUN_DIR[r]
    for key, d in out.items():
        if not d.is_dir():
            raise SystemExit(f"missing run dir for {key}: {d}")
    return out


def main():
    ap = argparse.ArgumentParser()
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

    created_utc = None
    if manifest_path.exists():
        try:
            created_utc = json.loads(manifest_path.read_text()).get("created_utc")
        except Exception:
            created_utc = None

    runs = run_dirs()

    arms = []
    candidates = {}
    for key, run_dir in runs.items():
        ckpts = checkpoint_paths(run_dir)
        if not ckpts:
            raise SystemExit(f"no checkpoints under {run_dir}")
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
            sys.executable, str(HERE / "v53_evaluate.py"),
            "--seeds", "selection", "--neighbor", "off",
            "--workers", str(args.workers), "--out", args.out_jsonl,
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
        best["policy_tensor_sha256"] = verify.policy_tensor_sha256_from_zip(_abs(best["checkpoint"]))
        best["policy_state_hash"] = best["policy_tensor_sha256"]
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
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "selection_seed_bank": common.STAGE_SEEDS["selection"],
        "selection_seeds": seeds,
        "report_seed_bank": common.STAGE_SEEDS["report"],
        "report_seeds": common.resolve_seeds("report"),
        "env": {
            "neighbor": False,
            "env_config": common.env_config(False),   # FULL config, heading bias included
        },
        "termination_arms": {
            "control": "timeout_bootstrap (reused v50 corrected survival_only)",
            "treatment": "finite_terminal (new v53 runs)",
        },
        "rule": (
            "For each run key (rep{r}_control, rep{r}_treatment) pick the stage with the "
            "highest mean final survival over the frozen selection bank 530101; ties "
            "broken toward the earliest update. Computed and frozen before any report."
        ),
        "statistic": "mean final_survival over selection episodes",
        "selected": selected,
        "selection_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "selected_only": {k: v["selected_arm"] for k, v in selected.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[written] {args.out_jsonl}")
    print(f"[written] {summary_path}")
    print(f"[written] {manifest_path}")
    print(json.dumps({k: (v["selected_arm"], round(v["selection_mean_final_survival"], 6))
                      for k, v in selected.items()}, indent=2))


if __name__ == "__main__":
    main()
