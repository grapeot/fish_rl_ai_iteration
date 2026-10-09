#!/usr/bin/env python3
"""v53 analysis: training curves + the primary termination-semantics contrast.

Writes one JSON with:
  * per-run focal survival curves for the reused control (v50 corrected
    survival_only) and the new treatment (v53 finite_terminal) runs;
  * per-arm report means/stds for selected and final stages plus the rule anchors
    (rule_hold / rule_flee_lead / rule_safe_top);
  * the three-run mean/std per arm (descriptive spread, NOT a training-randomness
    sampling CI with only three replicates);
  * the primary `finite_terminal - timeout_bootstrap` episode-paired contrast at
    the fixed six models, bootstrapped over report episodes.

Usage:
  experiments/v48/.venv/bin/python experiments/v53/v53_analyze.py \
      --report-summary experiments/v53/artifacts/results/report.summary.json \
      --report-jsonl experiments/v53/artifacts/results/report.jsonl \
      --manifest experiments/v53/artifacts/results/selection_manifest.json \
      --out experiments/v53/artifacts/results/analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v53_common as common  # noqa: E402
import v53_verify as verify  # noqa: E402


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def focal_survival(row):
    ep = row.get("episodes", 0)
    if not ep:
        return None
    # control (v50) logs `truncs`; treatment (v53) logs `finite_terminals`.
    ended_alive = row.get("truncs", 0) + row.get("finite_terminals", 0)
    return ended_alive / ep


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=530301):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    return (float(np.quantile(diffs[idx].mean(axis=1), alpha / 2)),
            float(np.quantile(diffs[idx].mean(axis=1), 1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    expected_n = len(manifest["report_seeds"])

    out = {
        "runs": {},
        "report_arms": None,
        "three_run": {},
        "primary_contrast": {},
        "anchors": {},
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "selection_seed_bank": manifest.get("selection_seed_bank"),
            "report_seed_bank": manifest.get("report_seed_bank"),
        },
        "caveats": [
            "Only 3 paired replicates: the three-run std is a descriptive spread, "
            "not a sampling CI over training randomness.",
            "The episode-paired CI conditions on the six fixed models (3 control + 3 treatment).",
            "The control is the reused accepted v50 corrected survival_only runs (timeout_bootstrap).",
            "Selection used the 24-episode bank 530101; report uses the disjoint 40-episode bank 530102.",
        ],
    }

    # --- training curves ----------------------------------------------------
    curves = {}
    run_dirs = {}
    for r in common.REPLICATES:
        run_dirs[f"rep{r}_control"] = ROOT / common.CONTROL_RUN_DIR[r]
        run_dirs[f"rep{r}_treatment"] = ROOT / common.TREATMENT_RUN_DIR[r]
    for key, run_dir in run_dirs.items():
        rows = load_jsonl(run_dir / "train_metrics.jsonl")
        curve = []
        for row in rows:
            curve.append({
                "iteration": row["iteration"],
                "wall_sec": row.get("wall_sec"),
                "focal_survival": focal_survival(row),
                "episodes": row.get("episodes"),
                "deaths": row.get("deaths"),
                "truncs": row.get("truncs"),
                "finite_terminals": row.get("finite_terminals"),
                "entropy_loss": row.get("train/entropy_loss"),
                "value_loss": row.get("train/value_loss"),
                "approx_kl": row.get("train/approx_kl"),
                "explained_variance": row.get("train/explained_variance"),
            })
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        summ = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").exists() else {}
        curves[key] = curve
        out["runs"][key] = {
            "termination_mode": cfg.get("termination_mode", "timeout_bootstrap" if "control" in key else None),
            "model_seed": cfg.get("model_seed"),
            "env_bank_worker_seeds": cfg.get("env_bank_worker_seeds"),
            "initial_policy_hash": cfg.get("initial_policy_hash"),
            "train_metrics_rows": len(rows),
            "final_focal_survival": curve[-1]["focal_survival"] if curve else None,
            "run_summary": summ,
            "curve": curve,
        }

    es = Path(args.report_summary)
    summary = json.loads(es.read_text()) if es.exists() else None
    out["report_arms"] = summary

    records = load_jsonl(Path(args.report_jsonl))
    if not records:
        raise SystemExit("report jsonl is empty or missing")
    try:
        resolved = verify.resolve_report_arms(records, manifest, expected_n=expected_n)
    except verify.ManifestError as exc:
        raise SystemExit(f"refusing to analyze: report does not match the frozen manifest: {exc}")
    by_arm = resolved["_by_arm"]

    # --- three-run mean/std for selected and final per arm ------------------
    for stage in ("selected", "final"):
        per_arm = {a: [] for a in common.ARMS}
        per_arm_rep = {a: {} for a in common.ARMS}
        for r in common.REPLICATES:
            for arm in common.ARMS:
                name = f"rep{r}_{arm}_{stage}"
                if name not in resolved:
                    continue
                v = float(np.mean(list(resolved[name].values())))
                per_arm[arm].append(v)
                per_arm_rep[arm][r] = v
        block = {}
        for arm, vals in per_arm.items():
            block[arm] = {
                "per_replicate": per_arm_rep[arm],
                "three_run_mean": float(np.mean(vals)) if vals else None,
                "three_run_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_replicates": len(vals),
            }
        out["three_run"][stage] = block

    # --- primary episode-paired contrast treatment - control ----------------
    for stage in ("selected", "final"):
        diffs = []
        per_rep = {}
        for r in common.REPLICATES:
            a_name = f"rep{r}_treatment_{stage}"
            b_name = f"rep{r}_control_{stage}"
            if a_name not in resolved or b_name not in resolved:
                raise SystemExit(f"missing validated arm pair for stage={stage} rep={r}")
            a_seeds, b_seeds = resolved[a_name], resolved[b_name]
            if set(a_seeds) != set(b_seeds):
                raise SystemExit(f"seed sets differ between {a_name} and {b_name}")
            d = [a_seeds[s] - b_seeds[s] for s in sorted(a_seeds)]
            per_rep[r] = {"mean_diff": float(np.mean(d)), "n": len(d)}
            diffs.append(d)
        if diffs and all(len(d) == len(diffs[0]) for d in diffs):
            stacked = np.array(diffs, dtype=float)          # (reps, episodes)
            episode_mean_diff = stacked.mean(axis=0)         # average over reps per episode
            lo, hi = bootstrap_ci(episode_mean_diff)
            out["primary_contrast"][stage] = {
                "contrast": "finite_terminal - timeout_bootstrap",
                "per_replicate": per_rep,
                "fixed_models_episode_paired_mean_diff": float(episode_mean_diff.mean()),
                "episode_paired_ci95": [lo, hi],
                "n_episodes": len(episode_mean_diff),
                "win_rate": float((episode_mean_diff > 0).mean()),
                "note": "mean over the 3 replicate pairs per episode, then bootstrap over episodes",
            }

    # --- rule anchors from the report summary ------------------------------
    arms = (summary or {}).get("arms", {})
    for anchor in ("rule_hold", "rule_flee_lead", "rule_safe_top"):
        if anchor in arms:
            out["anchors"][anchor] = {
                "mean_final_survival": arms[anchor]["mean_final_survival"],
                "std": arms[anchor]["std"],
                "n": arms[anchor]["n"],
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
