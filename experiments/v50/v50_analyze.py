#!/usr/bin/env python3
"""v50 analysis: training curves + the primary reward-bundle contrast.

Reads each run's `train_metrics.jsonl` (focal outcome stream) and the report
summary/jsonl, and writes one JSON with:

  * per-run focal survival curve (truncs/episodes) at every logged update, with
    the final update included;
  * per-arm report means/stds (selected and final stage, rule controls);
  * the three-run mean/std for each arm;
  * the primary `survival_only - original` episode-paired contrast at fixed
    six models (three replicate pairs), bootstrapped over report episodes.

Randomness caveat baked into the output: with only three replicates the
three-run std is a descriptive spread, not an independent-sample CI over
training randomness; the episode-paired CI conditions on these six fixed models.

Usage:
  experiments/v48/.venv/bin/python experiments/v50/v50_analyze.py \
      --runs-dir experiments/v50/artifacts/corrected_streams/runs \
      --report-summary experiments/v50/artifacts/corrected_streams/results/report.summary.json \
      --report-jsonl experiments/v50/artifacts/corrected_streams/results/report.jsonl \
      --manifest experiments/v50/artifacts/corrected_streams/results/selection_manifest.json \
      --out experiments/v50/artifacts/corrected_streams/results/analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v50_common as common  # noqa: E402
import v50_verify as verify  # noqa: E402


def load_metrics(run_dir: Path):
    path = run_dir / "train_metrics.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_key(run_dir: Path):
    stem = run_dir.name
    rep = int(stem.split("_")[0][3:])
    arm = stem.split("_", 1)[1]
    return rep, arm


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=500301):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    return (float(np.quantile(diffs[idx].mean(axis=1), alpha / 2)),
            float(np.quantile(diffs[idx].mean(axis=1), 1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True,
                    help="frozen selection manifest; binding is checked strictly")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = sorted([d for d in runs_dir.glob("rep*") if d.is_dir()])

    manifest = verify.load_manifest(args.manifest)
    expected_n = len(manifest["report_seeds"])
    out = {
        "runs": {},
        "report_arms": None,
        "three_run": {},
        "primary_contrast": {},
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "version": manifest.get("version"),
            "hash_scheme": manifest.get("hash_scheme"),
            "selection_seed_bank": manifest.get("selection_seed_bank"),
            "report_seed_bank": manifest.get("report_seed_bank"),
        },
        "caveats": [
            "Only 3 training replicates: the three-run std is a descriptive spread, "
            "not a sampling CI over training randomness.",
            "The episode-paired CI conditions on the six fixed models (3 replicates x 2 arms).",
            "Selection used the 24-episode selection bank 500201; report uses the disjoint "
            "40-episode bank 500202.",
        ],
    }

    curves = {}
    for run_dir in runs:
        rep, arm = run_key(run_dir)
        rows = load_metrics(run_dir)
        curve = []
        for r in rows:
            ep = r.get("episodes", 0)
            tr = r.get("truncs", 0)
            curve.append({
                "iteration": r["iteration"],
                "wall_sec": r.get("wall_sec"),
                "focal_survival": (tr / ep) if ep else None,
                "episodes": ep,
                "deaths": r.get("deaths"),
                "truncs": tr,
                "entropy_loss": r.get("train/entropy_loss"),
                "value_loss": r.get("train/value_loss"),
                "approx_kl": r.get("train/approx_kl"),
                "explained_variance": r.get("train/explained_variance"),
            })
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        summ = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").exists() else {}
        key = f"rep{rep}_{arm}"
        curves[key] = curve
        out["runs"][key] = {
            "replicate": rep,
            "arm": arm,
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

    # strict per-arm seed-set validation + manifest-bound selected/final resolution
    records = load_jsonl(Path(args.report_jsonl))
    if not records:
        raise SystemExit("report jsonl is empty or missing")
    try:
        resolved = verify.resolve_report_arms(records, manifest,
                                               expected_n=len(manifest["report_seeds"]))
    except verify.ManifestError as exc:
        raise SystemExit(f"refusing to analyze: report does not match the frozen manifest: {exc}")
    by_arm = resolved["_by_arm"]

    # three-run mean/std for selected and final per arm, straight from validated records
    for stage in ("selected", "final"):
        per_arm = {a: [] for a in ("original", "survival_only")}
        per_arm_rep = {a: {} for a in ("original", "survival_only")}
        for rep in range(len(common.REPLICATES)):
            for arm in common.ARMS:
                name = f"rep{rep}_{arm}_{stage}"
                if name not in resolved:
                    continue
                per_seed = resolved[name]
                v = float(np.mean(list(per_seed.values())))
                per_arm[arm].append(v)
                per_arm_rep[arm][rep] = v
        block = {}
        for arm, vals in per_arm.items():
            block[arm] = {
                "per_replicate": per_arm_rep[arm],
                "three_run_mean": float(np.mean(vals)) if vals else None,
                "three_run_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_replicates": len(vals),
            }
        out["three_run"][stage] = block

    # primary episode-paired contrast survival_only - original at fixed models
    for stage in ("selected", "final"):
        diffs = []
        per_rep = {}
        for rep in range(len(common.REPLICATES)):
            a_name = f"rep{rep}_survival_only_{stage}"
            b_name = f"rep{rep}_original_{stage}"
            if a_name not in resolved or b_name not in resolved:
                raise SystemExit(f"missing validated arm pair for stage={stage} rep={rep}")
            a_seeds, b_seeds = resolved[a_name], resolved[b_name]
            if set(a_seeds) != set(b_seeds):
                raise SystemExit(f"seed sets differ between {a_name} and {b_name}")
            seeds_sorted = sorted(a_seeds)
            d = [a_seeds[s] - b_seeds[s] for s in seeds_sorted]
            per_rep[rep] = {"mean_diff": float(np.mean(d)), "n": len(d)}
            diffs.append(d)
        if diffs and all(len(d) == len(diffs[0]) for d in diffs):
            stacked = np.array(diffs, dtype=float)  # (reps, episodes)
            episode_mean_diff = stacked.mean(axis=0)  # average over reps per episode
            lo, hi = bootstrap_ci(episode_mean_diff)
            out["primary_contrast"][stage] = {
                "contrast": "survival_only - original",
                "per_replicate": per_rep,
                "fixed_models_episode_paired_mean_diff": float(episode_mean_diff.mean()),
                "episode_paired_ci95": [lo, hi],
                "n_episodes": len(episode_mean_diff),
                "win_rate": float((episode_mean_diff > 0).mean()),
                "note": "mean over the 3 replicate pairs per episode, then bootstrap over episodes",
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
