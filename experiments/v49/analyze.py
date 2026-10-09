#!/usr/bin/env python3
"""v49 learning-curve + evaluation summary across the 3 training runs.

Reads each run's train_metrics.jsonl (focal-episode outcome stream) and the
eval summaries, and writes one JSON with:
  * per-run cumulative focal survival (trunc/(trunc+death)) at each logged
    iteration (a running statistic over the non-stationary training stream, not
    an independent test curve),
  * eval mean final survival per arm (untrained, staged checkpoints, final,
    rule controls),
  * paired diffs vs the reference rule already computed by evaluate.py.

This only aggregates what the training and eval runs already produced; it does
not re-tune on the report set.

Usage:
  experiments/v48/.venv/bin/python experiments/v49/analyze.py \
      --runs-dir experiments/v49/artifacts/runs \
      --eval-summary experiments/v49/artifacts/results/report.summary.json \
      --out experiments/v49/artifacts/results/analysis.json
"""

import argparse
import json
from pathlib import Path


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--eval-summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out = {"runs": {}, "eval": None}
    for run_dir in sorted(runs_dir.glob("seed*")):
        rows = load_metrics(run_dir)
        if not rows:
            continue
        curve = []
        for r in rows:
            ep = r.get("episodes_so_far", 0)
            tr = r.get("truncs_so_far", 0)
            curve.append({
                "iteration": r["iteration"],
                "wall_sec": r["wall_sec"],
                "focal_survival": (tr / ep) if ep else None,
                "deaths": r.get("deaths_so_far"),
                "episodes": ep,
                "explained_variance": r.get("train/explained_variance"),
                "approx_kl": r.get("train/approx_kl"),
                "entropy_loss": r.get("train/entropy_loss"),
                "value_loss": r.get("train/value_loss"),
            })
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        summ = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").exists() else {}
        out["runs"][run_dir.name] = {
            "seed": cfg.get("seed"),
            "config": cfg,
            "run_summary": summ,
            "curve": curve,
        }

    es = Path(args.eval_summary)
    if es.exists():
        out["eval"] = json.loads(es.read_text())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
