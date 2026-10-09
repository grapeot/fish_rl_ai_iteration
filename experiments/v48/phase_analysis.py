#!/usr/bin/env python3
"""Phase analysis: is death loss front-loaded or spread across the episode?
Reads an existing evaluate.py per-episode JSONL (which carries survival_at +
predator_second_half) and computes per-phase survival and per-phase loss from
the sampled survival values. No new episodes are run.

The phase losses are inferred from the survival sampled at steps 1/100/250/500
and averaged over episodes, so they are aggregate sample statistics, not
per-trajectory claims.

Usage:
  experiments/v48/.venv/bin/python experiments/v48/phase_analysis.py \
      --per-episode experiments/v48/artifacts/results/phase_report2.jsonl \
      --out experiments/v48/artifacts/results/phase_analysis.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PHASE_EDGES = [(1, 100), (101, 250), (251, 500)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-episode", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.per_episode).read_text().splitlines()]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["control"]].append(r)

    out = {}
    for arm, recs in by_arm.items():
        phased = {str(ph): [] for _, ph in [(0, 100), (0, 250), (0, 500)]}
        p100, p250, p500 = [], [], []
        for r in recs:
            s = r.get("survival_at", {})
            if "100" in s:
                p100.append(s["100"])
            if "250" in s:
                p250.append(s["250"])
            if "500" in s:
                p500.append(s["500"])
        # deaths attributable to each phase, inferred from phased survival:
        # losses in (1,100] ~ 1 - surv@100; (101,250] ~ surv@100 - surv@250;
        # (251,500] ~ surv@250 - surv@500. Averaged over episodes reaching 500.
        loss100 = [1.0 - s for s in p100]
        loss250 = [p100[i] - p250[i] for i in range(min(len(p100), len(p250)))]
        loss500 = [p250[i] - p500[i] for i in range(min(len(p250), len(p500)))]
        out[arm] = {
            "n": len(recs),
            "mean_survival_at": {
                "100": float(np.mean(p100)) if p100 else None,
                "250": float(np.mean(p250)) if p250 else None,
                "500": float(np.mean(p500)) if p500 else None,
            },
            "mean_loss_per_phase": {
                "1-100": float(np.mean(loss100)) if loss100 else None,
                "101-250": float(np.mean(loss250)) if loss250 else None,
                "251-500": float(np.mean(loss500)) if loss500 else None,
            },
            "late_loss_share_251_500": (
                float(np.mean(loss500) / (np.mean(loss100) + np.mean(loss250) + np.mean(loss500)))
                if loss500 and (np.mean(loss100) + np.mean(loss250) + np.mean(loss500)) > 0 else None
            ),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
