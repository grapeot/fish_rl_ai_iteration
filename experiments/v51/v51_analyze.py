#!/usr/bin/env python3
"""v51 analysis: deterministic vs stochastic action selection on frozen v49 models.

Reads the raw per-episode JSONL from v51_eval.py and computes, per model:

  * ``det`` mean final survival (40 scenarios),
  * ``stoch`` per-replicate means (raw action-stream spread),
  * the scenario-averaged stochastic value (mean of the 3 replicates), then
    ``stoch_avg - det`` paired per scenario with an episode bootstrap CI.

The average is taken over the 3 action streams WITHIN each scenario BEFORE the
paired inference (the spec's "先每场景对3actionstreams平均再做episode paired
bootstrap"). The raw per-replicate values are retained so the across-stream
spread stays inspectable. Only 3 fixed streams were run: their mean-difference
signs agree across the 40 scenarios (limited repeatability), but this does NOT
establish absence of lucky-trajectory influence nor stability over a broad range
of action RNGs.

Stage interaction: the three selection-stage models and the three final models
are summarized separately; the difference-in-differences (Δfinal - Δselection)
is reported as a DESCRIPTIVE contrast. With three training runs it is NOT a
training-randomness CI and cannot license a claim about training statistics.

Scope: this conditions on the six frozen models and on the fixed action-sampling
scheme. It contains no training uncertainty; the three runs are never resampled.

Usage:
  experiments/v48/.venv/bin/python experiments/v51/v51_analyze.py \
      --raw experiments/v51/artifacts/results/fresh.jsonl \
      --manifest experiments/v51/artifacts/results/model_manifest.json \
      --out experiments/v51/artifacts/results/analysis.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import v51_common as common  # noqa: E402


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=510302):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(diffs.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def load_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.raw)

    det = {}       # model -> {scenario_index: survival}
    stoch = {}     # model -> {replicate: {scenario_index: survival}}
    rules = {}     # rule -> {scenario_index: survival}
    for r in rows:
        m = r.get("model", "")
        si = r["scenario_index"]
        if r.get("arm") == "det":
            det.setdefault(m, {})[si] = r["final_survival"]
        elif r.get("arm") == "stoch":
            stoch.setdefault(m, {}).setdefault(r["replicate"], {})[si] = r["final_survival"]
        elif r.get("arm") == "rule":
            rules.setdefault(m, {})[si] = r["final_survival"]

    out = {"schema": "v51_analysis_v1",
           "raw": str(args.raw),
           "models": {},
           "stage_interaction": {},
           "rule_anchors": {}}

    model_summaries = {}
    for m in sorted(det):
        d = det[m]
        scen = sorted(d)
        stoch_reps = stoch.get(m, {})
        # scenario-averaged stochastic value across the fixed replicates
        rep_ids = sorted(stoch_reps)
        stoch_avg = {}
        for si in scen:
            vals = [stoch_reps[rep][si] for rep in rep_ids if si in stoch_reps[rep]]
            if vals:
                stoch_avg[si] = float(np.mean(vals))
        stoch_vals = {}
        for rep in rep_ids:
            v = [stoch_reps[rep][si] for si in sorted(stoch_reps[rep])]
            stoch_vals[str(rep)] = {
                "n": len(v), "mean": float(np.mean(v)) if v else None,
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            }
        diffs = [stoch_avg[si] - d[si] for si in scen if si in stoch_avg]
        mean_diff, lo, hi = bootstrap_ci(diffs)
        dvals = np.array([d[si] for si in scen], dtype=float)
        savals = np.array([stoch_avg[si] for si in scen if si in stoch_avg], dtype=float)
        adist_det = _mean_action_dist(rows, m, "det", None)
        adist_stoch = _mean_action_dist(rows, m, "stoch", None)
        # Across-stream spread: std over the 3 replicates, averaged over scenarios.
        per_scen_stream_vals = []
        for si in scen:
            vals = [stoch_reps[rep][si] for rep in rep_ids if si in stoch_reps[rep]]
            if len(vals) > 1:
                per_scen_stream_vals.append(float(np.std(vals, ddof=1)))
        across_stream_std = float(np.mean(per_scen_stream_vals)) if per_scen_stream_vals else None
        # Scenario-level paired outcome counts (tie = exact equality).
        dv = np.array(diffs, dtype=float)
        model_summaries[m] = {
            "n_scenarios": len(scen),
            "det": {"mean_final_survival": float(dvals.mean()),
                    "std": float(dvals.std(ddof=1)) if len(dvals) > 1 else 0.0},
            "stoch_avg": {"mean_final_survival": float(savals.mean()),
                          "std": float(savals.std(ddof=1)) if len(savals) > 1 else 0.0},
            "stoch_per_replicate": stoch_vals,
            "stoch_across_stream_std_mean": across_stream_std,
            "stoch_minus_det": {"mean_diff": mean_diff, "ci95_low": lo, "ci95_high": hi,
                                "n_pairs": len(diffs),
                                "scenarios_stoch_better": int((dv > 0).sum()),
                                "scenarios_tie": int((dv == 0).sum()),
                                "scenarios_det_better": int((dv < 0).sum()),
                                "win_rate": float(np.mean(dv > 0)) if diffs else None},
            "mean_action_dist_det": adist_det,
            "mean_action_dist_stoch": adist_stoch,
        }
    out["models"] = model_summaries

    # Stage interaction (descriptive): selection-stage vs final-stage models.
    sel = float(np.mean([model_summaries[m]["stoch_minus_det"]["mean_diff"]
                         for m in model_summaries if m.endswith("_sel")])) \
        if any(m.endswith("_sel") for m in model_summaries) else None
    fin = float(np.mean([model_summaries[m]["stoch_minus_det"]["mean_diff"]
                         for m in model_summaries if m.endswith("_final")])) \
        if any(m.endswith("_final") for m in model_summaries) else None
    out["stage_interaction"] = {
        "note": ("Δ = stoch_avg - det mean final survival, per model. The "
                 "selection/final phase means aggregate the three frozen runs; "
                 "with three runs this is a descriptive contrast, NOT a "
                 "training-randomness CI."),
        "selection_phase_mean_delta": sel,
        "final_phase_mean_delta": fin,
        "delta_of_delta_final_minus_selection": (fin - sel) if (sel is not None and fin is not None) else None,
        "per_model_delta": {m: model_summaries[m]["stoch_minus_det"]["mean_diff"]
                            for m in model_summaries},
    }

    for rule in sorted(rules):
        v = np.array(list(rules[rule].values()), dtype=float)
        out["rule_anchors"][rule] = {
            "n": len(v), "mean_final_survival": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "mean_action_dist": _mean_action_dist(rows, rule, "rule", None),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("stage_interaction",)}, indent=2))
    print(f"[written] {args.out}")


def _mean_action_dist(rows, model, arm, replicate):
    dists = []
    for r in rows:
        if r.get("model") != model or r.get("arm") != arm:
            continue
        if replicate is not None and r.get("replicate") != replicate:
            continue
        ad = r.get("action_dist")
        if ad:
            dists.append(ad)
    if not dists:
        return None
    return np.mean(np.array(dists, dtype=float), axis=0).tolist()


if __name__ == "__main__":
    main()
