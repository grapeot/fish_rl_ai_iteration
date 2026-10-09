#!/usr/bin/env python3
"""v57 analysis: the multi-speed balanced contrast for the training-time
initial-velocity randomization.

Primary question (pre-registered): does training with a per-episode velocity
factor drawn from {1.0,0.5,0.0} improve generalization ACROSS speed conditions
without losing the nominal condition?

Design of the contrast:
  * 6 selected models = 3 treatment (rep{r}_treatment_selected) and 3 control
    (rep{r}_control_selected) runs, each evaluated on 3 conditions x 40 report
    scenarios.
  * For each condition c in {nominal, half, zero}:
      treat_avg(c) = mean over the 3 treatment models' final_survival, per scenario;
      ctrl_avg(c)  = mean over the 3 control models' final_survival, per scenario;
      diff(c)      = treat_avg(c) - ctrl_avg(c), a per-scenario paired difference
                     (same scenario, same initial world).
    A paired episode bootstrap over the 40 scenarios gives the CI for diff(c).
  * overall = equal-weight mean over the 3 condition diffs (the balanced
    objective).
  * worst condition = the mean over the 40 scenarios of the PER-SCENARIO minimum
    of the three condition diffs (mean per-scenario minimum paired condition
    contrast). It is NOT the minimum of the three condition-average differences,
    NOT the worst single replicate, and NOT any arm's absolute minimum survival.
    It is a one-sided downward extreme-picking exploratory statistic.
  * The bootstrap resamples whole scenarios (each resample keeps all three
    conditions of a scenario together), matching the "same world" pairing.

Descriptive only: 3 treatment replicates is a spread, not a training-randomness
sampling CI. The episode-paired CI conditions on the six fixed models.

Also writes:
  * per-controller/per-condition means (from the summary);
  * rule anchors per condition;
  * the treatment augmentation factor coverage (from each treatment run config,
    the observed factor counts are recorded for coverage verification);
  * a cheap consistency guard: the summary means must equal a fresh raw
    recomputation.

Usage:
  experiments/v48/.venv/bin/python experiments/v57/v57_analyze.py \
      --report-summary experiments/v57/artifacts/results/report.summary.json \
      --report-jsonl experiments/v57/artifacts/results/report.jsonl \
      --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json \
      --out experiments/v57/artifacts/results/analysis.json
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

import v57_common as common  # noqa: E402
import v57_verify as verify  # noqa: E402


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=570301):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def check_summary_matches_raw(summary, grid, by_arm, conditions):
    """Guard: per-controller/per-condition summary means equal a fresh raw
    recomputation. Return (ctrl, cond, summary_mean, raw_mean) on mismatch."""
    for ctrl, block in summary["controllers"].items():
        for cond in conditions:
            stats = block.get(cond)
            if stats is None:
                continue
            vals = [cell[cond] for (c, _s), cell in grid.items() if c == ctrl]
            if not vals:
                continue
            raw_mean = float(np.mean(vals))
            if abs(raw_mean - float(stats["mean_final_survival"])) > 1e-9:
                return (ctrl, cond, float(stats["mean_final_survival"]), raw_mean)
    return None


def condition_diffs_by_scenario(grid, treat_names, ctrl_names, conditions):
    """Per-scenario, per-condition paired diff (treat model-avg - ctrl model-avg).

    Returns {cond: [diff per sorted scenario]} using the SAME scenario ordering
    for every condition, so a bootstrap resampling scenarios keeps all three
    conditions of a scenario together.
    """
    scenarios = sorted({s for (_c, s) in grid})
    out = {}
    for cond in conditions:
        diffs = []
        for s in scenarios:
            t = []
            for name in treat_names:
                cell = grid.get((name, s))
                if cell is None:
                    raise SystemExit(f"missing scenario {s} for {name}")
                t.append(cell[cond])
            c = []
            for name in ctrl_names:
                cell = grid.get((name, s))
                if cell is None:
                    raise SystemExit(f"missing scenario {s} for {name}")
                c.append(cell[cond])
            diffs.append(float(np.mean(t)) - float(np.mean(c)))
        out[cond] = np.asarray(diffs, dtype=float)
    return scenarios, out


def block_from_scenario_diffs(cond_diffs, conditions, n_boot=10000, alpha=0.05, seed=570301):
    """Condition CIs plus the balanced overall / worst-condition CIs.

    A single set of scenario resamples (seed 570301) is used for every statistic,
    so the overall/worst CIs are computed from the correctly paired scenario
    groups, not by combining independent condition CIs.
    """
    n = len(next(iter(cond_diffs.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    def ci_for(stat):
        # stat: (n_scenarios,) array
        samples = stat[idx].mean(axis=1)
        return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))

    per_cond = {}
    for cond in conditions:
        d = cond_diffs[cond]
        lo, hi = ci_for(d)
        per_cond[cond] = {
            "mean_diff": float(d.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "win_rate": float((d > 0).mean()),
            "loss_rate": float((d < 0).mean()),
        }

    # balanced overall = equal-weight mean of the 3 condition diffs per scenario
    stack = np.vstack([cond_diffs[c] for c in conditions])   # (3, n)
    overall = stack.mean(axis=0)
    overall_lo, overall_hi = ci_for(overall)
    worst = stack.min(axis=0)
    worst_lo, worst_hi = ci_for(worst)

    return {
        "per_condition": per_cond,
        "overall_equal_weight": {
            "mean_diff": float(overall.mean()),
            "ci95_low": overall_lo,
            "ci95_high": overall_hi,
            "win_rate": float((overall > 0).mean()),
            "note": "per-scenario equal-weight mean of the 3 condition diffs, then episode bootstrap",
        },
        "worst_condition": {
            "mean_diff": float(worst.mean()),
            "ci95_low": worst_lo,
            "ci95_high": worst_hi,
            "note": ("mean per-scenario minimum paired condition contrast: for each "
                     "scenario take the min of the 3 condition diffs, then mean over "
                     "scenarios; a one-sided downward extreme-picking exploratory "
                     "statistic, NOT the min of the 3 condition means, NOT the worst "
                     "replicate, NOT any absolute minimum survival"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    conditions = list(manifest["conditions"])
    treat_names = [f"rep{r}_treatment_selected" for r in common.REPLICATES]
    ctrl_names = [f"rep{r}_control_selected" for r in common.REPLICATES]

    out = {
        "conditions": conditions,
        "velocity_factors": manifest["velocity_factors"],
        "treatment_names": treat_names,
        "control_names": ctrl_names,
        "per_controller": {},
        "per_arm_condition": {},
        "rule_anchors": {},
        "treatment_factor_coverage": {},
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "role": manifest.get("role"),
        },
        "caveats": [
            "Only 3 paired replicates per arm: the three-model spread is descriptive, "
            "not a sampling CI over training randomness.",
            "The episode-paired CI conditions on the six fixed selected models.",
            "Statistical unit is the episode (scenario), never the 96 fish; each "
            "scenario is one initial world with 96 shared-policy fish.",
            "Every condition diff is paired (same scenario/world) between the 3-model "
            "treatment average and the 3-model control average.",
            "The overall CI resamples scenarios (all 3 conditions of a scenario kept "
            "together); it is NOT a combination of independent condition CIs.",
            "The selection objective is multi-speed balanced; its numbers are not "
            "directly comparable to the v50 nominal-only selector.",
        ],
    }

    records = load_jsonl(Path(args.report_jsonl))
    if not records:
        raise SystemExit("report jsonl is empty or missing")
    try:
        grid = verify.validate_paired_records(records, manifest, conditions=conditions)
    except verify.ManifestError as exc:
        raise SystemExit(f"refusing to analyze: records do not match the frozen manifest: {exc}")

    summary = json.loads(Path(args.report_summary).read_text())

    mismatch = check_summary_matches_raw(summary, grid, None, conditions)
    if mismatch is not None:
        ctrl, cond, s_mean, r_mean = mismatch
        raise SystemExit(
            f"refusing to analyze: summary mean for ({ctrl}, {cond}) = {s_mean} "
            f"!= raw recomputation {r_mean}")

    for ctrl, block in summary["controllers"].items():
        out["per_controller"][ctrl] = block

    # per-arm / per-condition means (arm average over the 3 models, per condition)
    scenarios = sorted({s for (_c, s) in grid})
    for arm, names in (("treatment", treat_names), ("control", ctrl_names)):
        block = {}
        for cond in conditions:
            vals = [grid[(n, s)][cond] for n in names for s in scenarios]
            block[cond] = {
                "n_models": len(names),
                "n_scenarios": len(scenarios),
                "mean_final_survival": float(np.mean(vals)),
            }
        out["per_arm_condition"][arm] = block

    for name in common.RULE_CONTROLLERS:
        if name in summary["controllers"]:
            out["rule_anchors"][name] = summary["controllers"][name]

    # treatment training-time factor coverage (from each treatment run config)
    for r in common.REPLICATES:
        cfg_path = ROOT / common.TREATMENT_RUN_DIR[r] / "config.json"
        summ_path = ROOT / common.TREATMENT_RUN_DIR[r] / "run_summary.json"
        entry = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            entry["augmentation"] = cfg.get("augmentation")
            entry["initial_policy_hash"] = cfg.get("initial_policy_hash")
            entry["model_seed"] = cfg.get("model_seed")
            entry["env_bank_worker_seeds"] = cfg.get("env_bank_worker_seeds")
        if summ_path.exists():
            summ = json.loads(summ_path.read_text())
            entry["augment_factor_counts"] = summ.get("augment_factor_counts")
            entry["augment_resets_total"] = summ.get("augment_resets_total")
        out["treatment_factor_coverage"][f"rep{r}"] = entry

    # control initial-policy hashes (paired-initial-weights check)
    for r in common.REPLICATES:
        cfg_path = ROOT / common.CONTROL_RUN_DIR[r] / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            out["treatment_factor_coverage"][f"rep{r}"]["control_initial_policy_hash"] = (
                cfg.get("initial_policy_hash")
            )

    # primary balanced contrast
    _scen, cond_diffs = condition_diffs_by_scenario(grid, treat_names, ctrl_names, conditions)
    out["balanced_contrast"] = block_from_scenario_diffs(cond_diffs, conditions)
    out["balanced_contrast"]["contrast"] = "3-model treatment avg - 3-model control avg, per condition"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")
    print(json.dumps({
        "per_condition": {k: round(v["mean_diff"], 6)
                          for k, v in out["balanced_contrast"]["per_condition"].items()},
        "overall": round(out["balanced_contrast"]["overall_equal_weight"]["mean_diff"], 6),
        "worst": round(out["balanced_contrast"]["worst_condition"]["mean_diff"], 6),
    }, indent=2))


if __name__ == "__main__":
    main()
