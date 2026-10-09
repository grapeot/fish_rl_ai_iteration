#!/usr/bin/env python3
"""v54 analysis: the one-axis post-reset initial-velocity generalization contrast.

Writes two files:
  * `analysis.json` (via `--out`): per-controller/per-condition means/std, phase
    survival, action distribution and early-failure counts; per-PPO-replicate
    means; the paired condition contrasts vs each controller's own nominal; the
    aggregate PPO contrast; rule anchors. Unchanged in this pass.
  * `relative_hold_analysis.json` (via `--relative-hold-out`): a baseline-relative
    view computed DIRECTLY FROM THE RAW RECORDS (not copied from the summary).
    Per scenario, average the fixed three PPO policies, subtract the paired HOLD
    survival, and report the nominal and zero advantages plus their difference
    (DiD), each as a paired episode-bootstrap CI over the 40 scenarios.

The raw records are also used for a cheap consistency guard: the summary's
per-controller/per-condition means must match a fresh recomputation from the raw
jsonl, otherwise analysis aborts.

Caveats baked in: episode (scenario) is the statistical unit, never the 96 fish;
each PPO policy is a fixed model, not a random draw; the DiD is a paired outcome
contrast, NOT a causal mediation decomposition (the passive HOLD baseline's own
task difficulty changes with the intervention). It must not be replaced by
subtracting two independent CIs.

Usage:
  experiments/v48/.venv/bin/python experiments/v54/v54_analyze.py \
      --report-summary experiments/v54/artifacts/results/report.summary.json \
      --report-jsonl experiments/v54/artifacts/results/report.jsonl \
      --manifest experiments/v54/artifacts/frozen_config/v54_frozen_manifest.json \
      --out experiments/v54/artifacts/results/analysis.json \
      --relative-hold-out experiments/v54/artifacts/results/relative_hold_analysis.json
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

import v54_common as common  # noqa: E402
import v54_verify as verify  # noqa: E402

PHASE_STEPS = (1, 100, 250, 500)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=540301):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_contrast(grid, controller, cond, base="nominal"):
    """Per-scenario diffs (cond - base) for one controller, sorted by scenario."""
    diffs, seeds = [], []
    for (ctrl, seed), cell in grid.items():
        if ctrl != controller:
            continue
        diffs.append(cell[cond] - cell[base])
        seeds.append(seed)
    order = np.argsort(seeds)
    return [diffs[i] for i in order], [seeds[i] for i in order]


def check_summary_matches_raw(summary, grid):
    """Cheap consistency guard: the summary per-controller means must equal a
    fresh recomputation from the raw records. Abort the analysis otherwise."""
    for ctrl, block in summary["controllers"].items():
        for cond, stats in block.items():
            vals = [cell[cond] for (c, _s), cell in grid.items() if c == ctrl]
            if not vals:
                continue
            raw_mean = float(np.mean(vals))
            if abs(raw_mean - float(stats["mean_final_survival"])) > 1e-9:
                return (ctrl, cond, float(stats["mean_final_survival"]), raw_mean)
    return None


def relative_hold_contrasts(grid, n_boot=10000, alpha=0.05, boot_seed=540301):
    """Raw-derived PPO-vs-HOLD contrast.

    Per scenario: A(c) = mean over the fixed three PPO policies of survival
    minus the paired HOLD survival. Reports the nominal and zero advantage and
    their difference D = A(zero) - A(nominal), each as a paired episode bootstrap
    CI over scenarios. Unit is the scenario (never the fish or policy-scenario).
    """
    scenarios = sorted({s for (_c, s) in grid})

    def ppo_avg(cond, s):
        per_policy = []
        for name in common.PPO_CONTROLLER_NAMES:
            cell = grid.get((name, s))
            if cell is None:
                raise SystemExit(f"missing scenario {s} for {name}")
            per_policy.append(cell[cond])
        return float(np.mean(per_policy))

    def hold(cond, s):
        cell = grid.get(("rule_hold", s))
        if cell is None:
            raise SystemExit(f"missing HOLD scenario {s}")
        return cell[cond]

    adv_nom = np.array([ppo_avg("nominal", s) - hold("nominal", s) for s in scenarios])
    adv_zero = np.array([ppo_avg("zero", s) - hold("zero", s) for s in scenarios])
    did = adv_zero - adv_nom

    def block(x, label):
        lo, hi = bootstrap_ci(x, n_boot=n_boot, alpha=alpha, seed=boot_seed)
        return {
            "contrast": label,
            "mean": float(x.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "pos_scenarios": int((x > 0).sum()),
            "zero_scenarios": int((x == 0).sum()),
            "neg_scenarios": int((x < 0).sum()),
        }

    return {
        "unit": "scenario (40); each scenario averages the fixed 3 PPO policies, then pairs against HOLD",
        "nominal_ppo_minus_hold": block(adv_nom, "nominal: fixed-3 PPO average - HOLD"),
        "zero_ppo_minus_hold": block(adv_zero, "zero: fixed-3 PPO average - HOLD"),
        "difference_in_differences": block(did, "DiD: zero advantage - nominal advantage"),
        "note": ("Paired outcome contrast, NOT a causal mediation decomposition: HOLD's own "
                 "task difficulty changes with the intervention. Do not subtract two "
                 "independent CIs. Same sorted report seeds, numpy RNG 540301, 10,000 "
                 "resamples, percentile 95% CI."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relative-hold-out", default=None,
                    help="optional path for the raw-derived PPO-vs-HOLD contrast JSON")
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    conditions = list(manifest["conditions"])

    out = {
        "conditions": conditions,
        "velocity_factors": manifest["velocity_factors"],
        "per_controller": {},
        "per_ppo_replicate": {},
        "paired_vs_nominal": {},
        "aggregate_ppo_contrast": {},
        "rule_anchors": {},
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "intervention": manifest.get("intervention"),
        },
        "caveats": [
            "Statistical unit is the episode (scenario), never the 96 fish.",
            "Every condition diff is paired to the SAME controller's nominal on the SAME initial world.",
            "The three PPO policies are fixed models, not a random draw; their spread is descriptive.",
            "The aggregate contrast averages the 3 PPO policies per scenario, then bootstraps over scenarios.",
            "No episode is dropped; an early all-death scenario stays in with a fixed /96 denominator.",
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

    # cheap consistency guard: summary means must equal a fresh raw recomputation
    mismatch = check_summary_matches_raw(summary, grid)
    if mismatch is not None:
        ctrl, cond, s_mean, r_mean = mismatch
        raise SystemExit(
            f"refusing to analyze: summary mean for ({ctrl}, {cond}) = {s_mean} "
            f"!= raw recomputation {r_mean}")

    # per-controller, per-condition means/std from the validated summary
    for ctrl, block in summary["controllers"].items():
        out["per_controller"][ctrl] = block

    # per-PPO-replicate means (each policy is one replicate)
    for name in common.PPO_CONTROLLER_NAMES:
        rep_block = {}
        for cond in conditions:
            vals = [cell[cond] for (ctrl, _s), cell in grid.items() if ctrl == name]
            rep_block[cond] = {
                "n": len(vals),
                "mean_final_survival": float(np.mean(vals)) if vals else None,
            }
        out["per_ppo_replicate"][name] = rep_block

    # paired contrasts vs own nominal, per controller
    for ctrl in sorted({c for (c, _s) in grid}):
        out["paired_vs_nominal"][ctrl] = {}
        for cond in conditions:
            if cond == "nominal":
                continue
            diffs, seeds = paired_contrast(grid, ctrl, cond, "nominal")
            lo, hi = bootstrap_ci(diffs)
            d = np.array(diffs, dtype=float)
            out["paired_vs_nominal"][ctrl][cond] = {
                "contrast": f"{cond} - nominal",
                "n_pairs": len(diffs),
                "mean_diff": float(d.mean()) if len(d) else None,
                "std_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "win_rate": float((d > 0).mean()) if len(d) else None,
                "tie_rate": float((d == 0).mean()) if len(d) else None,
                "loss_rate": float((d < 0).mean()) if len(d) else None,
            }

    # aggregate over the fixed three PPO policies, per scenario, then episode bootstrap
    report_seeds = sorted({s for (_c, s) in grid})
    for cond in conditions:
        if cond == "nominal":
            continue
        diffs = []
        for seed in report_seeds:
            per_policy = []
            for name in common.PPO_CONTROLLER_NAMES:
                cell = grid.get((name, seed))
                if cell is None:
                    raise SystemExit(f"missing scenario {seed} for {name}")
                per_policy.append(cell[cond] - cell["nominal"])
            diffs.append(float(np.mean(per_policy)))
        lo, hi = bootstrap_ci(diffs)
        d = np.array(diffs, dtype=float)
        out["aggregate_ppo_contrast"][cond] = {
            "contrast": f"{cond} - nominal (mean over 3 frozen PPO policies per scenario)",
            "n_scenarios": len(diffs),
            "mean_diff": float(d.mean()) if len(d) else None,
            "ci95_low": lo,
            "ci95_high": hi,
            "win_rate": float((d > 0).mean()) if len(d) else None,
            "loss_rate": float((d < 0).mean()) if len(d) else None,
            "note": "per scenario average over the 3 fixed PPO policies, then bootstrap over scenarios",
        }

    # rule anchors (reference)
    for name in common.RULE_CONTROLLERS:
        if name in summary["controllers"]:
            out["rule_anchors"][name] = summary["controllers"][name]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")

    if args.relative_hold_out:
        rel = {
            "source": "computed directly from raw report records (not copied from the summary)",
            "manifest": {"path": str(args.manifest), "created_utc": manifest.get("created_utc")},
            "caveats": [
                "Statistical unit is the scenario (40), never the 96 fish or the policy-scenario.",
                "Per scenario the fixed 3 PPO policies are averaged, then paired against HOLD.",
                "DiD is a paired outcome contrast, NOT a causal mediation decomposition.",
                "Do not replace the DiD CI by subtracting two independent CIs.",
            ],
            "relative_hold": relative_hold_contrasts(grid),
        }
        Path(args.relative_hold_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.relative_hold_out).write_text(json.dumps(rel, indent=2))
        print(f"[written] {args.relative_hold_out}")


if __name__ == "__main__":
    main()
