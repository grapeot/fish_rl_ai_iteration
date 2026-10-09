#!/usr/bin/env python3
"""v56 analysis: the one-axis post-reset predator initial-speed generalization
contrast.

Reads the gated report jsonl/summary and writes one JSON with:
  * per-controller, per-condition final-survival mean/std, phase survival
    (step 1/100/250/500), action distribution, early-failure counts and the
    applied/nominal predator-speed descriptors (with the zero-norm count);
  * per-PPO-replicate means;
  * the paired condition contrasts, each relative to that SAME controller's own
    `scale1p00` (nominal) on the SAME scenario: (scale0p75 - base) and
    (scale1p25 - base);
  * the aggregate contrast: per scenario, average the paired difference over the
    fixed three PPO policies, then bootstrap over scenarios (episode unit). The
    three policies' initial worlds are not treated as extra samples;
  * a raw-derived asymmetry summary: the mean slowdown effect and speedup effect
    (and their difference) over the 40 scenarios, each as a paired episode-bootstrap
    CI, so both directions are reported together and neither is selected post hoc;
  * rule anchors for reference.

Caveats baked in: episode (scenario) is the statistical unit, never the 96 fish;
each PPO policy is a fixed model, not a random draw; the reported spread across three
policies is descriptive. The scale probes generalization across the post-reset
predator initial-speed magnitude, not the natural initial-speed distribution; the
`scale1p00` condition is the exact identity. Slower and faster are both reported; no
sign is assumed in advance.

Usage:
  experiments/v48/.venv/bin/python experiments/v56/v56_analyze.py \
      --report-summary experiments/v56/artifacts/results/report.summary.json \
      --report-jsonl experiments/v56/artifacts/results/report.jsonl \
      --manifest experiments/v56/artifacts/frozen_config/v56_frozen_manifest.json \
      --out experiments/v56/artifacts/results/analysis.json \
      --speed-view-out experiments/v56/artifacts/results/speed_robustness.json
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

import v56_common as common  # noqa: E402
import v56_verify as verify  # noqa: E402

PHASE_STEPS = (1, 100, 250, 500)
BASE_CONDITION = common.BASE_CONDITION
SLOW_CONDITIONS = ("scale0p75",)
FAST_CONDITIONS = ("scale1p25",)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=560301):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_contrast(grid, controller, cond, base=BASE_CONDITION):
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
    """Cheap consistency guard: the summary per-controller means must equal a fresh
    recomputation from the raw records. Abort the analysis otherwise."""
    for ctrl, block in summary["controllers"].items():
        for cond, stats in block.items():
            vals = [cell[cond] for (c, _s), cell in grid.items() if c == ctrl]
            if not vals:
                continue
            raw_mean = float(np.mean(vals))
            if abs(raw_mean - float(stats["mean_final_survival"])) > 1e-9:
                return (ctrl, cond, float(stats["mean_final_survival"]), raw_mean)
    return None


def relative_hold_contrasts(grid, n_boot=10000, alpha=0.05, boot_seed=560301):
    """Raw-derived PPO-vs-HOLD contrast for each condition.

    Per scenario: A(c) = mean over the fixed three PPO policies of survival minus the
    paired HOLD survival on the same seed/condition. Reports the mean advantage and
    its paired episode-bootstrap CI for each condition, plus the paired difference of
    advantages (fast 1.25x advantage - nominal 1.00x advantage), bootstrapped
    directly on the paired difference (never by subtracting interval endpoints). The
    unit is the scenario (never the fish or the policy-scenario). This is a
    reviewer-added descriptive analysis, not a prespecified primary endpoint; HOLD is a
    contextual anchor, not the strongest rule here.
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

    adv = {
        cond: np.array([ppo_avg(cond, s) - hold(cond, s) for s in scenarios])
        for cond in (SLOW_CONDITIONS[0], BASE_CONDITION, FAST_CONDITIONS[0])
    }
    slow_c, base_c, fast_c = SLOW_CONDITIONS[0], BASE_CONDITION, FAST_CONDITIONS[0]
    did = adv[fast_c] - adv[base_c]

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
        "ppo_minus_hold": {
            slow_c: block(adv[slow_c], f"PPO - HOLD at {slow_c}"),
            base_c: block(adv[base_c], f"PPO - HOLD at {base_c}"),
            fast_c: block(adv[fast_c], f"PPO - HOLD at {fast_c}"),
        },
        "advantage_fast_minus_nominal": block(
            did, f"advantage at {fast_c} - advantage at {base_c} (paired difference)"),
        "note": ("Descriptive raw-derived baseline contrasts, not prespecified primary "
                 "endpoints. HOLD is a contextual anchor, not the strongest rule here "
                 "(flee_lead exceeds PPO in all conditions). The advantage-change CI spans "
                 "both decrease and increase, so a standalone faster-condition PPO survival "
                 "rise does not establish a reliably larger relative advantage over HOLD. "
                 "Bootstrap the paired advantage difference directly; do not subtract two "
                 "independent CIs. Same sorted report seeds, numpy RNG 560301, 10,000 "
                 "resamples, percentile two-sided 95% CI."),
    }


def speed_robustness_contrasts(grid, n_boot=10000, alpha=0.05, boot_seed=560301):
    """Raw-derived slowdown / speedup asymmetry over the fixed three PPO policies.

    Per scenario: D(c) = mean over the 3 PPO of (survival(c) - survival(base)). The
    slowdown effect is D(scale0p75), the speedup effect is D(scale1p25), and their
    difference D(fast) - D(slow) is reported as a paired episode-bootstrap CI. The
    unit is the scenario. Both directions are reported; no sign is preselected.
    """
    scenarios = sorted({s for (_c, s) in grid})

    def ppo_avg_diff(cond, s):
        per_policy = []
        for name in common.PPO_CONTROLLER_NAMES:
            cell = grid.get((name, s))
            if cell is None:
                raise SystemExit(f"missing scenario {s} for {name}")
            per_policy.append(cell[cond] - cell[BASE_CONDITION])
        return float(np.mean(per_policy))

    slow = np.array([ppo_avg_diff(c, s) for s in scenarios for c in SLOW_CONDITIONS])
    fast = np.array([ppo_avg_diff(c, s) for s in scenarios for c in FAST_CONDITIONS])
    asym = fast - slow

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
        "unit": "scenario (40); each scenario averages the fixed 3 PPO policies, then pairs to scale1p00",
        "slowdown_scale0p75_minus_base": block(slow, "scale0p75 - scale1p00 (mean over 3 PPO)"),
        "speedup_scale1p25_minus_base": block(fast, "scale1p25 - scale1p00 (mean over 3 PPO)"),
        "asymmetry_fast_minus_slow": block(asym, "scale1p25 effect - scale0p75 effect"),
        "note": ("Paired outcome contrast, NOT a causal decomposition. Both directions are "
                 "reported; a positive asymmetry means the faster condition hurts survival "
                 "less (or helps more) than the slower one. Do not subtract independent CIs. "
                 "Same sorted report seeds, numpy RNG 560301, 10,000 resamples, percentile 95% CI."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed-view-out", default=None,
                    help="optional path for the raw-derived slowdown/speedup asymmetry JSON")
    ap.add_argument("--relative-hold-out", default=None,
                    help="optional path for the raw-derived PPO-vs-HOLD baseline contrast JSON")
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    conditions = list(manifest["conditions"])

    out = {
        "conditions": conditions,
        "speed_factors": manifest["speed_factors"],
        "base_condition": BASE_CONDITION,
        "per_controller": {},
        "per_ppo_replicate": {},
        "paired_vs_base": {},
        "aggregate_ppo_contrast": {},
        "rule_anchors": {},
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "intervention": manifest.get("intervention"),
        },
        "caveats": [
            "Statistical unit is the episode (scenario), never the 96 fish.",
            "Every condition diff is paired to the SAME controller's scale1p00 (nominal) on the SAME initial world.",
            "The three PPO policies are fixed models, not a random draw; their spread is descriptive.",
            "The aggregate contrast averages the 3 PPO policies per scenario, then bootstraps over scenarios.",
            "No episode is dropped; an early all-death scenario stays in with a fixed /96 denominator.",
            "The scales probe post-reset predator initial-speed generalization, not the natural initial-speed distribution.",
            "Both slower and faster directions are reported; no sign is assumed in advance.",
            "The applied predator speed is read from the state, not from an action or the nominal field.",
            "A step-one death caused by the scaled spawn is kept, not filtered.",
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

    # paired contrasts vs own base, per controller
    for ctrl in sorted({c for (c, _s) in grid}):
        out["paired_vs_base"][ctrl] = {}
        for cond in conditions:
            if cond == BASE_CONDITION:
                continue
            diffs, seeds = paired_contrast(grid, ctrl, cond, BASE_CONDITION)
            lo, hi = bootstrap_ci(diffs)
            d = np.array(diffs, dtype=float)
            out["paired_vs_base"][ctrl][cond] = {
                "contrast": f"{cond} - {BASE_CONDITION}",
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
        if cond == BASE_CONDITION:
            continue
        diffs = []
        for seed in report_seeds:
            per_policy = []
            for name in common.PPO_CONTROLLER_NAMES:
                cell = grid.get((name, seed))
                if cell is None:
                    raise SystemExit(f"missing scenario {seed} for {name}")
                per_policy.append(cell[cond] - cell[BASE_CONDITION])
            diffs.append(float(np.mean(per_policy)))
        lo, hi = bootstrap_ci(diffs)
        d = np.array(diffs, dtype=float)
        out["aggregate_ppo_contrast"][cond] = {
            "contrast": f"{cond} - {BASE_CONDITION} (mean over 3 frozen PPO policies per scenario)",
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

    if args.speed_view_out:
        view = {
            "source": "computed directly from raw report records (not copied from the summary)",
            "manifest": {"path": str(args.manifest), "created_utc": manifest.get("created_utc")},
            "caveats": [
                "Statistical unit is the scenario (40), never the 96 fish or the policy-scenario.",
                "Per scenario the fixed 3 PPO policies are averaged, then paired to scale1p00.",
                "Both slower and faster directions are reported; no sign is preselected.",
                "Do not replace a difference CI by subtracting two independent CIs.",
            ],
            "speed_robustness": speed_robustness_contrasts(grid),
        }
        Path(args.speed_view_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.speed_view_out).write_text(json.dumps(view, indent=2))
        print(f"[written] {args.speed_view_out}")

    if args.relative_hold_out:
        rel = {
            "source": "computed directly from raw report records (not copied from the summary)",
            "scope": ("supplemental post-run analysis; no re-evaluation and no modification of "
                      "the raw records, report summary or original analysis.json"),
            "manifest": {"path": str(args.manifest), "created_utc": manifest.get("created_utc")},
            "caveats": [
                "Statistical unit is the scenario (40), never the 96 fish or the policy-scenario.",
                "Per scenario the fixed 3 PPO policies are averaged, then paired against HOLD on the same seed/condition.",
                "Descriptive raw-derived baseline contrasts, not prespecified primary endpoints.",
                "HOLD is a contextual anchor, not the strongest rule here; flee_lead exceeds PPO in all conditions.",
                "The advantage-change CI is bootstrapped directly on the paired difference; do not subtract two independent CIs.",
                "Crossing zero on the advantage-change does not establish negligible impact; excluding zero does not establish importance.",
            ],
            "relative_hold": relative_hold_contrasts(grid),
        }
        Path(args.relative_hold_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.relative_hold_out).write_text(json.dumps(rel, indent=2))
        print(f"[written] {args.relative_hold_out}")


if __name__ == "__main__":
    main()
