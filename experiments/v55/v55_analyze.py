#!/usr/bin/env python3
"""v55 analysis: the one-axis post-reset predator-velocity-rotation generalization
contrast.

Reads the gated report jsonl/summary and writes one JSON with:
  * per-controller, per-condition final-survival mean/std and phase survival
    (step 1/100/250/500), action distribution and early-failure counts;
  * per-PPO-replicate means;
  * the paired condition contrasts, each relative to that SAME controller's own
    `deg0` (nominal) on the SAME scenario: (deg90 - deg0), (deg180 - deg0),
    (deg270 - deg0);
  * the aggregate contrast: per scenario, average the paired difference over the
    fixed three PPO policies, then bootstrap over scenarios (episode unit). The
    three policies' initial worlds are not treated as extra samples;
  * a descriptive "worst rotation" listing: for each controller the rotation
    condition with the lowest mean final survival (and the nominal worst-case
    episode seed), for description only, never to pick a training plan;
  * rule anchors for reference.

Caveats baked in: episode (scenario) is the statistical unit, never the 96 fish;
each PPO policy is a fixed model, not a random draw; the reported spread across
three policies is descriptive. The rotations probe generalization across the
post-reset predator heading, not the natural initial-heading distribution; the
`deg0` condition is the exact identity.

Usage:
  experiments/v48/.venv/bin/python experiments/v55/v55_analyze.py \
      --report-summary experiments/v55/artifacts/results/report.summary.json \
      --report-jsonl experiments/v55/artifacts/results/report.jsonl \
      --manifest experiments/v55/artifacts/frozen_config/v55_frozen_manifest.json \
      --out experiments/v55/artifacts/results/analysis.json
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

import v55_common as common  # noqa: E402
import v55_verify as verify  # noqa: E402

PHASE_STEPS = (1, 100, 250, 500)
BASE_CONDITION = "deg0"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=550301):
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


HOLD_CONTROLLER = "rule_hold"


def ppo_minus_hold(grid, cond, ppo_names):
    """Per-scenario (mean over the fixed PPO policies) - same-condition HOLD.

    For each scenario seed: first average the fixed three PPO policies' survival
    in this condition, then subtract the SAME condition's HOLD survival. Returns
    the per-scenario values sorted by seed.
    """
    vals, seeds = [], []
    report_seeds = sorted({s for (_c, s) in grid})
    for seed in report_seeds:
        ppo_cells = [grid.get((name, seed)) for name in ppo_names]
        hold_cell = grid.get((HOLD_CONTROLLER, seed))
        if any(c is None or cond not in c for c in ppo_cells) or hold_cell is None or cond not in hold_cell:
            raise SystemExit(f"relative-to-HOLD analysis needs a full grid; missing seed {seed} cond {cond}")
        ppo_mean = float(np.mean([c[cond] for c in ppo_cells]))
        vals.append(ppo_mean - hold_cell[cond])
        seeds.append(seed)
    return vals, seeds


def did_advantage(grid, cond, base, ppo_names):
    """Per-scenario DiD: (PPO-HOLD advantage in cond) - (same in base).

    Computed per scenario, not by subtracting two CI endpoints.
    """
    adv_cond, seeds = ppo_minus_hold(grid, cond, ppo_names)
    adv_base, _ = ppo_minus_hold(grid, base, ppo_names)
    return [a - b for a, b in zip(adv_cond, adv_base)], seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relative-out", default=None,
                    help="optional: write the post-hoc PPO-HOLD relative analysis here "
                         "(does not modify analysis.json)")
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    conditions = list(manifest["conditions"])

    out = {
        "conditions": conditions,
        "rotation_degrees": manifest["rotation_degrees"],
        "per_controller": {},
        "per_ppo_replicate": {},
        "paired_vs_deg0": {},
        "aggregate_ppo_contrast": {},
        "worst_rotation_descriptive": {},
        "rule_anchors": {},
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "intervention": manifest.get("intervention"),
        },
        "caveats": [
            "Statistical unit is the episode (scenario), never the 96 fish.",
            "Every condition diff is paired to the SAME controller's deg0 (nominal) on the SAME initial world.",
            "The three PPO policies are fixed models, not a random draw; their spread is descriptive.",
            "The aggregate contrast averages the 3 PPO policies per scenario, then bootstraps over scenarios.",
            "No episode is dropped; an early all-death scenario stays in with a fixed /96 denominator.",
            "The rotations probe post-reset predator-heading generalization, not the natural initial-heading distribution.",
            "The worst-rotation listing is descriptive only and must not select a training plan.",
            "Gravity stays fixed at +y; rotating the predator velocity is not a gravity turn.",
            "A first-step death caused by the rotated spawn direction is kept, not filtered.",
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

    # paired contrasts vs own deg0, per controller
    for ctrl in sorted({c for (c, _s) in grid}):
        out["paired_vs_deg0"][ctrl] = {}
        for cond in conditions:
            if cond == BASE_CONDITION:
                continue
            diffs, seeds = paired_contrast(grid, ctrl, cond, BASE_CONDITION)
            lo, hi = bootstrap_ci(diffs)
            d = np.array(diffs, dtype=float)
            out["paired_vs_deg0"][ctrl][cond] = {
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

    # descriptive worst-rotation listing (never used to pick a training plan)
    for ctrl in sorted({c for (c, _s) in grid}):
        cond_means = {}
        for cond in conditions:
            vals = [cell[cond] for (c, _s), cell in grid.items() if c == ctrl]
            cond_means[cond] = float(np.mean(vals)) if vals else None
        worst_cond = min(conditions, key=lambda c: cond_means[c])
        # nominal worst-case episode seed for that controller (descriptive)
        nominal_pairs = [(seed, cell[BASE_CONDITION]) for (c, seed), cell in grid.items() if c == ctrl]
        worst_seed, worst_val = None, None
        if nominal_pairs:
            worst_seed, worst_val = min(nominal_pairs, key=lambda kv: kv[1])
        out["worst_rotation_descriptive"][ctrl] = {
            "condition_means": cond_means,
            "worst_condition": worst_cond,
            "worst_condition_mean": cond_means[worst_cond],
            "deg0_worst_episode_seed": int(worst_seed) if worst_seed is not None else None,
            "deg0_worst_episode_survival": float(worst_val) if worst_val is not None else None,
            "note": "descriptive only; not a basis for selecting a training plan",
        }

    # rule anchors (reference)
    for name in common.RULE_CONTROLLERS:
        if name in summary["controllers"]:
            out["rule_anchors"][name] = summary["controllers"][name]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")

    # --- post-hoc relative-to-HOLD analysis (separate file; never touches analysis.json) ---
    if args.relative_out:
        rel = {
            "scope": "post-hoc supplementary analysis on the SAME frozen raw; no new eval, "
                     "no training, no model/bank/threshold change. NOT a pre-registered primary metric.",
            "conditions": conditions,
            "rotation_degrees": manifest["rotation_degrees"],
            "definition": (
                "For each scenario seed and condition: average the fixed three PPO policies' "
                "survival, then subtract the SAME condition's rule_hold survival "
                "(G = mean(rep0,rep1,rep2) - HOLD). Paired episode bootstrap over the 40 seeds. "
                "The DiD is computed per scenario as G(cond) - G(deg0), never by subtracting CI endpoints."
            ),
            "reference_controller": HOLD_CONTROLLER,
            "ppo_minus_hold": {},
            "did_advantage_vs_deg0": {},
            "rule_hold_contrast_vs_deg0": {},
            "bootstrap": {"n_boot": 10000, "alpha": 0.05, "rng_seed": 550301,
                          "method": "percentile over episodes (scenarios)"},
            "caveats": [
                "Episode (scenario) is the unit; the fixed three PPO policies are averaged first, then bootstrapped over scenarios.",
                "The DiD is the advantage change under THIS specific passive reference (rule_hold); it is not a causal removal of all environment difficulty.",
                "rule_hold's own CI crossing zero does not imply baseline difficulty is unchanged.",
                "PPO still exceeds same-condition HOLD at all four points; this is a comparison, not proof of direction robustness.",
                "Post-hoc supplement: cannot be retroactively promoted to a pre-run primary metric.",
            ],
        }
        for cond in conditions:
            vals, seeds = ppo_minus_hold(grid, cond, common.PPO_CONTROLLER_NAMES)
            lo, hi = bootstrap_ci(vals)
            d = np.array(vals, dtype=float)
            rel["ppo_minus_hold"][cond] = {
                "mean": float(d.mean()) if len(d) else None,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": len(vals),
                "win_rate": float((d > 0).mean()) if len(d) else None,
                "loss_rate": float((d < 0).mean()) if len(d) else None,
            }
        for cond in conditions:
            if cond == BASE_CONDITION:
                continue
            vals, seeds = did_advantage(grid, cond, BASE_CONDITION, common.PPO_CONTROLLER_NAMES)
            lo, hi = bootstrap_ci(vals)
            d = np.array(vals, dtype=float)
            rel["did_advantage_vs_deg0"][cond] = {
                "contrast": f"(PPO-HOLD {cond}) - (PPO-HOLD {BASE_CONDITION})",
                "mean": float(d.mean()) if len(d) else None,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": len(vals),
                "win_rate": float((d > 0).mean()) if len(d) else None,
                "loss_rate": float((d < 0).mean()) if len(d) else None,
            }
        hold_diffs, _ = paired_contrast(grid, HOLD_CONTROLLER, "deg180", BASE_CONDITION)
        lo, hi = bootstrap_ci(hold_diffs)
        d = np.array(hold_diffs, dtype=float)
        rel["rule_hold_contrast_vs_deg0"]["deg180"] = {
            "contrast": f"rule_hold deg180 - {BASE_CONDITION}",
            "mean": float(d.mean()) if len(d) else None,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": len(hold_diffs),
        }
        Path(args.relative_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.relative_out).write_text(json.dumps(rel, indent=2))
        print(f"[written] {args.relative_out}")


if __name__ == "__main__":
    main()
