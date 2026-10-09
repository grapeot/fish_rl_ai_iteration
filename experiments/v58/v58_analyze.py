#!/usr/bin/env python3
"""v58 analysis: the pre-registered independent confirmation of the training-time
initial-velocity randomization under a finite joint-stress distribution.

Primary contrast (pre-registered): the fixed-three-treatment mean minus the
fixed-three-control mean, computed WITHIN each scene for the combined-stress
condition. Per scene: treat_avg = mean over the 3 treatment models' final_survival;
ctrl_avg = mean over the 3 control models'; diff = treat_avg - ctrl_avg. A paired
episode bootstrap over the 64 scenes gives the 95% CI (each resample keeps the whole
base-scene row: both conditions and all six models together).

Secondary:
  * the nominal-condition contrast (same fixed-six models);
  * each controller's PPO-HOLD advantage change: per scene, average the 3 treatment
    (resp. 3 control) PPO policies, subtract the same-condition rule_hold survival,
    and take (combined_stress - nominal). Episode bootstrap over scenes.
  * every replicate's own direction.

Recommendation (report ONLY; no deployment action): support replacing the existing
nominal-training baseline only if BOTH of these hold —
  (A) the combined-stress mean difference is positive in ALL 3 paired replicates AND
      its episode-paired 95% CI lower endpoint exceeds 0;
  (B) the nominal mean loss is < 1 pp AND the nominal 95% CI lower endpoint is no
      worse than -2 pp.
Otherwise retain nominal training as default and report mixed/negative/inconclusive.
The 2 pp threshold is an explicit engineering tolerance chosen in the pre-declaration,
not a discovered statistical constant. This rule selects between these two recipes
only; neither need outperform the hand-written lead heuristic.

Usage:
  experiments/v48/.venv/bin/python experiments/v58/v58_analyze.py \
      --report-summary experiments/v58/artifacts/results/report.summary.json \
      --report-jsonl experiments/v58/artifacts/results/report.jsonl \
      --manifest experiments/v58/artifacts/frozen_config/v58_frozen_manifest.json \
      --out experiments/v58/artifacts/results/analysis.json
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

import v58_common as common  # noqa: E402
import v58_verify as verify  # noqa: E402

PHASE_STEPS = (1, 100, 250, 500)
PRIMARY_CONDITION = "combined_stress"
NOMINAL_CONDITION = "nominal"
HOLD_CONTROLLER = "rule_hold"
BOOTSTRAP_SEED = 580301
N_BOOT = 10000


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_bootstrap_ci(scene_diffs, n_boot=N_BOOT, alpha=0.05, seed=BOOTSTRAP_SEED):
    """Percentile CI over resampled SCENES (the whole base-scene row stays together)."""
    d = np.asarray(scene_diffs, dtype=float)
    if len(d) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def arm_condition_scene_means(grid, names, cond):
    """Per-scene mean final_survival over an arm's models for one condition."""
    out = {}
    for (_c, seed) in grid:
        out.setdefault(seed, None)
    for seed in list(out):
        vals = []
        for n in names:
            cell = grid.get((n, seed))
            if cell is None or cond not in cell:
                raise SystemExit(f"missing ({n}, {seed}, {cond}) in the grid")
            vals.append(cell[cond])
        out[seed] = float(np.mean(vals))
    return out


def scene_diffs_pair(grid, treat_names, ctrl_names, cond):
    """Per-scene (treatment-arm mean - control-arm mean) for a condition, seed-sorted."""
    seeds = sorted({s for (_c, s) in grid})
    t = arm_condition_scene_means(grid, treat_names, cond)
    c = arm_condition_scene_means(grid, ctrl_names, cond)
    return np.array([t[s] - c[s] for s in seeds], dtype=float), seeds


def ppo_minus_hold(grid, cond, ppo_names):
    """Per-scene (mean over PPO policies) - same-condition rule_hold survival."""
    seeds = sorted({s for (_c, s) in grid})
    vals = []
    for s in seeds:
        ppo = [grid[(n, s)][cond] for n in ppo_names]
        hold = grid[(HOLD_CONTROLLER, s)][cond]
        vals.append(float(np.mean(ppo)) - float(hold))
    return np.array(vals, dtype=float), seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-summary", required=True)
    ap.add_argument("--report-jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    conditions = list(manifest["conditions"])
    treat_names = [common.TREATMENT_MODEL_NAME[r] for r in common.REPLICATES]
    ctrl_names = [common.CONTROL_MODEL_NAME[r] for r in common.REPLICATES]

    out = {
        "conditions": conditions,
        "nominal_triple": manifest["nominal_triple"],
        "treatment_names": treat_names,
        "control_names": ctrl_names,
        "per_controller": {},
        "per_arm_condition": {},
        "rule_anchors": {},
        "combined_stress_triples_by_seed": {
            str(s): list(t)
            for s, t in zip(manifest["report_seeds"], manifest["combined_stress_triples"])
        },
        "manifest": {
            "path": str(args.manifest),
            "created_utc": manifest.get("created_utc"),
            "role": manifest.get("role"),
            "v57_manifest_sha256": manifest.get("depends_on", {}).get("v57_manifest_sha256"),
        },
        "caveats": [
            "Only 3 paired replicates per arm: the three-model spread is descriptive, "
            "not a sampling CI over training randomness.",
            "The episode-paired CI conditions on the six fixed selected models.",
            "Statistical unit is the episode (scene), never the 96 shared-policy fish.",
            "Each scene is one initial world; the combined-stress condition is a finite "
            "discrete artificial joint draw, not a natural deployment distribution.",
            "The paired bootstrap resamples whole base-scene rows (both conditions and "
            "all six models kept together); it never treats fish, conditions or "
            "model-scenario pairs as independent observations.",
            "No episode is dropped; an early all-death scene stays in with a fixed /96 "
            "denominator, and a first-step death induced by the combined stress is kept.",
            "The recommendation is a selection between the two recipes only; it is not a "
            "claim that either beats the hand-written lead heuristic.",
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

    # raw-vs-summary guard
    for ctrl, block in summary["controllers"].items():
        for cond in conditions:
            stats = block.get(cond)
            if stats is None:
                continue
            vals = [cell[cond] for (c, _s), cell in grid.items() if c == ctrl]
            if vals and abs(float(np.mean(vals)) - float(stats["mean_final_survival"])) > 1e-9:
                raise SystemExit(
                    f"refusing to analyze: summary mean for ({ctrl}, {cond}) != raw recomputation")

    for ctrl, block in summary["controllers"].items():
        out["per_controller"][ctrl] = block
    for name in common.RULE_CONTROLLERS:
        if name in summary["controllers"]:
            out["rule_anchors"][name] = summary["controllers"][name]

    # per-arm / per-condition means (arm average over the 3 models, per condition)
    seeds = sorted({s for (_c, s) in grid})
    for arm, names in (("treatment", treat_names), ("control", ctrl_names)):
        block = {}
        for cond in conditions:
            vals = [grid[(n, s)][cond] for n in names for s in seeds]
            block[cond] = {
                "n_models": len(names),
                "n_scenes": len(seeds),
                "mean_final_survival": float(np.mean(vals)),
            }
        out["per_arm_condition"][arm] = block

    # ---- primary: combined-stress treatment-vs-control, scene-paired -------------
    combined_diffs, _ = scene_diffs_pair(grid, treat_names, ctrl_names, PRIMARY_CONDITION)
    combined_mean, combined_lo, combined_hi = paired_bootstrap_ci(combined_diffs)

    # ---- secondary: nominal contrast ---------------------------------------------
    nominal_diffs, _ = scene_diffs_pair(grid, treat_names, ctrl_names, NOMINAL_CONDITION)
    nominal_mean, nominal_lo, nominal_hi = paired_bootstrap_ci(nominal_diffs)

    # ---- secondary: each controller's PPO-HOLD advantage change ------------------
    treat_adv, _ = ppo_minus_hold(grid, PRIMARY_CONDITION, treat_names)
    treat_adv_nom, _ = ppo_minus_hold(grid, NOMINAL_CONDITION, treat_names)
    ctrl_adv, _ = ppo_minus_hold(grid, PRIMARY_CONDITION, ctrl_names)
    ctrl_adv_nom, _ = ppo_minus_hold(grid, NOMINAL_CONDITION, ctrl_names)
    treat_did = treat_adv - treat_adv_nom
    ctrl_did = ctrl_adv - ctrl_adv_nom
    did = treat_did - ctrl_did
    did_mean, did_lo, did_hi = paired_bootstrap_ci(did)

    out["primary_combined_stress"] = {
        "contrast": "mean(3 treatment models) - mean(3 control models), per scene, combined_stress",
        "mean_diff": combined_mean,
        "ci95_low": combined_lo,
        "ci95_high": combined_hi,
        "win_rate": float((combined_diffs > 0).mean()),
        "loss_rate": float((combined_diffs < 0).mean()),
        "tie_rate": float((combined_diffs == 0).mean()),
        "n_scenes": len(combined_diffs),
    }
    out["secondary_nominal"] = {
        "contrast": "mean(3 treatment models) - mean(3 control models), per scene, nominal",
        "mean_diff": nominal_mean,
        "ci95_low": nominal_lo,
        "ci95_high": nominal_hi,
        "win_rate": float((nominal_diffs > 0).mean()),
        "loss_rate": float((nominal_diffs < 0).mean()),
        "tie_rate": float((nominal_diffs == 0).mean()),
        "n_scenes": len(nominal_diffs),
    }
    out["secondary_ppo_minus_hold"] = {
        "definition": "per scene mean(PPO policies) - rule_hold, same condition",
        "treatment": {
            "combined_stress": {"mean": float(treat_adv.mean())},
            "nominal": {"mean": float(treat_adv_nom.mean())},
        },
        "control": {
            "combined_stress": {"mean": float(ctrl_adv.mean())},
            "nominal": {"mean": float(ctrl_adv_nom.mean())},
        },
        "advantage_change_did": {
            "contrast": "(treatment PPO-HOLD adv in combined_stress - nominal) "
                        "- (control PPO-HOLD adv in combined_stress - nominal)",
            "mean": did_mean,
            "ci95_low": did_lo,
            "ci95_high": did_hi,
            "n_scenes": len(did),
        },
    }

    # ---- per-replicate direction (each pair of fixed models) ----------------------
    per_rep = {}
    for r in common.REPLICATES:
        d, _ = scene_diffs_pair(grid, [common.TREATMENT_MODEL_NAME[r]],
                                [common.CONTROL_MODEL_NAME[r]], PRIMARY_CONDITION)
        dn, _ = scene_diffs_pair(grid, [common.TREATMENT_MODEL_NAME[r]],
                                 [common.CONTROL_MODEL_NAME[r]], NOMINAL_CONDITION)
        m, lo, hi = paired_bootstrap_ci(d)
        mn, lon, hin = paired_bootstrap_ci(dn)
        per_rep[f"rep{r}"] = {
            "combined_stress": {"mean_diff": m, "ci95_low": lo, "ci95_high": hi,
                                "direction": "positive" if m > 0 else ("negative" if m < 0 else "zero")},
            "nominal": {"mean_diff": mn, "ci95_low": lon, "ci95_high": hin,
                        "direction": "positive" if mn > 0 else ("negative" if mn < 0 else "zero")},
        }
    out["per_replicate_direction"] = per_rep

    # ---- recommendation predicates (report only) ----------------------------------
    all_rep_positive = all(per_rep[f"rep{r}"]["combined_stress"]["mean_diff"] > 0
                           for r in common.REPLICATES)
    primary_predicate = bool(all_rep_positive and combined_lo > 0.0)
    nominal_loss = -nominal_mean  # positive = treatment worse
    nominal_predicate = bool(nominal_loss < manifest["recommendation_rule"]["nominal_loss_tolerance"]
                             and nominal_lo >= manifest["recommendation_rule"]["nominal_ci_lower_floor"])
    overall_pass = bool(primary_predicate and nominal_predicate)
    out["recommendation"] = {
        "note": "pre-registered rule; report only, not a deployment action",
        "primary_predicate_all3rep_positive_and_ci_lower_gt0": primary_predicate,
        "primary_detail": {
            "all_3_replicates_positive": bool(all_rep_positive),
            "combined_ci95_low": combined_lo,
            "combined_ci_lower_gt_0": bool(combined_lo > 0.0),
        },
        "nominal_predicate_loss_lt_1pp_and_ci_lower_ge_-2pp": nominal_predicate,
        "nominal_detail": {
            "nominal_mean_loss_pp": nominal_loss * 100.0,
            "loss_tolerance_pp": manifest["recommendation_rule"]["nominal_loss_tolerance"] * 100.0,
            "nominal_ci95_low_pp": nominal_lo * 100.0,
            "ci_lower_floor_pp": manifest["recommendation_rule"]["nominal_ci_lower_floor"] * 100.0,
        },
        "overall_pass": overall_pass,
        "recommendation": ("support replacing nominal-training baseline"
                           if overall_pass else "retain nominal training as default"),
        "tolerance_is_engineering_choice": True,
    }

    # ---- phase survival means (fixed denominators) --------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")
    print(json.dumps({
        "primary_combined_stress": {k: out["primary_combined_stress"][k]
                                    for k in ("mean_diff", "ci95_low", "ci95_high")},
        "secondary_nominal": {k: out["secondary_nominal"][k]
                              for k in ("mean_diff", "ci95_low", "ci95_high")},
        "recommendation": out["recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
