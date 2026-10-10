#!/usr/bin/env python3
"""v52 analysis: read raw episodes + frozen spec, emit phased survival, action
distributions, per-controller paired stats and the fixed-3-PPO aggregate.

Unit of aggregation: the episode (per scenario the 3 PPO models are averaged
first, then bootstrapped), matching ``v52_eval``. No training uncertainty is
represented; CIs describe scenario-sampling only.

Usage:
  experiments/v48/.venv/bin/python experiments/v52/current_baseline/v52_analyze.py \
      --raw experiments/v52/current_baseline/artifacts/raw_episodes.jsonl \
      --manifest experiments/v52/current_baseline/artifacts/v52_spec_manifest.json \
      --out experiments/v52/current_baseline/artifacts/analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V50 = ROOT / "experiments" / "v50"
for p in (str(ROOT), str(V50), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v52_eval as v52  # noqa: E402


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_records_strict(records, manifest):
    """Strict key gate: every (controller, mode) must cover exactly the frozen
    scenario indices 0..N-1 once, with no missing, duplicate or extra record,
    and each record's seed must equal the frozen seed for that index."""
    seeds = [int(s) for s in manifest["scenarios"]]
    expected = set(range(len(seeds)))
    by_key = {}
    for r in records:
        key = (r["controller"], r["mode"])
        by_key.setdefault(key, {})
        idx = int(r["scenario_index"])
        if idx in by_key[key]:
            raise ValueError(f"duplicate record ({key}, index={idx})")
        if idx in expected and int(r["seed"]) != seeds[idx]:
            raise ValueError(
                f"seed mismatch ({key}, index={idx}): {r['seed']} != {seeds[idx]}")
        by_key[key][idx] = r
    expected_keys = {(c, m) for c in v52.CONTROLLERS for m in v52.MODES}
    if set(by_key) != expected_keys:
        raise ValueError(f"controller/mode set mismatch: {sorted(set(by_key))}")
    for key, got in by_key.items():
        if set(got) != expected:
            missing = sorted(expected - set(got))[:5]
            extra = sorted(set(got) - expected)[:5]
            raise ValueError(
                f"{key} scenario mismatch: n={len(got)} missing={missing} extra={extra}")
    return by_key


def matched_observation_diagnostic(manifest, out_path):
    """Reset-only diagnostic: no environment steps are advanced.

    Resets each of the frozen scenarios to obtain the SAME initial observations,
    then compares the deterministic action of each frozen final model under the
    full observation vs each masked copy of that identical observation. Counts
    how many rows change action. This isolates same-state input dependence; it is
    not a trajectory flip rate and does not by itself imply a survival benefit.
    """
    import v50_common as common
    from stable_baselines3 import PPO

    seeds = [int(s) for s in manifest["scenarios"]]
    obs_chunks = []
    for seed in seeds:
        env = common.make_base_env(include_neighbor_features=False)
        try:
            obs, _ = env.reset(seed=seed)
            obs_chunks.append(np.asarray(obs, dtype=np.float32))
        finally:
            env.close()
    base_obs = np.concatenate(obs_chunks, axis=0)
    n_rows = int(base_obs.shape[0])
    visible = int((base_obs[:, 5] > 0.5).sum())

    result = {
        "kind": "v52_matched_observation_diagnostic",
        "note": "Reset-only same-observation diagnostic; no env steps advanced. "
                "Counts deterministic action changes on identical observations; "
                "not a trajectory flip rate and not proof of beneficial use.",
        "n_seeds": len(seeds),
        "n_observations": n_rows,
        "n_visible": visible,
        "models": {},
    }
    for name in v52.PPO_CONTROLLERS:
        path = Path(manifest["models"][name]["checkpoint"])
        model = PPO.load(str(ROOT / path), device="cpu")
        base_act = np.asarray(model.predict(base_obs, deterministic=True)[0])
        entry = {}
        for mode in ("mask_predator", "mask_velocity_only"):
            masked = v52.apply_mask(base_obs, mode)
            act = np.asarray(model.predict(masked, deterministic=True)[0])
            changed = act != base_act
            entry[mode] = {
                "changes": int(changed.sum()),
                "changes_visible_only": int((changed & (base_obs[:, 5] > 0.5)).sum()),
            }
        result["models"][name] = entry
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def aggregate(records, manifest):
    n = int(manifest["n_scenarios"])
    phases = [str(p) for p in manifest["phase_steps"]]
    by_key = {}
    for r in records:
        by_key.setdefault((r["controller"], r["mode"]), []).append(r)

    out = {"per_controller_mode": {}, "paired_vs_full": {}, "ppo_aggregate": {}}
    for (controller, mode), recs in sorted(by_key.items()):
        recs = sorted(recs, key=lambda r: r["scenario_index"])
        surv = np.array([r["final_survival"] for r in recs], dtype=float)
        phased = {}
        for ph in phases:
            vals = [r["survival_at"][ph] for r in recs if ph in r.get("survival_at", {})]
            phased[ph] = float(np.mean(vals)) if vals else None
        adist = np.array([r.get("action_dist", [0, 0, 0, 0, 0]) for r in recs], dtype=float)
        out["per_controller_mode"][f"{controller}|{mode}"] = {
            "n": int(len(surv)),
            "mean_final_survival": float(surv.mean()),
            "std": float(surv.std(ddof=1)) if len(surv) > 1 else 0.0,
            "mean_survival_at_step": phased,
            "mean_action_dist": adist.mean(axis=0).tolist() if adist.size else [0] * 5,
        }

    def full_idx(controller):
        d = {r["scenario_index"]: r["final_survival"]
             for r in by_key.get((controller, "full"), [])}
        return d

    for (controller, mode), recs in sorted(by_key.items()):
        if mode == "full":
            continue
        f = full_idx(controller)
        diffs = [r["final_survival"] - f[r["scenario_index"]]
                 for r in recs if r["scenario_index"] in f]
        d = np.array(diffs, dtype=float)
        lo, hi = v52.bootstrap_ci(d)
        out["paired_vs_full"][f"{controller}|{mode}"] = {
            "n_pairs": int(len(d)),
            "mean_diff": float(d.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "win_rate": float((d > 0).mean()),
            "tie_rate": float((d == 0).mean()),
            "loss_rate": float((d < 0).mean()),
        }

    for mode in ("mask_predator", "mask_velocity_only"):
        per_scene = []
        for i in range(n):
            ds = []
            for controller in v52.PPO_CONTROLLERS:
                a = {r["scenario_index"]: r["final_survival"]
                     for r in by_key.get((controller, mode), [])}.get(i)
                f = full_idx(controller).get(i)
                if a is not None and f is not None:
                    ds.append(a - f)
            per_scene.append(float(np.mean(ds)) if ds else np.nan)
        arr = np.array(per_scene, dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = v52.bootstrap_ci(arr)
        out["ppo_aggregate"][mode] = {
            "n_scenes": int(len(arr)),
            "mean_diff": float(arr.mean()) if len(arr) else float("nan"),
            "ci95_low": lo,
            "ci95_high": hi,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matched-obs", default=None,
                    help="optional: write a reset-only same-observation action-dependence "
                         "diagnostic to this path (no env steps, no new eval episodes)")
    args = ap.parse_args()

    records = load_records(args.raw)
    manifest = json.loads(Path(args.manifest).read_text())
    validate_records_strict(records, manifest)
    result = {"kind": "v52_analysis", "n_records": len(records),
              "manifest_script_sha256": manifest.get("script_sha256")}
    result.update(aggregate(records, manifest))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[written] {args.out}")
    if args.matched_obs:
        diag = matched_observation_diagnostic(manifest, args.matched_obs)
        print(json.dumps(diag, indent=2))
        print(f"[written] {args.matched_obs}")


if __name__ == "__main__":
    main()
