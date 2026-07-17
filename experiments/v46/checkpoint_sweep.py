#!/usr/bin/env python3
"""Best-checkpoint sweep: evaluate every saved checkpoint of one or more runs
on a fixed held-out selection set, then score each run's best-selected
checkpoint and its final checkpoint on a disjoint report set.

Motivation (v46 finding): the bad training seed 450060 is healthy on held-out
eval through iter6 (avg_final 0.80-0.82) and only collapses afterwards, while
on-policy sr never flags a problem. The good policy exists mid-run and is
destroyed by later training. train.py already saves checkpoints every 5 iters,
so best-checkpoint selection can rescue such runs post-hoc with zero retraining.

Protocol (two disjoint eval sets to avoid selection overfitting):
  - selection set: 20 episodes, seeds from default_rng(555001)
  - report set:    40 episodes, seeds from default_rng(555002)
  Best checkpoint is chosen on the selection set only; the number that may be
  compared against baselines is the report-set score.

Env config mirrors the training-time multi-eval probe (argparse defaults of
train.py, verified against reproduce_eval.py): 96 fish, neighbor features on,
escape_boost_speed fixed at 0.8 (final-stage value) so every checkpoint faces
the same test distribution.

Usage:
  venv/bin/python experiments/v46/checkpoint_sweep.py \
      --run-dir experiments/v45/artifacts/checkpoints/v45_avgfinal_gate \
      --run-dir experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000 \
      --workers 24 --out experiments/v46/artifacts/checkpoint_sweep.json
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

NUM_FISH = 96
SELECTION_RNG_SEED = 555001
REPORT_RNG_SEED = 555002
ESCAPE_BOOST_SPEED = 0.8

PREDATOR_HEADING_BIAS_SPEC = (
    "0-60:1.1,60-90:0.8,90-120:0.08,120-150:0.05,"
    "150-210:0.9,210-270:1.3,270-330:1.2,330-360:0.95"
)
PREDATOR_SPEED_BIAS_SPEC = "0-1.0:1.6,1.0-1.6:1.3,1.6-2.0:0.7,2.0-2.4:0.35,2.4-3.0:0.2"


def parse_heading_bias(spec):
    out = []
    for chunk in spec.split(","):
        rng, weight = chunk.strip().split(":")
        start, end = rng.split("-")
        out.append({"start_deg": float(start), "end_deg": float(end), "weight": float(weight)})
    return out


def parse_speed_bias(spec):
    out = []
    for chunk in spec.split(","):
        rng, weight = chunk.strip().split(":")
        lo, hi = rng.split("-")
        out.append({"min_speed": float(lo), "max_speed": float(hi), "weight": float(weight)})
    return out


def episode_seeds(rng_seed, n):
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def eval_checkpoint(job):
    """Worker: evaluate one checkpoint on one seed list. Returns summary dict."""
    model_path, seeds, tag = job
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO

    from fish_env import FishEscapeEnv

    env = FishEscapeEnv(
        num_fish=NUM_FISH,
        include_neighbor_features=True,
        neighbor_radius=3.0,
        neighbor_average_count=6,
        initial_escape_boost=True,
        escape_boost_speed=ESCAPE_BOOST_SPEED,
        escape_jitter_std=0.35,
        divergence_reward_coef=0.0,
        density_penalty_coef=0.05,
        density_target=0.4,
        predator_spawn_jitter_radius=1.6,
        predator_pre_roll_steps=16,
        predator_pre_roll_angle_jitter=0.3,
        predator_pre_roll_speed_jitter=0.2,
        predator_heading_bias=parse_heading_bias(PREDATOR_HEADING_BIAS_SPEC),
        predator_pre_roll_speed_bias=parse_speed_bias(PREDATOR_SPEED_BIAS_SPEC),
    )
    model = PPO.load(str(model_path), device="cpu")
    finals = []
    for seed in seeds:
        obs, _info = env.reset(seed=seed)
        while True:
            if len(obs) > 0:
                batch_actions, _ = model.predict(np.asarray(obs), deterministic=True)
                actions = [int(a) for a in np.atleast_1d(batch_actions)]
            else:
                actions = []
            obs, _r, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                finals.append(int(info.get("num_alive", 0)) / float(NUM_FISH))
                break
    env.close()
    arr = sorted(finals)
    return {
        "tag": tag,
        "model": str(model_path),
        "episodes": len(finals),
        "avg_final": sum(finals) / len(finals),
        "min_final": arr[0],
        "max_final": arr[-1],
    }


def list_checkpoints(run_dir):
    out = []
    for p in sorted(run_dir.glob("model_iter_*.zip")):
        m = re.search(r"model_iter_(\d+)\.zip", p.name)
        out.append((int(m.group(1)), p))
    out.sort()
    final = run_dir / "model_final.zip"
    if final.exists():
        out.append(("final", final))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--selection-episodes", type=int, default=20)
    ap.add_argument("--report-episodes", type=int, default=40)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    sel_seeds = episode_seeds(SELECTION_RNG_SEED, args.selection_episodes)
    rep_seeds = episode_seeds(REPORT_RNG_SEED, args.report_episodes)

    run_dirs = [Path(d) for d in args.run_dir]
    jobs = []
    for rd in run_dirs:
        for it, path in list_checkpoints(rd):
            jobs.append((path, sel_seeds, f"{rd.name}|sel|{it}"))
    print(f"[sweep] selection phase: {len(jobs)} checkpoint evals x {len(sel_seeds)} eps", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        sel_results = list(ex.map(eval_checkpoint, jobs))

    by_run = {}
    for r in sel_results:
        run, _, it = r["tag"].split("|")
        by_run.setdefault(run, []).append({**r, "iter": it})
    for run, rows in by_run.items():
        rows.sort(key=lambda x: (x["iter"] != "final", int(x["iter"]) if x["iter"] != "final" else 0))
        print(f"\n[{run}] selection-set avg_final by checkpoint:")
        for row in rows:
            print(f"  iter {row['iter']:>5}: avg={row['avg_final']:.4f} min={row['min_final']:.4f}")

    report_jobs = []
    chosen = {}
    for run, rows in by_run.items():
        best = max(rows, key=lambda x: x["avg_final"])
        final = next((x for x in rows if x["iter"] == "final"), None)
        chosen[run] = {"best_iter": best["iter"], "best_sel_avg": best["avg_final"]}
        report_jobs.append((Path(best["model"]), rep_seeds, f"{run}|rep_best|{best['iter']}"))
        if final is not None and final["model"] != best["model"]:
            report_jobs.append((Path(final["model"]), rep_seeds, f"{run}|rep_final|final"))
        elif final is not None:
            chosen[run]["best_is_final"] = True
    print(f"\n[sweep] report phase: {len(report_jobs)} evals x {len(rep_seeds)} eps", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rep_results = list(ex.map(eval_checkpoint, report_jobs))

    for r in rep_results:
        run, kind, it = r["tag"].split("|")
        chosen[run][kind] = {"iter": it, "avg_final": r["avg_final"], "min_final": r["min_final"]}
    for run, info in chosen.items():
        if info.get("best_is_final") and "rep_best" in info:
            info["rep_final"] = dict(info["rep_best"])

    print("\n=== Report-set comparison (disjoint 40-episode set) ===")
    for run in sorted(chosen):
        info = chosen[run]
        b = info.get("rep_best", {})
        f = info.get("rep_final", {})
        print(
            f"  {run}: final avg={f.get('avg_final', float('nan')):.4f} -> "
            f"best(iter {info['best_iter']}) avg={b.get('avg_final', float('nan')):.4f}"
        )

    out = {
        "config": {
            "num_fish": NUM_FISH,
            "escape_boost_speed": ESCAPE_BOOST_SPEED,
            "selection_rng_seed": SELECTION_RNG_SEED,
            "report_rng_seed": REPORT_RNG_SEED,
            "selection_episodes": args.selection_episodes,
            "report_episodes": args.report_episodes,
        },
        "selection": sel_results,
        "chosen": chosen,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {out_path}", flush=True)


if __name__ == "__main__":
    main()
