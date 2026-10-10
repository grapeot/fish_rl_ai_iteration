#!/usr/bin/env python3
"""Same-scene before/after benchmark: 5 controllers x N frozen scenarios.

Every controller runs the SAME scenario seed on its own env (neighbor flag matching
its observation dim), at the nominal reset. Deterministic PPO inference and pure
rule actions. Records per-episode final alive/96 and phase survival at
steps 1/100/250/500 with a fixed /96 denominator.

Usage:
  python benchmark.py --seeds debug --workers 4 --out results/debug_raw.jsonl
  python benchmark.py --seeds report --workers 4 --out results/report_raw.jsonl

The protocol binding written by prepare_protocol.py is loaded and the loaded model
tensor hashes are re-verified against it before any episode runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import demo_common as C  # noqa: E402


def worker(job):
    import torch

    torch.set_num_threads(1)
    controller, seeds = job
    env = C.make_env(controller["neighbor"])
    model = None
    if controller["kind"] == "policy":
        from stable_baselines3 import PPO

        model = PPO.load(str(C.ROOT / controller["path"]), device="cpu")
    records = []
    try:
        for seed in seeds:
            t0 = time.time()
            rec = C.run_episode(env, seed, controller, model)
            rec["elapsed_sec"] = time.time() - t0
            records.append(rec)
    finally:
        env.close()
    return controller["name"], records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["report", "debug"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Verify the frozen protocol binding before running anything.
    binding = json.loads((C.DEMO_DIR / "results" / "protocol_binding.json").read_text())
    by_name = {c["name"]: c for c in binding["controllers"]}
    for c in C.CONTROLLERS:
        b = by_name[c["name"]]
        assert b["include_neighbor_features"] == c["neighbor"], f"neighbor flag mismatch {c['name']}"
        assert b["checkpoint"] == c.get("path"), f"path mismatch {c['name']}"
        if c["kind"] == "policy":
            h = C.policy_tensor_sha256_from_zip(C.ROOT / c["path"])
            assert h == b["policy_tensor_sha256"], f"hash mismatch {c['name']}"
    print("[protocol] binding re-verified")

    seeds = C.report_seeds() if args.seeds == "report" else C.debug_seeds()
    n_chunks = max(1, min(args.workers, len(seeds)))
    chunks = [list(map(int, c)) for c in np.array_split(np.array(seeds), n_chunks)]
    jobs = [(c, ch) for c in C.CONTROLLERS for ch in chunks]

    t_start = time.time()
    all_records = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for name, recs in ex.map(worker, jobs):
            all_records.setdefault(name, []).extend(recs)
    wall = time.time() - t_start

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name in C.CONTROLLER_NAMES:
            for rec in sorted(all_records[name], key=lambda r: (r["seed"],)):
                f.write(json.dumps(rec) + "\n")

    summary = {
        "seed_set": args.seeds,
        "n_seeds": len(seeds),
        "controllers": C.CONTROLLER_NAMES,
        "workers": args.workers,
        "wall_sec": wall,
        "episodes": sum(len(all_records[n]) for n in C.CONTROLLER_NAMES),
        "per_controller": {},
    }
    for name in C.CONTROLLER_NAMES:
        recs = all_records[name]
        vals = np.array([r["final_survival"] for r in recs], dtype=float)
        summary["per_controller"][name] = {
            "n": len(vals),
            "mean_final_survival": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "mean_steps": float(np.mean([r["steps"] for r in recs])),
            "total_step_one_deaths": int(sum(r["num_step_one_deaths"] for r in recs)),
            "mean_survival_at": {
                str(ph): float(np.mean([r["survival_at"][str(ph)] for r in recs]))
                for ph in C.PHASE_STEPS
            },
        }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_controller"], indent=2))
    print(f"[wall] {wall:.1f}s for {summary['episodes']} episodes, {args.workers} workers")
    print(f"[written] {out_path}\n[written] {out_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
