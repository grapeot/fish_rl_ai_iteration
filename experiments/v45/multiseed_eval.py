#!/usr/bin/env python3
"""Multi-seed held-out evaluation harness.

Motivation (v45 finding): the multi-eval predator configs are already fixed and
shared across runs (same multi_eval_seed_base -> identical episode_seeds), so the
eval is fair and deterministic. The large run-to-run variance lives in the
TRAINING seed: identical config with two training seeds produced final held-out
avg_final of ~0.70 (v42) vs ~0.51 (v45). A single-seed run therefore cannot be
used as a merge criterion.

This harness runs the SAME training config across K seeds, extracts each run's
held-out avg_final (mean over the last N multi-evals), and reports the cross-seed
mean +/- std. That distribution is the unit of comparison between configs.

Usage:
    python multiseed_eval.py --label baseline_v42cfg --seeds 3 \
        --train-args "<full train.py arg string WITHOUT --seed/--run_name>"

Runs are launched with bounded parallelism (default 2 concurrent 128-env runs on
the 32-core box) and results are aggregated into a JSON summary.
"""
import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train.py"
PY = str(Path(sys.executable))


def extract_final_avg_final(eval_history_path: Path, last_n: int) -> float | None:
    """Mean of avg_final_survival_rate over the last `last_n` multi-evals."""
    if not eval_history_path.exists():
        return None
    vals = []
    for line in eval_history_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        s = d.get("summary") or {}
        v = s.get("avg_final_survival_rate")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    tail = vals[-last_n:] if last_n > 0 else vals
    return float(statistics.mean(tail))


def run_batch(commands, max_parallel, log_dir):
    """Run (run_name, cmd_list) tuples with bounded parallelism."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(commands)
    active = []  # (run_name, Popen, log_fh)
    while pending or active:
        while pending and len(active) < max_parallel:
            run_name, cmd = pending.pop(0)
            log_fh = open(log_dir / f"{run_name}.log", "w")
            print(f"[launch] {run_name}", flush=True)
            p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            active.append((run_name, p, log_fh))
        still = []
        for run_name, p, fh in active:
            if p.poll() is None:
                still.append((run_name, p, fh))
            else:
                fh.close()
                print(f"[done]   {run_name} exit={p.returncode}", flush=True)
        active = still
        if active:
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="config label for output naming")
    ap.add_argument("--seeds", type=int, default=3, help="number of training seeds")
    ap.add_argument("--seed-base", type=int, default=700000, help="first seed; run i uses seed-base + i*1000")
    ap.add_argument("--train-args", required=True, help="full train.py arg string WITHOUT --seed / --run_name")
    ap.add_argument("--max-parallel", type=int, default=2, help="concurrent 128-env runs (32-core box)")
    ap.add_argument("--last-n", type=int, default=3, help="avg over last N multi-evals per run")
    ap.add_argument("--out", type=str, default=None, help="output JSON path")
    args = ap.parse_args()

    base_train_args = shlex.split(args.train_args)
    ckpt_root = HERE / "artifacts" / "checkpoints"
    log_dir = HERE / "artifacts" / "logs" / f"multiseed_{args.label}"

    commands = []
    run_names = []
    for i in range(args.seeds):
        seed = args.seed_base + i * 1000
        run_name = f"ms_{args.label}_seed{seed}"
        run_names.append((run_name, seed))
        cmd = [PY, str(TRAIN), "--run_name", run_name, "--seed", str(seed)] + base_train_args
        commands.append((run_name, cmd))

    print(f"[multiseed] label={args.label} seeds={args.seeds} max_parallel={args.max_parallel}", flush=True)
    t0 = time.monotonic()
    run_batch(commands, args.max_parallel, log_dir)
    elapsed = time.monotonic() - t0

    per_seed = []
    for run_name, seed in run_names:
        hist = ckpt_root / run_name / "eval_multi_history.jsonl"
        val = extract_final_avg_final(hist, args.last_n)
        per_seed.append({"run_name": run_name, "seed": seed, "final_avg_final": val})
        print(f"[result] {run_name} final_avg_final={val}", flush=True)

    vals = [r["final_avg_final"] for r in per_seed if r["final_avg_final"] is not None]
    summary = {
        "label": args.label,
        "seeds": args.seeds,
        "last_n": args.last_n,
        "elapsed_sec": round(elapsed, 1),
        "per_seed": per_seed,
        "mean": round(statistics.mean(vals), 4) if vals else None,
        "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else None,
        "min": round(min(vals), 4) if vals else None,
        "max": round(max(vals), 4) if vals else None,
        "n_ok": len(vals),
    }
    out_path = Path(args.out) if args.out else (HERE / "artifacts" / f"multiseed_{args.label}.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[SUMMARY] {args.label}: mean={summary['mean']} std={summary['std']} "
          f"range=[{summary['min']},{summary['max']}] n={summary['n_ok']}/{args.seeds}", flush=True)
    print(f"[written] {out_path}", flush=True)


if __name__ == "__main__":
    main()
