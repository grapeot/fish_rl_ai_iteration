#!/usr/bin/env python3
"""v58 report orchestrator: build the report arm list from the frozen manifest and run
the fresh v58 report eval, gated on that manifest.

Report arms (exactly the frozen set):
  * rule controls: rule_hold (reference), rule_flee_lead, rule_safe_top;
  * the six frozen selected models (3 reused v50 controls + 3 v57 treatments).

9 controllers x 2 conditions x 64 scenes = 1152 episodes. v58_evaluate refuses to run
without the manifest and binds the frozen combined-stress triples.

Usage:
  experiments/v48/.venv/bin/python experiments/v58/v58_report.py \
      --manifest experiments/v58/artifacts/frozen_config/v58_frozen_manifest.json \
      --out experiments/v58/artifacts/results/report.jsonl
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v58_verify as verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)

    arms = ["rule_hold", "rule_flee_lead", "rule_safe_top"]
    for name, rec in sorted(manifest["models"].items()):
        arms.append(f"{name}={rec['checkpoint']}")

    cmd = [
        sys.executable, str(HERE / "v58_evaluate.py"),
        "--seeds", "report", "--workers", str(args.workers),
        "--out", args.out, "--require-manifest", args.manifest,
    ]
    for a in arms:
        cmd += ["--arm", a]
    print("running report eval:\n", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-3000:])
    if res.returncode != 0:
        print(res.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"report eval failed ({res.returncode})")
    print(f"[written] {args.out}")
    print(f"[written] {args.out.replace('.jsonl', '.summary.json')}")


if __name__ == "__main__":
    main()
