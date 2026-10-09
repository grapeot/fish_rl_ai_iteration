#!/usr/bin/env python3
"""v56 report orchestrator: build the controller arm list from the frozen manifest
and run the fresh report evaluation, gated on that manifest.

Controllers: the three frozen v50 corrected `survival_only` FINAL policies +
rule_hold / rule_flee_lead / rule_safe_top. Report bank 560102 (40 scenarios) is
disjoint from the debug bank 560101 (2).

Usage:
  experiments/v48/.venv/bin/python experiments/v56/v56_report.py \
      --manifest experiments/v56/artifacts/frozen_config/v56_frozen_manifest.json \
      --out experiments/v56/artifacts/results/report.jsonl
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

import v56_verify as verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="report")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)

    arms = []
    for name in sorted(manifest["controllers"]):
        path = Path(ROOT / manifest["controllers"][name]["checkpoint"])
        if not path.exists():
            raise SystemExit(f"missing frozen checkpoint: {path}")
        arms.append(f"{name}={path}")
    arms += list(manifest["rule_controllers"])

    cmd = [
        sys.executable, str(HERE / "v56_evaluate.py"),
        "--seeds", args.seeds, "--workers", str(args.workers),
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
