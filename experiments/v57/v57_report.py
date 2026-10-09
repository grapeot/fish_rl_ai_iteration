#!/usr/bin/env python3
"""v57 report orchestrator: build the report arm list from the frozen manifest and
run the fresh report eval, gated on that manifest.

Report arms (exactly the pre-registered set):
  * rule controls: rule_hold (reference), rule_flee_lead, rule_safe_top;
  * for each run key (rep{r}_control, rep{r}_treatment): the selection-selected
    checkpoint only. Non-selected finals are NOT reported (compute is saved; the
    report table is never fabricated from unrun models).

6 selected models + 3 rules = 9 controllers x 3 conditions x 40 = 1080 episodes.
Report bank 57010240 (40 eps) is disjoint from the selection bank 57010112 (24).
v57_evaluate refuses to run without the manifest.

Usage:
  experiments/v48/.venv/bin/python experiments/v57/v57_report.py \
      --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json \
      --out experiments/v57/artifacts/results/report.jsonl
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

import v57_verify as verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)

    arms = ["rule_hold", "rule_flee_lead", "rule_safe_top"]
    for key in sorted(manifest["runs"]):
        sel = manifest["runs"][key]["selected"]
        arms.append(f"{key}_selected={sel['checkpoint']}")

    cmd = [
        sys.executable, str(HERE / "v57_evaluate.py"),
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
