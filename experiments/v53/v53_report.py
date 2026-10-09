#!/usr/bin/env python3
"""v53 report orchestrator: build the report arm list from the frozen selection
manifest and run the fresh report eval, gated on that manifest.

Arms:
  * rule controls: rule_hold (reference), rule_flee_lead, rule_safe_top;
  * per run key (rep{r}_control, rep{r}_treatment): the selection-selected
    checkpoint, plus the final checkpoint unless it IS the selected stage
    (hash-verified `selected_is_final`).

Report bank 530102 (40 eps) is disjoint from selection bank 530101 (24 eps).
v53_evaluate refuses to run without the manifest.

Usage:
  experiments/v48/.venv/bin/python experiments/v53/v53_report.py \
      --manifest experiments/v53/artifacts/results/selection_manifest.json \
      --out experiments/v53/artifacts/results/report.jsonl
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

import v53_verify as verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)

    arms = ["rule_hold", "rule_flee_lead", "rule_safe_top"]
    for key in sorted(manifest["selected"]):
        sel = manifest["selected"][key]
        arms.append(f"{key}_selected={sel['checkpoint']}")
        if sel["selected_is_final"]:
            continue
        final_ckpt = Path(sel["final_checkpoint"])
        if not final_ckpt.exists():
            raise SystemExit(f"missing final checkpoint: {final_ckpt}")
        arms.append(f"{key}_final={final_ckpt}")

    cmd = [
        sys.executable, str(HERE / "v53_evaluate.py"),
        "--seeds", "report", "--reference", "rule_hold", "--neighbor", "off",
        "--workers", str(args.workers), "--out", args.out,
        "--require-manifest", args.manifest,
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
