#!/usr/bin/env python3
"""v50 report orchestrator: build the report arm list from the frozen selection
manifest and run the fresh report evaluation, gated on that manifest.

Arms:
  * rule controls: rule_hold (reference), rule_flee_lead, rule_safe_top;
  * for each replicate and arm: the selection-selected checkpoint
    (`rep{r}_{arm}_selected`);
  * the final checkpoint of each replicate/arm, de-duplicated when it equals the
    selected stage (`rep{r}_{arm}_final`).

The report bank (500202, 40 eps) is disjoint from the selection bank (500201,
24 eps). `v50_evaluate.py` refuses to run if the selection manifest is missing or
if any policy arm's checkpoint path / model identity fails the frozen binding.

Usage:
  experiments/v48/.venv/bin/python experiments/v50/v50_report.py \
      --runs-dir experiments/v50/artifacts/corrected_streams/runs \
      --manifest experiments/v50/artifacts/corrected_streams/results/selection_manifest.json \
      --out experiments/v50/artifacts/corrected_streams/results/report.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v50_verify as verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = verify.load_manifest(args.manifest)
    runs_dir = Path(args.runs_dir)

    arms = [
        "rule_hold",
        "rule_flee_lead",
        "rule_safe_top",
    ]
    for key in sorted(manifest["selected"]):
        sel = manifest["selected"][key]
        arms.append(f"{key}_selected={sel['checkpoint']}")
        # de-duplicate only when the manifest explicitly records selected==final
        # (hash-verified); otherwise the final record must be produced separately.
        if sel["selected_is_final"]:
            continue
        final_ckpt = Path(sel["final_checkpoint"])
        if not final_ckpt.exists():
            raise SystemExit(f"missing final checkpoint: {final_ckpt}")
        arms.append(f"{key}_final={final_ckpt}")

    cmd = [
        sys.executable, str(HERE / "v50_evaluate.py"),
        "--seeds", "report", "--reference", "rule_hold", "--neighbor", "off",
        "--workers", str(args.workers),
        "--out", args.out,
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
