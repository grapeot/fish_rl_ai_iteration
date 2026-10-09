#!/usr/bin/env python3
"""v57 run-time / post-run source snapshot.

Two snapshots are conceptually distinct:

  * run-time training-source hashes are embedded in each treatment run's
    `config.json` / `run_summary.json` (`runtime_source_sha256`) and never
    refreshed;
  * the evaluation/selection/report tooling hashes are embedded in
    `report.summary.json` (`source_sha256`) at report time and never refreshed.

This script records a clearly-labelled post-run snapshot of exactly the modules
that `v57_common.training_source_sha256()` and `v57_common.evaluation_source_sha256()`
cover (common, env, train, evaluate, verify, the read-only v50/v54 modules, and the
shared `fish_env.py`). It does NOT cover the selector, report orchestrator,
analyzer, this snapshot script, the tests, or the dev record; those are captured
only by the external candidate SHA list taken at review time. It exists so a later
doc/tooling edit cannot silently masquerade as run-time provenance for the modules
it does cover.

Usage:
  experiments/v48/.venv/bin/python experiments/v57/v57_snapshot.py \
      --out experiments/v57/artifacts/results/post_run_source_hashes.json
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v57_common as common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "scope": "post-run snapshot; run-time hashes are embedded in the run/result files and never refreshed",
        "note": "v57 round 9/10 trains 3 treatment runs and evaluates frozen models. "
                "This snapshot is for auditing doc/tooling edits after the run.",
        "post_run_training_source_sha256": common.training_source_sha256(),
        "post_run_evaluation_source_sha256": common.evaluation_source_sha256(),
        "fish_env_sha256": common.sha256_file(ROOT / "fish_env.py"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
