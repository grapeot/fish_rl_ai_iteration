#!/usr/bin/env python3
"""v58 run-time / post-run source snapshot.

Two snapshots are conceptually distinct:

  * run-time evaluation-source hashes are embedded in `report.summary.json`
    (`source_sha256`) at report time and never refreshed;
  * this script records a clearly-labelled post-run snapshot of exactly the modules
    `v58_common.evaluation_source_sha256()` covers (common, env, evaluate, verify, the
    read-only v50 modules, and the shared `fish_env.py`). It does NOT cover the freeze
    script, report orchestrator, analyzer, this snapshot script, the tests, or the dev
    record; those are captured only by the external candidate hash list.

Usage:
  experiments/v48/.venv/bin/python experiments/v58/v58_snapshot.py \
      --out experiments/v58/artifacts/results/post_run_source_hashes.json
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

import v58_common as common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "scope": "post-run snapshot; run-time hashes are embedded in the result files and never refreshed",
        "note": "v58 round 10/10 is a no-training independent confirmation on the "
                "frozen v57/v50 models. This snapshot is for auditing doc/tooling edits "
                "after the run.",
        "post_run_evaluation_source_sha256": common.evaluation_source_sha256(),
        "fish_env_sha256": common.sha256_file(ROOT / "fish_env.py"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
