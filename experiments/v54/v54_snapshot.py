#!/usr/bin/env python3
"""v54 post-run source snapshot.

The run-time source hashes embedded in each result are captured at evaluation
time and never refreshed. Because v54 is a no-training round, the doc/tooling
files may be corrected after the run; this script records the CURRENT hashes of
the v54 tooling plus the shared `fish_env.py` and the read-only v50 modules, with
an explicit "post-run" scope, so a later doc edit cannot silently masquerade as
run-time provenance.

Usage:
  experiments/v48/.venv/bin/python experiments/v54/v54_snapshot.py \
      --out experiments/v54/artifacts/results/post_run_source_hashes.json
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

import v54_common as common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "scope": "post-run snapshot; run-time hashes are embedded in the result summary and never refreshed",
        "note": "v54 is a no-training evaluation round; no model weights change. This snapshot is for auditing doc/tooling edits after the run.",
        "post_run_source_sha256": common.evaluation_source_sha256(),
        "fish_env_sha256": common.sha256_file(ROOT / "fish_env.py"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
