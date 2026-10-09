#!/usr/bin/env python3
"""v55 post-run source snapshot.

The run-time source hashes embedded in each result are captured at evaluation
time and never refreshed. This snapshot records the CURRENT hashes of ONLY the
seven evaluation dependencies used by a v55 run (v55 common/env/evaluate/verify,
the read-only v50 common/env, and the shared `fish_env.py`) under an explicit
"post-run" scope.

Scope limit: it does NOT cover the v55 freeze/report/analyze/snapshot tools, the
two test files, or the dev document; those are bound by the final candidate
SHA-256 manifest instead. A later edit to a file outside this seven-file set must
not be described as covered by this snapshot.

Usage:
  experiments/v48/.venv/bin/python experiments/v55/v55_snapshot.py \
      --out experiments/v55/artifacts/results/post_run_source_hashes.json
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

import v55_common as common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "scope": "post-run snapshot; run-time hashes are embedded in the result summary and never refreshed",
        "note": "v55 is a no-training evaluation round; no model weights change. This snapshot is for auditing doc/tooling edits after the run.",
        "post_run_source_sha256": common.evaluation_source_sha256(),
        "fish_env_sha256": common.sha256_file(ROOT / "fish_env.py"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
