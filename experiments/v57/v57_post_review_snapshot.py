#!/usr/bin/env python3
"""v57 post-review hash scope (separate from the run-time and post-run snapshots).

`post_run_source_hashes.json` (written by `v57_snapshot.py`) covers only the
modules referenced by `v57_common.training_source_sha256()` and
`evaluation_source_sha256()` (common, env, train, evaluate, verify, the read-only
v50/v54 modules, and the shared `fish_env.py`). It does NOT cover the full v57
toolchain.

This script records a clearly-labelled **post-review** snapshot of the COMPLETE
v57 public toolchain: all source modules, both tests, and the dev record. It is
written to its own file so the original run-time and post-run snapshots are
preserved unchanged. It does not refresh any run-time hash embedded in a run or
report artifact.

Usage:
  experiments/v48/.venv/bin/python experiments/v57/v57_post_review_snapshot.py \
      --out experiments/v57/artifacts/results/post_review_source_hashes.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TOOLCHAIN = [
    "v57_common.py",
    "v57_env.py",
    "v57_train.py",
    "v57_evaluate.py",
    "v57_select.py",
    "v57_report.py",
    "v57_analyze.py",
    "v57_verify.py",
    "v57_snapshot.py",
    "v57_post_review_snapshot.py",
    "tests/test_v57_semantics.py",
    "tests/test_v57_acceptance_gates.py",
    "dev_v57.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "scope": "post-review snapshot of the COMPLETE v57 public toolchain "
                 "(all source modules, both tests, dev record). Separate from the "
                 "run-time and post-run snapshots, which are preserved unchanged.",
        "note": "Records current on-disk hashes after the review-driven doc/comment/"
                "gate hardening edits. Run-time hashes embedded in run/report artifacts "
                "are NOT refreshed by this script.",
        "post_review_toolchain_sha256": {
            name: sha256_file(HERE / name) for name in TOOLCHAIN
        },
        "fish_env_sha256": sha256_file(ROOT / "fish_env.py"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
