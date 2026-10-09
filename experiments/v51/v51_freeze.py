#!/usr/bin/env python3
"""Freeze the v51 full experiment manifest: config, source hashes, model hashes, seeds.

Writes `artifacts/config.json` with:
  * the exact env config used (v49 physics verbatim),
  * the frozen scenario/action seed banks and all derived seeds,
  * the six-model path/sha256/update-count manifest,
  * sha256 of every v51 source file at run time.

This is written before the fresh evaluation is interpreted, so the pinned seeds
and hyperparameters cannot be re-tuned on the result.

Usage:
  experiments/v48/.venv/bin/python experiments/v51/v51_freeze.py \
      [--out experiments/v51/artifacts/repro/config.json]

The default `--out` is the archived `artifacts/config.json`; pass an explicit
`--out` under a new directory for a non-destructive reproduction run.
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

import v51_common as common  # noqa: E402

SOURCE_FILES = [
    "experiments/v51/v51_common.py",
    "experiments/v51/v51_eval.py",
    "experiments/v51/v51_analyze.py",
    "experiments/v51/v51_freeze.py",
    "experiments/v51/tests/test_v51_semantics.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "artifacts" / "config.json"),
                    help="output path; default is the archived artifacts/config.json")
    args = ap.parse_args()

    manifest = common.build_manifest(ROOT)
    scenario_seeds = common.episode_seeds(common.SCENARIO_RNG_SEED, common.N_SCENARIOS)
    debug_seeds = common.episode_seeds(common.DEBUG_SCENARIO_RNG_SEED, common.N_DEBUG_SCENARIOS)
    action_streams = {
        f"ep{ei}_rep{ri}": common.action_stream_seed(ei, ri)
        for ei in range(common.N_SCENARIOS) for ri in common.REPLICATE_INDEX
    }
    payload = {
        "schema": "v51_config_v1",
        "role": "round 3 of 10: deterministic argmax vs stochastic action sampling on frozen v49 policies",
        "training_performed": False,
        "env_config": common.env_config(include_neighbor_features=False),
        "env_config_note": "exact v49 common.env_config; predator_heading_bias included",
        "num_fish_denominator": common.NUM_FISH,
        "max_timesteps": common.MAX_TIMESTEPS,
        "scenario_bank": {
            "fresh_rng_seed": common.SCENARIO_RNG_SEED,
            "n": common.N_SCENARIOS,
            "seeds": scenario_seeds,
        },
        "debug_bank": {
            "rng_seed": common.DEBUG_SCENARIO_RNG_SEED,
            "n": common.N_DEBUG_SCENARIOS,
            "seeds": debug_seeds,
        },
        "action_streams": {
            "master_seed": common.ACTION_STREAM_MASTER_SEED,
            "n_replicates": common.N_ACTION_STREAMS,
            "replicate_index": list(common.REPLICATE_INDEX),
            "derivation": "np.random.SeedSequence([master, episode_index, replicate_index]).generate_state(1, dtype=uint32)[0]",
            "seeds": action_streams,
        },
        "bootstrap": {"n_boot": 10000, "alpha": 0.05,
                      "eval_seed": 510301, "analyze_seed": 510302},
        "models": manifest["models"],
        "source_sha256": {f: common.sha256_file(ROOT / f) for f in SOURCE_FILES},
        "dependency_snapshot": common.dependency_snapshot(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[written] {out}")
    print("source hashes:")
    for f, h in payload["source_sha256"].items():
        print(f"  {h[:16]}  {f}")


if __name__ == "__main__":
    main()
