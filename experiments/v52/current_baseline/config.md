# v52 runbook / config

Frozen constants and exact reproduction commands. The authoritative frozen spec
is `artifacts/v52_spec_manifest.json` (written before scoring, exclusive).

## Constants

- Report bank: `default_rng(520202)`, n=40. Debug bank: `default_rng(5202012)`, n=3.
- Modes: `full`, `mask_predator` (zero `5:11`), `mask_velocity_only` (zero `8:10`).
- Controllers: `rep{0,1,2}_survival_only_final` (PPO, deterministic) + `rule_flee_lead` (control).
- Env: exact v50 config, 96 fish, neighbor off, 11-dim obs, heading/speed bias.
- Bootstrap: 10000, seed 520202, alpha 0.05, episode unit; PPO aggregate averages
  3 models per scene first, then bootstraps.
- Workers: 4, torch threads 1.

## Run tests

    experiments/v48/.venv/bin/python experiments/v52/current_baseline/tests/test_v52_mask.py

## Run formal eval (manifest written first, exclusive)

Full argument set: `--out PATH --manifest PATH --workers 4` (no `--debug`). The
run writes `raw_episodes.jsonl` and `raw_episodes.summary.json`. Always pass a
NEW `--out` to avoid overwriting existing artifacts. `--skip-manifest-write`
reuses an existing spec but first reads it back and rejects any input mismatch
(seed bank, scenarios, modes, controllers, env, model hashes); prefer the plain
exclusive form below.

    OMP_NUM_THREADS=1 experiments/v48/.venv/bin/python \
        experiments/v52/current_baseline/v52_eval.py --workers 4 \
        --manifest experiments/v52/current_baseline/artifacts/v52_spec_manifest.json \
        --out experiments/v52/current_baseline/artifacts/raw_episodes.jsonl

## Analyse (optional reset-only matched-observation diagnostic)

    OMP_NUM_THREADS=1 experiments/v48/.venv/bin/python \
        experiments/v52/current_baseline/v52_analyze.py \
        --raw experiments/v52/current_baseline/artifacts/raw_episodes.jsonl \
        --manifest experiments/v52/current_baseline/artifacts/v52_spec_manifest.json \
        --out experiments/v52/current_baseline/artifacts/analysis.json \
        --matched-obs experiments/v52/current_baseline/artifacts/matched_observation_diagnostic.json

The `--matched-obs` diagnostic resets the frozen scenarios only (no env steps,
no new episodes) and counts deterministic action changes on identical
observations. It is cheap and not a new evaluation.

## Debug estimate

    OMP_NUM_THREADS=1 experiments/v48/.venv/bin/python \
        experiments/v52/current_baseline/v52_eval.py --debug --workers 4 \
        --manifest experiments/v52/current_baseline/artifacts/_debug_spec.json \
        --out experiments/v52/current_baseline/artifacts/_debug_raw.jsonl
