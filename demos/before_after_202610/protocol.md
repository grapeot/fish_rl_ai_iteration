# Before/after same-scene benchmark — frozen protocol

Created (UTC): 2026-10-10T15:26:22.711645+00:00
Protocol binding SHA-256: `207315857dc322fbfd6fd2b0b87147f0898b98e72920fc29cf1bf746851b0d1c`

This protocol was written and hashed **before any benchmark score was computed**.
Physics, reward, termination and reset are the unchanged v50 world
(`experiments/v50/v50_common.env_config`); this is not a re-training and no
checkpoint is re-selected on any demo result.

## Question

On one fresh matched scenario bank, compared against the historical best single
checkpoint, how does the current three-model method perform end to end, and where
does the accepted hand-written evasion rule sit? This is a whole-method
comparison on identical physical resets. It is not an isolated training-bug fix,
not a pure-reward ablation, and the old side is a single representative checkpoint
(not three runs).

## Controllers (all deterministic)

| name | kind | obs dim | neighbor | checkpoint | policy-tensor SHA-256 |
|---|---|---|---|---|---|
| old_ppo | policy | 18 | on | experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip | 96b03c31fbfbdfa2... |
| new_rep0 | policy | 11 | off | experiments/v50/artifacts/corrected_streams/runs/rep0_survival_only/checkpoints/model_final.zip | aac9610481879e56... |
| new_rep1 | policy | 11 | off | experiments/v50/artifacts/corrected_streams/runs/rep1_survival_only/checkpoints/model_final.zip | 6aeb33ee08a715a8... |
| new_rep2 | policy | 11 | off | experiments/v50/artifacts/corrected_streams/runs/rep2_survival_only/checkpoints/model_final.zip | f77a0ec8eba6ed3d... |
| rule_flee_lead | rule | 11 | off | (rule logic) | (rule) |

`old_ppo` uses `include_neighbor_features=True` only so its observation matches
its own 18-dim input; the three NEW models and the rule run at 11 dims. This flag
changes the observation vector length and nothing else in the simulation.

NEW model hashes were checked against the `final_policy_tensor_sha256` fields of
`experiments/v50/artifacts/corrected_streams/results/selection_manifest.json` (all match; recorded in the binding).

## Scenes

- report bank: `default_rng(590102)`, 64 scenarios
- debug bank: `default_rng(590101)`, 2 scenarios

Each scene is one shared world: the same seed drives every controller, and the
simulation RNG / positions / velocities / collision physics are identical across
the neighbor-on and neighbor-off envs (asserted in `initial_state_identity.py`
before scoring).

## Runs

`5 controllers x 64 scenarios = 320 episodes`
on the report bank, plus a 2-scene debug bank. Survival is recorded as
`num_alive / 96` at steps [1, 100, 250, 500]; the denominator is always 96, no
episode is dropped, and a first-step death is kept.

## Statistics

- per-controller mean final survival over the 64 scenarios;
- `new_fixed3_mean` = per-scene mean of the three NEW replicates;
- paired contrast: per-scene `new_fixed3_mean - old`, then scenario bootstrap
  (10000) 95% CI; the same for `rule - old` and `rule - new_fixed3_mean`.
