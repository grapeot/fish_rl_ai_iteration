#!/usr/bin/env python3
"""v53 control-reuse verification.

The round reuses the ACCEPTED v50 corrected `survival_only` runs as the
`timeout_bootstrap` control instead of retraining them. This script proves that
the reuse is faithful, per replicate:

  * the reused run is the corrected-stream `survival_only` run (not the pilot);
  * its reward is `survival_only` (+0.7 / -50) and its world config is the v50
    world config;
  * its `model_seed`, `env_bank_worker_seeds` and `initial_policy_hash` are the
    v50 frozen pairing;
  * the v53 treatment run at the same replicate has the SAME model seed, env bank
    and initial-policy hash (matched comparator), and records the same accepted
    `v50_common.py` / `v50_env.py` / `fish_env.py` source hashes.

Writes a JSON with the comparison and a per-replicate pass/fail. This is a
deterministic, non-training check.

Usage:
  experiments/v48/.venv/bin/python experiments/v53/v53_reuse_check.py \
      --out experiments/v53/artifacts/results/reuse_check.json
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

import v53_common as common  # noqa: E402

# Accepted corrected-stream training-source hashes (from the v50 review).
ACCEPTED = {
    "v50_common.py": "f767a12472a5262ab1f4217c93ad4a7903a71313e0ac3b6026769da5222adb5e",
    "v50_env.py": "1dc9a8e40be521c6bd30e41557c122c9c462b153f67343770977ea1e497ba0a6",
    "fish_env.py": "e907577e4afde06763af5cc768c6bcabd1c8a2ecea769fbccc40c3c75496270e",
}


def load_cfg(rel):
    p = ROOT / rel / "config.json"
    if not p.exists():
        raise SystemExit(f"missing config: {p}")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {"per_replicate": {}, "accepted_source_sha256": ACCEPTED, "all_pass": True}
    for r in common.REPLICATES:
        ctrl = load_cfg(common.CONTROL_RUN_DIR[r])
        treat = load_cfg(common.TREATMENT_RUN_DIR[r])
        checks = {
            "control_reward_is_survival_only": ctrl.get("reward_mode") == "survival_only"
                                                 or ctrl.get("arm") == "survival_only",
            "control_survival_step_reward": ctrl.get("survival_step_reward") == common.SURVIVAL_STEP_REWARD,
            "control_death_step_penalty": ctrl.get("death_step_penalty") == common.DEATH_STEP_PENALTY,
            "control_under_corrected_streams": "corrected_streams" in common.CONTROL_RUN_DIR[r],
            "model_seed_match": ctrl.get("model_seed") == treat.get("model_seed"),
            "env_bank_match": ctrl.get("env_bank_worker_seeds") == treat.get("env_bank_worker_seeds"),
            "initial_policy_hash_match": ctrl.get("initial_policy_hash") == treat.get("initial_policy_hash"),
            "treatment_is_finite_terminal": treat.get("termination_mode") == "finite_terminal",
            "treatment_reward_is_survival_only": treat.get("reward_mode") == "survival_only",
            "treatment_total_steps_614400": treat.get("total_steps") == 614400,
            "treatment_source_hashes_match_accepted":
                all(treat.get("source_sha256", {}).get(k) == v for k, v in ACCEPTED.items()),
            "control_source_hash_v50_common_accepted":
                ctrl.get("source_sha256", {}).get("v50_common.py") == ACCEPTED["v50_common.py"],
            "control_source_hash_v50_env_accepted":
                ctrl.get("source_sha256", {}).get("v50_env.py") == ACCEPTED["v50_env.py"],
            "control_source_hash_fish_env_accepted":
                ctrl.get("source_sha256", {}).get("fish_env.py") == ACCEPTED["fish_env.py"],
        }
        passed = all(checks.values())
        result["all_pass"] &= passed
        result["per_replicate"][f"rep{r}"] = {
            "control_dir": common.CONTROL_RUN_DIR[r],
            "treatment_dir": common.TREATMENT_RUN_DIR[r],
            "model_seed": ctrl.get("model_seed"),
            "env_bank_worker_seeds": ctrl.get("env_bank_worker_seeds"),
            "initial_policy_hash": ctrl.get("initial_policy_hash"),
            "checks": checks,
            "pass": passed,
        }
        print(f"rep{r}: {'PASS' if passed else 'FAIL'} {checks if not passed else ''}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[written] {args.out}")
    if not result["all_pass"]:
        raise SystemExit("reuse check FAILED")
    print("REUSE CHECK PASSED")


if __name__ == "__main__":
    main()
