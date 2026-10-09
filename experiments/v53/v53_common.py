"""v53 shared config: one-axis termination-semantics contrast on top of the
accepted v50 corrected survival_only baseline.

Scientific axis (round 5 of 10): with physics, observation, action, reward,
optimizer, budget and every other variable held fixed, does the *episode-end
semantics at step 500* change learning?

  * control  = the accepted v50 corrected `survival_only` runs. They use the
               v50 trainer convention: the focal fish alive at `MAX_TIMESTEPS`
               yields `truncated=True`, so SB3 adds a `gamma * V(terminal_obs)`
               bootstrap term (a continuing / infinite-horizon target).
  * treatment = freshly trained `finite_terminal` runs: the focal alive at
               `MAX_TIMESTEPS` yields `terminated=True` with no bootstrap
               (a genuine finite-horizon terminal value of 0). Death anywhere is
               still terminal with -50; alive steps still reward +0.7.

The control is *reused*, not retrained (its source/seed/model-init hashes were
already independently audited in v50). Only the three treatment runs are new.

This module imports the v50 world config verbatim (physics identical) and only
adds the termination-mode switch plus fresh, disjoint v53 evaluation banks. All
v53 modules are prefixed `v53_` so that v50/v49 modules can never shadow them in
spawned workers.
"""

import hashlib
import importlib.metadata as md
import platform
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

HERE = Path(__file__).resolve().parent
V50_DIR = ROOT_DIR / "experiments" / "v50"
if str(V50_DIR) not in sys.path:
    sys.path.insert(0, str(V50_DIR))

import v50_common as v50  # noqa: E402  (world config source of truth)

NUM_FISH = v50.NUM_FISH
MAX_TIMESTEPS = v50.MAX_TIMESTEPS

REWARD_MODE = "survival_only"              # both arms use this reward
SURVIVAL_STEP_REWARD = v50.SURVIVAL_STEP_REWARD
DEATH_STEP_PENALTY = v50.DEATH_STEP_PENALTY

TERMINATION_MODES = ("timeout_bootstrap", "finite_terminal")

# Fresh v53 evaluation banks, disjoint from every prior bank.
STAGE_SEEDS = {
    "smoke": {"rng_seed": 530100, "n": 2},
    "selection": {"rng_seed": 530101, "n": 24},
    "report": {"rng_seed": 530102, "n": 40},
}

# Per-replicate pairing is inherited from v50: within replicate r the control
# and treatment share the model seed and the continuous worker env bank, so the
# two streams start from the same initial policy AND the same set of starting
# episodes. Replicates are independent.
NUM_ENVS = v50.NUM_ENVS
ENV_BANK_BASE = dict(v50.ENV_BANK_BASE)    # {0:6000011,1:6000021,2:6000031}
MODEL_SEEDS = dict(v50.MODEL_SEEDS)        # {0:7000011,1:7000021,2:7000031}
REPLICATES = tuple(v50.REPLICATES)
ARMS = ("control", "treatment")

# Repo-relative run locations.
CONTROL_RUN_DIR = {
    r: f"experiments/v50/artifacts/corrected_streams/runs/rep{r}_survival_only"
    for r in REPLICATES
}
TREATMENT_RUN_DIR = {
    r: f"experiments/v53/artifacts/runs/rep{r}_finite_terminal"
    for r in REPLICATES
}


def env_config(include_neighbor_features=False):
    """Full v50 env config including predator_heading_bias (v53 binds all fields)."""
    return v50.env_config(include_neighbor_features=include_neighbor_features)


def make_base_env(include_neighbor_features=False):
    return v50.make_base_env(include_neighbor_features=include_neighbor_features)


def episode_seeds(rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def resolve_seeds(name):
    if name not in STAGE_SEEDS:
        raise KeyError(f"unknown seed set: {name}")
    spec = STAGE_SEEDS[name]
    return episode_seeds(spec["rng_seed"], spec["n"])


def env_bank_worker_seeds(replicate, num_envs=NUM_ENVS):
    base = ENV_BANK_BASE[replicate]
    return [base + i for i in range(num_envs)]


def model_seed(replicate):
    return MODEL_SEEDS[replicate]


def dependency_snapshot():
    pkgs = {}
    for name in ("torch", "stable_baselines3", "gymnasium", "numpy", "pygame"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Training-source hashes for a v53 treatment run. `fish_env.py` is the shared
# physics; the v50 modules are imported read-only and their hashes are recorded
# so a treatment run's provenance states exactly which accepted baseline it
# built on. This snapshot is taken at run time and never refreshed afterwards.
def training_source_sha256():
    return {
        "v53_common.py": sha256_file(HERE / "v53_common.py"),
        "v53_env.py": sha256_file(HERE / "v53_env.py"),
        "v53_train.py": sha256_file(HERE / "v53_train.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }
