"""v57 shared config: one-axis *training-time initial-velocity randomization*
intervention on top of the accepted v50 corrected `survival_only` baseline
(round 9 of 10).

Scientific axis (from the v54 evidence): with physics, observation, action,
reward, optimizer, budget and every other variable held fixed, does randomizing
the fish *initial velocity* at every training episode reset improve
generalization across initial-speed conditions without losing the nominal
condition?

Evidence base: v54 measured the three frozen v50 `survival_only` FINAL policies as
nominal 0.9338 / half 0.9238 / zero 0.8839 (3-policy mean). Their advantage over
same-condition HOLD also narrowed under zero initial velocity. These measurements
motivate testing a post-reset velocity factor drawn uniformly from {1.0, 0.5, 0.0}
at each training reset. Exposure to all three regimes does not guarantee improved
generalization or establish how much survival is caused by initial velocity.

  * control  = the accepted v50 corrected `survival_only` runs (timeout
               bootstrap). REUSED, never retrained: their source/seed/model-init
               hashes were already independently audited in v50 (PR#11).
  * treatment = three freshly trained runs where, after each episode reset, an
               independent augmentation RNG draws factor ~ Uniform{1.0,0.5,0.0}
               and scales ALL fish initial velocities by that factor, then
               recomputes the focal observation. The augmentation RNG is
               independent of the base world RNG (it neither reads nor advances
               it) and is explicit-seed reproducible. It consumes exactly one
               independent draw per reset, so consecutive draws may coincide by
               chance; there is no guarantee that adjacent factors differ.

The ONLY training change is the per-episode velocity augmentation. The wrapper
still fixes the focal id per episode and every other fish HOLDs; reward (+0.7
alive / -50 death), the 500-step timeout-bootstrap horizon, the 11-dim local
observation (neighbor off), the 96-fish world, and PPO hyperparameters are
identical to the v50 control.

All v57 modules are prefixed `v57_` so they can never shadow, or be shadowed by,
the historical `common.py` / `v50_*` / `v53_*` / `v54_*` modules inside spawned
worker processes. `v50_common.py` is imported read-only as the single source of
truth for the world physics/reward config; `v54_common.py` is imported read-only
as the single source of truth for the evaluation condition factors (so the v57
evaluation intervention is exactly the v54-matched one).
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
V50_DIR = ROOT_DIR / "experiments" / "v50"
V54_DIR = ROOT_DIR / "experiments" / "v54"
for _p in (V50_DIR, V54_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v50_common as v50  # noqa: E402  (world config source of truth)
import v54_common as v54  # noqa: E402  (evaluation condition source of truth)

NUM_FISH = v50.NUM_FISH
MAX_TIMESTEPS = v50.MAX_TIMESTEPS

REWARD_MODE = "survival_only"              # both arms use this reward
SURVIVAL_STEP_REWARD = v50.SURVIVAL_STEP_REWARD
DEATH_STEP_PENALTY = v50.DEATH_STEP_PENALTY

# --- training-time augmentation axis ---------------------------------------
# The per-episode factor is drawn uniformly from these values and multiplies
# EVERY fish initial velocity vector after reset (then the focal obs is
# recomputed). 1.0 / 0.5 / 0.0 are all exactly representable in float32.
AUG_FACTORS = (1.0, 0.5, 0.0)
AUG_SEED_OFFSET = 570_000_000  # aug RNG stream is separate from the world RNG

# --- evaluation conditions (exactly the v54 intervention) ------------------
CONDITIONS = v54.CONDITIONS                     # ("nominal", "half", "zero")
VELOCITY_FACTORS = dict(v54.VELOCITY_FACTORS)  # {"nominal":1.0,"half":0.5,"zero":0.0}

# --- fresh v57 banks, disjoint from every prior bank ------------------------
STAGE_SEEDS = {
    "smoke": {"rng_seed": 57010101, "n": 2},
    "selection": {"rng_seed": 57010112, "n": 24},
    "report": {"rng_seed": 57010240, "n": 40},
}

# Per-replicate pairing inherited from v50: the control and treatment share the
# model seed and the continuous worker env bank, so the two arms start from the
# same initial policy AND the same starting world. Replicates are independent.
NUM_ENVS = v50.NUM_ENVS                        # 6
ENV_BANK_BASE = dict(v50.ENV_BANK_BASE)       # {0:6000011,1:6000021,2:6000031}
MODEL_SEEDS = dict(v50.MODEL_SEEDS)           # {0:7000011,1:7000021,2:7000031}
REPLICATES = tuple(v50.REPLICATES)            # (0,1,2)
ARMS = ("control", "treatment")

# Selection pre-registration: only these two stages are considered, for BOTH
# arms (matched), to avoid duplicate compute. 200 is the run's final checkpoint.
SELECTION_STAGES = (100, 200)

# Repo-relative run locations.
CONTROL_RUN_DIR = {
    r: f"experiments/v50/artifacts/corrected_streams/runs/rep{r}_survival_only"
    for r in REPLICATES
}
TREATMENT_RUN_DIR = {
    r: f"experiments/v57/artifacts/runs/rep{r}_augmented_velocity"
    for r in REPLICATES
}

RULE_CONTROLLERS = ("rule_hold", "rule_flee_lead", "rule_safe_top")


def augmentation_seed(worker_seed):
    """Explicit, per-worker augmentation RNG seed. Independent of the world RNG."""
    return int(AUG_SEED_OFFSET) + int(worker_seed)


def env_config(include_neighbor_features=False):
    """Full v50 world config (initial_escape_boost=True, escape_boost_speed=0.8)."""
    return v50.env_config(include_neighbor_features=include_neighbor_features)


def make_base_env(include_neighbor_features=False):
    """The unchanged FishEscapeEnv (all v48 reward terms active)."""
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


HERE = Path(__file__).resolve().parent


def training_source_sha256():
    """Run-time source snapshot for a v57 treatment run. Never refreshed after
    the run. `fish_env.py` is the shared physics; v50/v54 modules are imported
    read-only; v54_env/v54_common are recorded because the evaluation condition
    semantics come from there (training itself does not import them)."""
    return {
        "v57_common.py": sha256_file(HERE / "v57_common.py"),
        "v57_env.py": sha256_file(HERE / "v57_env.py"),
        "v57_train.py": sha256_file(HERE / "v57_train.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }


def evaluation_source_sha256():
    """Run-time source snapshot for the v57 evaluation/selection/report tooling."""
    return {
        "v57_common.py": sha256_file(HERE / "v57_common.py"),
        "v57_env.py": sha256_file(HERE / "v57_env.py"),
        "v57_evaluate.py": sha256_file(HERE / "v57_evaluate.py"),
        "v57_verify.py": sha256_file(HERE / "v57_verify.py"),
        "v54_common.py": sha256_file(V54_DIR / "v54_common.py"),
        "v54_env.py": sha256_file(V54_DIR / "v54_env.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }
