"""v50 shared config: env factory (physics/reward identical to v49), frozen seed
banks, per-replicate env banks and model seeds, reward modes.

Provenance: the world/physics/reward kwargs are copied verbatim from v49
(`experiments/v49/common.py`), which in turn mirrors v48's evaluation config and
the historical held-out multi-eval probe. The only v50 additions are:

  * `reward_mode` forwarded into the single-agent wrapper (see v50_env.py). The
    underlying FishEscapeEnv is always built with the unchanged v48 reward terms
    (survival +2, distance, boundary, density 0.05, scale 0.1, death -50); the
    `survival_only` mode discards the env reward vector at the wrapper boundary
    and feeds PPO only +0.7 per alive step and -50 on the death step. Physics,
    observation and termination are identical in both modes.
  * explicit, non-overlapping env seed banks and model seeds per replicate, plus
    the fixed selection (500101, 24 eps) and report (500102, 40 eps) episode
    banks. All are frozen before any v50 result is seen and are disjoint from
    v48 (481xxx / 555xxx), v49 (482xxx) and the v49 training worker seeds
    (4901001-4901008).

Module naming: every v50 module is prefixed `v50_` so that the historical
`common.py` / `evaluate.py` modules from v48/v49 can never shadow or be shadowed
by v50 code inside spawned worker processes.
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

from fish_env import FishEscapeEnv  # noqa: E402

NUM_FISH = 96
MAX_TIMESTEPS = 500

REWARD_MODES = ("original", "survival_only")
SURVIVAL_STEP_REWARD = 0.7
DEATH_STEP_PENALTY = -50.0

# Frozen v50 evaluation episode banks. Disjoint from v48 481xxx / 555xxx,
# v49 482xxx, and the pilot v50 500101/500102 banks (verified at the
# generated-value level; see tests/test_v50_semantics.py).
STAGE_SEEDS = {
    "smoke": {"rng_seed": 500200, "n": 2},
    "selection": {"rng_seed": 500201, "n": 24},
    "report": {"rng_seed": 500202, "n": 40},
}

# Per-replicate training design. Both reward arms share the SAME env bank and
# the SAME model seed within a replicate (paired control); different replicates
# use disjoint banks and independent model seeds. Worker seeds are
# `bank_base .. bank_base + (num_envs - 1)`.
NUM_ENVS = 6
ENV_BANK_BASE = {0: 6000011, 1: 6000021, 2: 6000031}
MODEL_SEEDS = {0: 7000011, 1: 7000021, 2: 7000031}
REPLICATES = (0, 1, 2)
ARMS = ("original", "survival_only")

PREDATOR_HEADING_BIAS_SPEC = (
    "0-60:1.1,60-90:0.8,90-120:0.08,120-150:0.05,"
    "150-210:0.9,210-270:1.3,270-330:1.2,330-360:0.95"
)
PREDATOR_SPEED_BIAS_SPEC = "0-1.0:1.6,1.0-1.6:1.3,1.6-2.0:0.7,2.0-2.4:0.35,2.4-3.0:0.2"


def parse_heading_bias(spec):
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        rng, weight = chunk.split(":")
        start, end = rng.split("-")
        out.append({"start_deg": float(start), "end_deg": float(end), "weight": float(weight)})
    return out


def parse_speed_bias(spec):
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        rng, weight = chunk.split(":")
        lo, hi = rng.split("-")
        out.append({"min_speed": float(lo), "max_speed": float(hi), "weight": float(weight)})
    return out


def env_config(include_neighbor_features=False):
    """Exact v49 evaluation kwargs. Identical physics and reward constants."""
    return {
        "num_fish": NUM_FISH,
        "include_neighbor_features": include_neighbor_features,
        "neighbor_radius": 3.0,
        "neighbor_average_count": 6,
        "initial_escape_boost": True,
        "escape_boost_speed": 0.8,
        "escape_jitter_std": 0.35,
        "divergence_reward_coef": 0.0,
        "density_penalty_coef": 0.05,
        "density_target": 0.4,
        "predator_spawn_jitter_radius": 1.6,
        "predator_pre_roll_steps": 16,
        "predator_pre_roll_angle_jitter": 0.3,
        "predator_pre_roll_speed_jitter": 0.2,
        "predator_heading_bias": parse_heading_bias(PREDATOR_HEADING_BIAS_SPEC),
        "predator_pre_roll_speed_bias": parse_speed_bias(PREDATOR_SPEED_BIAS_SPEC),
    }


def make_base_env(include_neighbor_features=False):
    """The unchanged FishEscapeEnv (all v48 reward terms active)."""
    return FishEscapeEnv(**env_config(include_neighbor_features=include_neighbor_features))


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


def config_hash(payload):
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:16]
