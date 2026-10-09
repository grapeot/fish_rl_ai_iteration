"""v49 shared config: env factory (identical physics/reward to v48), fresh seed banks.

Provenance: env kwargs are copied verbatim from the recorded v48 evaluation
config (`experiments/v48/common.py`) which itself mirrors the historical
held-out multi-eval probe of the v45 baseline. The world physics, action space,
spawn and per-fish reward terms are unchanged. The only v49 addition is a
`include_neighbor_features` switch so the training wrapper can expose the 11-dim
own-observation while keeping the reward computation (including the 0.05 density
penalty) identical.

Fresh seed banks here are disjoint from v48's (481000/481001/481002) and the
legacy ones (555001/555002); the generated episode seeds were verified to have
no overlap. They were fixed in code before training and are not re-tuned on the
report set. Note the selection/report split was NOT independently pre-registered:
the report set was generated before the selection set (see dev_v49.md and
artifacts/metadata_notes.json). The split is a post-hoc descriptive device, not
an untouched test set.
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

# Frozen v49 seed banks (disjoint from v48 481xxx and legacy 555xxx).
STAGE_SEEDS = {
    "smoke": {"rng_seed": 482100, "n": 2},
    "selection": {"rng_seed": 482101, "n": 20},
    "report": {"rng_seed": 482102, "n": 40},
}

# Independent training seeds, one per run. Fixed before training.
TRAIN_SEEDS = [4901001, 4901002, 4901003]

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
    """Exact v48 evaluation kwargs, with the neighbor-exposure flag configurable.

    The neighbor flag only changes what the observation exposes; the density
    penalty reward uses `_collect_neighbor_stats` independently of this flag, so
    the per-fish reward and all dynamics are unchanged from v48.
    """
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


def make_env(include_neighbor_features=False):
    return FishEscapeEnv(**env_config(include_neighbor_features=include_neighbor_features))


def episode_seeds(rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def resolve_seeds(name):
    if name not in STAGE_SEEDS:
        raise KeyError(f"unknown seed set: {name}")
    spec = STAGE_SEEDS[name]
    return episode_seeds(spec["rng_seed"], spec["n"])


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
