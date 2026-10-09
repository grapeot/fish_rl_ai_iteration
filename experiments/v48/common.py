"""v48 shared config: env factory, stage seed sets, dependency snapshot.

This module deliberately does NOT import experiments/v45/train.py. train.py
pulls in tensorboard/imageio/matplotlib and the full 4800-line training stack;
the eval path only needs FishEscapeEnv plus a faithful re-implementation of the
SingleFishEnv broadcast/round-robin semantics. Keeping eval self-contained
avoids accidental coupling to training-only imports.

Env config mirrors the recorded v45 held-out multi-eval probe (not argparse
defaults): 96 fish, neighbor features on, escape_boost_speed fixed at 0.8,
predator pre-roll + heading/speed bias exactly as checkpoint_sweep.py and
reproduce_eval.py use. This is the distribution under which the documented
report number 0.899 was produced.

Stage seed sets use fresh RNG seeds, disjoint from the historical
selection/report seeds 555001/555002. The legacy report set is still available
behind --set legacy_report purely to reproduce the historical 0.899, never for
decision-making.
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

# Historical held-out seeds. Used only for legacy reproduction, never for
# v48 decisions.
LEGACY_SELECTION_RNG_SEED = 555001
LEGACY_REPORT_RNG_SEED = 555002

# Fresh v48 seeds. Disjoint from the historical ones on purpose: decisions run
# on data the old sweep never touched.
STAGE_SEEDS = {
    "smoke": {"rng_seed": 481000, "n": 2},
    "dev": {"rng_seed": 481001, "n": 12},
    "report": {"rng_seed": 481002, "n": 40},
}

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


def env_config():
    """Return the exact kwargs used to build the evaluation env."""
    return {
        "num_fish": NUM_FISH,
        "include_neighbor_features": True,
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


def make_env():
    return FishEscapeEnv(**env_config())


def episode_seeds(rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def resolve_seeds(name):
    if name == "legacy_report":
        return episode_seeds(LEGACY_REPORT_RNG_SEED, 40)
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
