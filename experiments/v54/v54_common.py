"""v54 shared config: one-axis *post-reset initial-velocity* generalization probe
on the accepted v50 corrected `survival_only` FINAL models (round 6 of 10).

Scientific axis: measure the generalization sensitivity of the frozen v50 FINAL
policies to a post-reset scaling of the fish initial velocity. Scale ONLY the
post-reset fish initial velocities by {1.0, 0.5, 0.0} while holding the rest of
the world fixed, and record how survival changes for the frozen policies and for
the rule anchors.

This is a NO-TRAINING evaluation round. It does not depend on v53. It uses only
the three accepted v50 corrected `survival_only` final checkpoints (200 updates
/ 614,400 steps each, merged in PR#11); not the old pilot, not the `original`
arm, not the unaccepted v53 runs.

Every v54 module is prefixed `v54_` so it can never shadow, or be shadowed by,
the historical `common.py` / `evaluate.py` / `v50_*` / `v53_*` modules inside
spawned worker processes. `v50_common.py` is imported read-only as the single
source of truth for the world physics/reward config (identical to v49/v50).
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
if str(V50_DIR) not in sys.path:
    sys.path.insert(0, str(V50_DIR))

import v50_common as v50  # noqa: E402  (world config source of truth)

NUM_FISH = v50.NUM_FISH
MAX_TIMESTEPS = v50.MAX_TIMESTEPS

# The post-reset velocity conditions. The factor multiplies the fish initial
# velocity vector exactly (1.0 / 0.5 / 0.0 are all exactly representable).
CONDITIONS = ("nominal", "half", "zero")
VELOCITY_FACTORS = {"nominal": 1.0, "half": 0.5, "zero": 0.0}

# Fresh v54 evaluation scenario banks, disjoint from every prior bank.
STAGE_SEEDS = {
    "debug": {"rng_seed": 540101, "n": 2},
    "report": {"rng_seed": 540102, "n": 40},
}

# Controllers: the three frozen PPO FINAL policies + three rule anchors.
PPO_RUNS = {r: f"rep{r}_survival_only" for r in (0, 1, 2)}
PPO_CHECKPOINT_RELPATH = {
    r: (
        f"experiments/v50/artifacts/corrected_streams/runs/"
        f"rep{r}_survival_only/checkpoints/model_final.zip"
    )
    for r in (0, 1, 2)
}
PPO_CONTROLLERS = {f"rep{r}_ppo": PPO_CHECKPOINT_RELPATH[r] for r in (0, 1, 2)}
RULE_CONTROLLERS = ("rule_hold", "rule_flee_lead", "rule_safe_top")
CONTROLLERS = tuple(PPO_CONTROLLERS) + RULE_CONTROLLERS
PPO_CONTROLLER_NAMES = tuple(PPO_CONTROLLERS)

FROZEN_MANIFEST_RELPATH = "experiments/v54/artifacts/frozen_config/v54_frozen_manifest.json"


def env_config(include_neighbor_features=False):
    """Exact v50 world config (initial_escape_boost=True, escape_boost_speed=0.8,
    i.e. a nominal boost speed FISH_MAX_SPEED*0.8 = 1.6)."""
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


# Run-time source snapshot for the v54 evaluation tooling. `fish_env.py` is the
# shared physics; `v50_common.py`/`v50_env.py` are imported read-only. Recorded
# so a v54 run's provenance states exactly which accepted baseline it built on.
# Never refreshed afterwards (doc edits get a separate post-run snapshot).
def evaluation_source_sha256():
    here = Path(__file__).resolve().parent
    return {
        "v54_common.py": sha256_file(here / "v54_common.py"),
        "v54_env.py": sha256_file(here / "v54_env.py"),
        "v54_evaluate.py": sha256_file(here / "v54_evaluate.py"),
        "v54_verify.py": sha256_file(here / "v54_verify.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }
