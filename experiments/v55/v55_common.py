"""v55 shared config: one-axis *post-reset predator-velocity rotation*
generalization probe on the accepted v50 corrected `survival_only` FINAL models
(round 7 of 10).

Scientific axis (original plan: "initial heading coverage"): the frozen policies
may generalize poorly across the predator's initial approach direction. Rather
than resampling the natural initial-heading distribution (which would change the
pre-roll sequence and RNG consumption), we hold ONE nominal world per scenario and
rotate ONLY the post-reset predator velocity vector by a deterministic rotation of
0/90/180/270 degrees. Norm is preserved exactly, positions / fish state / timestep
/ RNG are identical, and the subsequent gravity / collision / physics are
unchanged. 0 degrees must reproduce the nominal bit-for-bit.

This is a NO-TRAINING evaluation round. It uses only the three accepted v50
corrected `survival_only` final checkpoints (200 updates / 614,400 steps each,
merged in PR#11); not the old pilot, not the `original` arm, not v53/v54.

Every v55 module is prefixed `v55_` so it can never shadow, or be shadowed by, the
historical `common.py` / `evaluate.py` / `v50_*` / `v53_*` / `v54_*` modules inside
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

# The post-reset predator-velocity rotation conditions, in degrees. The rotation
# matrices below are EXACT integer matrices (no trig round-off): cos/sin of
# 0/90/180/270 are exactly {0, +-1}. Hence the L2 norm is preserved bit-exactly and
# the 0-degree condition is the exact identity.
CONDITIONS = ("deg0", "deg90", "deg180", "deg270")
ROTATION_DEGREES = {"deg0": 0, "deg90": 90, "deg180": 180, "deg270": 270}
ROTATION_MATRICES = {
    "deg0": ((1, 0), (0, 1)),
    "deg90": ((0, -1), (1, 0)),
    "deg180": ((-1, 0), (0, -1)),
    "deg270": ((0, 1), (-1, 0)),
}

# Fresh v55 evaluation scenario banks, disjoint from every prior bank
# (v48 481xxx/555xxx, v49 482xxx, v50 500xxx, v51 5101xx, v53 530xxx, v54 540xxx).
STAGE_SEEDS = {
    "debug": {"rng_seed": 550101, "n": 2},
    "report": {"rng_seed": 550102, "n": 40},
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

FROZEN_MANIFEST_RELPATH = "experiments/v55/artifacts/frozen_config/v55_frozen_manifest.json"


def env_config(include_neighbor_features=False):
    """Exact v50 world config, incl. the full predator heading/speed bias."""
    return v50.env_config(include_neighbor_features=include_neighbor_features)


def make_base_env(include_neighbor_features=False):
    """The unchanged FishEscapeEnv (all v48 reward terms active)."""
    return v50.make_base_env(include_neighbor_features=include_neighbor_features)


def rotate_predator_velocity(vel, condition):
    """Rotate a 2-vector by the exact integer matrix for `condition`.

    Returns a new float32 array; the input is not modified. The norm is preserved
    exactly because the matrices are exact signed permutations.
    """
    if condition not in ROTATION_MATRICES:
        raise KeyError(f"unknown rotation condition: {condition}")
    (a, b), (c, d) = ROTATION_MATRICES[condition]
    x, y = float(vel[0]), float(vel[1])
    return np.array([a * x + b * y, c * x + d * y], dtype=np.float32)


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


# Run-time source snapshot for the v55 evaluation tooling. `fish_env.py` is the
# shared physics; `v50_common.py`/`v50_env.py` are imported read-only. Recorded so
# a v55 run's provenance states exactly which accepted baseline it built on.
# Never refreshed afterwards (doc edits get a separate post-run snapshot).
def evaluation_source_sha256():
    here = Path(__file__).resolve().parent
    return {
        "v55_common.py": sha256_file(here / "v55_common.py"),
        "v55_env.py": sha256_file(here / "v55_env.py"),
        "v55_evaluate.py": sha256_file(here / "v55_evaluate.py"),
        "v55_verify.py": sha256_file(here / "v55_verify.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }
