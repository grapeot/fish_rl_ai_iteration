"""v56 shared config: one-axis *post-reset predator-speed scale* generalization
probe on the accepted v50 corrected `survival_only` FINAL models (round 8 of 10).

Scientific axis (plan row 8: "predator speed robustness"): the frozen policies may
generalize poorly when the post-reset predator speed is scaled up or down. Rather
than change the initial-draw distribution or the pre-roll RNG sequence, we hold ONE
nominal world per scenario and scale ONLY the post-reset predator velocity vector by
a deterministic factor of {0.75, 1.0, 1.25}. The direction of the predator velocity
is preserved exactly (the factor is a positive scalar), the resulting speed is scaled
proportionally. Fish positions/velocities/alive, predator position, timestep, the
pre-roll trace and the env RNG stream are restored bit-identically, and observations
are recomputed. Subsequent gravity / bounce / collision physics are unchanged; the
velocity itself will evolve differently because its initial magnitude differs.

Identity / norm guarantee: the factor 1.0 condition is the exact scalar identity, so
it reproduces the raw nominal post-reset state and rollout bit-for-bit. The factors
0.75 and 1.25 are exactly representable in float32 and act as a positive scalar on the
velocity vector, so the direction (unit vector) is preserved and the norm scales by
the factor up to the ordinary float32 rounding of the product. Exercised by the
semantic tests, not assumed.

This is explicitly NOT a whole-episode constant-speed change and NOT a change of the
predator's maximum speed; it is the post-reset initial velocity magnitude. It is also
NOT a resample of the natural initial-speed distribution (which would change the
pre-roll draw sequence and RNG consumption).

Zero-norm guard: if the nominal post-reset predator velocity norm is ~0 (rare), the
scale is undefined. We do NOT divide by zero: the condition keeps velocity 0 and sets
a ``zero_predator_norm`` record flag. Exercised by the semantic tests.

This is a NO-TRAINING evaluation round. It uses only the three accepted v50 corrected
`survival_only` final checkpoints (200 updates / 614,400 steps each, merged in PR#11);
not the old pilot, not the `original` arm, not v53/v54/v55.

Every v56 module is prefixed `v56_` so it can never shadow, or be shadowed by, the
historical `common.py` / `evaluate.py` / `v50_*` / `v53_*` / `v54_*` / `v55_*` modules
inside spawned worker processes. `v50_common.py` is imported read-only as the single
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

# The post-reset predator-speed scale conditions. The factor multiplies the predator
# velocity vector exactly; 1.0 is the exact identity. 0.75 and 1.25 are exactly
# representable in float32.
CONDITIONS = ("scale0p75", "scale1p00", "scale1p25")
SPEED_FACTORS = {"scale0p75": 0.75, "scale1p00": 1.0, "scale1p25": 1.25}
# The condition that is the exact identity; used as the paired baseline.
BASE_CONDITION = "scale1p00"

# Fresh v56 evaluation scenario banks, disjoint from every prior bank
# (v48 481xxx/555xxx, v49 482xxx, v50 500xxx, v51 5101xx, v53 530xxx,
#  v54 540xxx, v55 550xxx).
STAGE_SEEDS = {
    "debug": {"rng_seed": 560101, "n": 2},
    "report": {"rng_seed": 560102, "n": 40},
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

FROZEN_MANIFEST_RELPATH = "experiments/v56/artifacts/frozen_config/v56_frozen_manifest.json"


def env_config(include_neighbor_features=False):
    """Exact v50 world config, incl. the full predator heading/speed bias."""
    return v50.env_config(include_neighbor_features=include_neighbor_features)


def make_base_env(include_neighbor_features=False):
    """The unchanged FishEscapeEnv (all v48 reward terms active)."""
    return v50.make_base_env(include_neighbor_features=include_neighbor_features)


def scale_predator_velocity(vel, factor):
    """Return ``vel`` scaled by the positive scalar ``factor``, direction preserved.

    Returns a new float32 array; the input is not modified. ``factor`` must be a
    finite positive float. If the input norm is ~0 the scale is undefined; the caller
    (``v56_env.apply_condition``) handles that case by keeping the zero vector and
    flagging it, so this helper never divides by zero.
    """
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"predator speed factor must be finite and positive: {factor}")
    return (np.asarray(vel, dtype=np.float32) * np.float32(factor)).astype(np.float32)


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


# Run-time source snapshot for the v56 evaluation tooling. `fish_env.py` is the shared
# physics; `v50_common.py`/`v50_env.py` are imported read-only. Recorded so a v56 run's
# provenance states exactly which accepted baseline it built on. Never refreshed
# afterwards (doc edits get a separate post-run snapshot).
def evaluation_source_sha256():
    here = Path(__file__).resolve().parent
    return {
        "v56_common.py": sha256_file(here / "v56_common.py"),
        "v56_env.py": sha256_file(here / "v56_env.py"),
        "v56_evaluate.py": sha256_file(here / "v56_evaluate.py"),
        "v56_verify.py": sha256_file(here / "v56_verify.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }
