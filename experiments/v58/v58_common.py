"""v58 shared config: a fresh-scene confirmation of the round-9 training-time
initial-velocity randomization, evaluated on unseen scenes under a finite
joint-stress distribution (round 10 of 10).

Design: the complete criteria, model list, concrete scenes and joint-stress arrays
are recorded in the public frozen manifest
`experiments/v58/artifacts/frozen_config/v58_frozen_manifest.json`. This module and
that manifest are self-contained; no external plan file is a dependency.

  * Inputs are the v57 frozen speed-balanced selection manifest: six selected
    policies, three reused nominal controls (`rep{r}_control`, the accepted v50
    corrected `survival_only` checkpoints) and three speed-randomized treatments
    (`rep{r}_treatment`, the v57 augmented-velocity checkpoints). All six are
    reused as-is; nothing is retrained and no checkpoint is reselected on v58
    results.
  * Scenes are 64 fresh base-world seeds from `default_rng(580102)` (debug bank
    `580101`, 2 scenes), validated disjoint from every earlier bank at the
    generated-value level.
  * Two CONDITIONS per scene:
      - `nominal`         : post-reset fish velocity factor 1.0, predator heading
                            rotation 0 deg, predator speed factor 1.0 (the exact
                            identity: reproduces the raw nominal rollout);
      - `combined_stress` : a single pre-specified finite joint stress draw per
                            scene, shared by every controller.
  * The combined-stress draw is one triple per scene from an independent
    augmentation RNG `default_rng(580202)`:
      fish factor ~ Uniform{0.0, 0.5, 1.0},
      predator heading rotation ~ Uniform{0, 90, 180, 270} degrees,
      predator speed factor ~ Uniform{0.75, 1.0, 1.25},
    each uniformly and independently. The arrays are drawn and frozen before any
    score is computed. A scene whose triple equals the nominal identity
    (1.0, 0, 1.0) is kept as drawn, never redrawn. This is a finite discrete
    artificial joint stress distribution, not a natural deployment distribution and
    not full continuous-domain coverage.
  * The intervention changes ONLY post-reset velocity vectors: the fish initial
    velocities are scaled, the predator initial velocity is rotated then scaled; all
    other state (positions, alive, death timesteps, predator position, timestep,
    pre-roll trace) and the env RNG stream are restored bit-identically. Subsequent
    gravity/bounce physics are unchanged.

This is NOT training. It reuses the accepted v57/v50 models and the accepted world
physics (`v50_common`). Every v58 module is prefixed `v58_` so it can never shadow,
or be shadowed by, the historical `common.py` / `v50_*` / `v53_*` / `v54_*` /
`v55_*` / `v56_*` / `v57_*` modules inside spawned worker processes.
"""

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
V50_DIR = ROOT_DIR / "experiments" / "v50"
V57_DIR = ROOT_DIR / "experiments" / "v57"
for _p in (V50_DIR, V57_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v50_common as v50  # noqa: E402  (world config source of truth)

NUM_FISH = v50.NUM_FISH
MAX_TIMESTEPS = v50.MAX_TIMESTEPS

# --- evaluation conditions ---------------------------------------------------
CONDITIONS = ("nominal", "combined_stress")

# Combined-stress components. All values are exactly representable in float32.
FISH_FACTORS = (0.0, 0.5, 1.0)
ROTATION_DEGREES = (0, 90, 180, 270)
SPEED_FACTORS = (0.75, 1.0, 1.25)

# The nominal identity triple; the nominal condition applies exactly this.
NOMINAL_TRIPLE = (1.0, 0, 1.0)

# Exact integer rotation matrices (cos/sin of 0/90/180/270 are exactly {0,+-1}).
ROTATION_MATRICES = {
    0: ((1, 0), (0, 1)),
    90: ((0, -1), (1, 0)),
    180: ((-1, 0), (0, -1)),
    270: ((0, 1), (-1, 0)),
}

# --- fresh v58 banks, disjoint from every prior bank -------------------------
STAGE_SEEDS = {
    "debug": {"rng_seed": 580101, "n": 2},
    "report": {"rng_seed": 580102, "n": 64},
}
# Independent augmentation RNG for the frozen combined-stress draws.
AUG_RNG_SEED = 580202

# --- frozen six selected models from the accepted v57 manifest ---------------
REPLICATES = (0, 1, 2)
CONTROL_RUN_KEY = {r: f"rep{r}_control" for r in REPLICATES}
TREATMENT_RUN_KEY = {r: f"rep{r}_treatment" for r in REPLICATES}
CONTROL_MODEL_NAME = {r: f"rep{r}_control_selected" for r in REPLICATES}
TREATMENT_MODEL_NAME = {r: f"rep{r}_treatment_selected" for r in REPLICATES}
PPO_CONTROLLER_NAMES = tuple(
    [CONTROL_MODEL_NAME[r] for r in REPLICATES] + [TREATMENT_MODEL_NAME[r] for r in REPLICATES]
)

RULE_CONTROLLERS = ("rule_hold", "rule_flee_lead", "rule_safe_top")
CONTROLLERS = PPO_CONTROLLER_NAMES + RULE_CONTROLLERS

V57_MANIFEST_RELPATH = "experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json"

# Recommendation thresholds (pre-declared engineering tolerances, NOT discovered
# statistical constants).
NOMINAL_LOSS_TOLERANCE = 0.01          # nominal mean loss must be < 1 pp
NOMINAL_CI_LOWER_FLOOR = -0.02        # nominal 95% CI lower endpoint no worse than -2 pp


def env_config(include_neighbor_features=False):
    """Exact v50 world config (physics/reward identical to v49/v50/v57)."""
    return v50.env_config(include_neighbor_features=include_neighbor_features)


def make_base_env(include_neighbor_features=False):
    """The unchanged FishEscapeEnv (all v48 reward terms active; not used here)."""
    return v50.make_base_env(include_neighbor_features=include_neighbor_features)


def episode_seeds(rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def resolve_seeds(name):
    if name not in STAGE_SEEDS:
        raise KeyError(f"unknown seed set: {name}")
    spec = STAGE_SEEDS[name]
    return episode_seeds(spec["rng_seed"], spec["n"])


def draw_combined_stress_triples(seeds, rng_seed=AUG_RNG_SEED):
    """Draw one frozen combined-stress triple per scene (in seed-list order).

    fish factor ~ Uniform{0.0,0.5,1.0}; heading rotation ~ Uniform{0,90,180,270};
    speed factor ~ Uniform{0.75,1.0,1.25}; each uniformly and independently, one
    triple per scene, stored as exact Python types. Drawn ONCE and frozen before any
    score is computed; all controllers share the same per-scene triple.
    """
    rng = np.random.default_rng(int(rng_seed))
    triples = []
    for _ in seeds:
        fish = float(FISH_FACTORS[int(rng.integers(0, len(FISH_FACTORS)))])
        rot = int(ROTATION_DEGREES[int(rng.integers(0, len(ROTATION_DEGREES)))])
        speed = float(SPEED_FACTORS[int(rng.integers(0, len(SPEED_FACTORS)))])
        triples.append([fish, rot, speed])
    return triples


def rotate_predator_velocity(vel, deg):
    """Rotate a 2-vector by the exact integer matrix for `deg` (0/90/180/270)."""
    if int(deg) not in ROTATION_MATRICES:
        raise KeyError(f"unknown rotation: {deg}")
    (a, b), (c, d) = ROTATION_MATRICES[int(deg)]
    x, y = float(vel[0]), float(vel[1])
    return np.array([a * x + b * y, c * x + d * y], dtype=np.float32)


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


def load_v57_manifest():
    p = ROOT_DIR / V57_MANIFEST_RELPATH
    if not p.exists():
        raise FileNotFoundError(f"accepted v57 manifest not found: {p}")
    return json.loads(p.read_text())


def frozen_models_from_v57():
    """The six frozen selected (path, policy_tensor_sha256) pairs from the accepted
    v57 manifest. Returns {model_name: {path, policy_tensor_sha256, run_key, arm}}."""
    man = load_v57_manifest()
    out = {}
    for r in REPLICATES:
        for run_key, arm, name in (
            (CONTROL_RUN_KEY[r], "control", CONTROL_MODEL_NAME[r]),
            (TREATMENT_RUN_KEY[r], "treatment", TREATMENT_MODEL_NAME[r]),
        ):
            rec = man["runs"][run_key]["selected"]
            out[name] = {
                "run_key": run_key,
                "arm": arm,
                "replicate": r,
                "path": rec["checkpoint"],
                "policy_tensor_sha256": rec["policy_tensor_sha256"],
                "update": rec["update"],
            }
    return out


def evaluation_source_sha256():
    """Source hashes for the v58 evaluation tooling. `fish_env.py` is the shared
    physics; `v50_common.py`/`v50_env.py` are imported read-only.

    Scope: these seven files are the code actually exercised on the scoring path.
    They are written into `report.summary.json` when the report completes, so they
    are a run-time snapshot, not a pre-freeze whole-toolchain hash asset. They do not
    cover the freeze/report/analyze/snapshot scripts, the tests or the dev record;
    those files are only captured by the external candidate hash list. Later doc or
    tooling edits get a separate post-review snapshot and never rewrite these values.
    """
    return {
        "v58_common.py": sha256_file(HERE / "v58_common.py"),
        "v58_env.py": sha256_file(HERE / "v58_env.py"),
        "v58_evaluate.py": sha256_file(HERE / "v58_evaluate.py"),
        "v58_verify.py": sha256_file(HERE / "v58_verify.py"),
        "v50_common.py": sha256_file(V50_DIR / "v50_common.py"),
        "v50_env.py": sha256_file(V50_DIR / "v50_env.py"),
        "fish_env.py": sha256_file(ROOT_DIR / "fish_env.py"),
    }
