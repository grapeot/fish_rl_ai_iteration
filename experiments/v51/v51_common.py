"""v51 shared config: frozen v49 model manifest + fresh action-selection banks.

Round 3 of 10. The question is an action-selection ablation on the ACCEPTED v49
frozen policies: does deterministic argmax versus stochastic sampling from the
policy distribution change survival, and if so, how does that difference move
between the mid-training ("selection-stage") and final checkpoints? Result: no
universal stochastic benefit (model-specific sign), but sampling does change
specific models' performance. No training happens in this round.

What is frozen here and what is fresh
-------------------------------------
* Frozen inputs: the six v49 models (three runs x {selection-stage, final}) and
  their exact update counts, plus the complete v49 world config (physics, spawn,
  predator bias, 11-dim neighbour-off observation, 96-fish denominator). The
  model paths and SHA-256 are pinned in `MANIFEST_MODELS` before any evaluation.
  Actual update counts are read from each zip's `data['_n_updates']` and recorded
  in the manifest, not inferred from the v49 stage label (the label lags one
  update; i50/i100/i150/final = 49/99/149/200 updates).
* Fresh: this round uses its own scenario RNG (510102) and its own action-stream
  RNG (510101 for the 2-episode debug only; per-(episode, replicate) streams are
  derived from a per-episode SeedSequence for the real run). Neither the v49
  selection bank (482101) nor the v49 report bank (482102) is touched, and no
  old result file is read as an input.

The stochastic arm draws actions from the policy's own categorical distribution.
That consumes the global torch RNG. To keep the world's `np_random` stream
independent of the action draw, the runner reseeds the WORLD explicitly per
episode (`env.reset(seed=...)`) after a one-time torch seeding; it never ties
the env reset seed to the action RNG. See v51_eval.py for the exact ordering.

Module naming: every v51 module is prefixed `v51_` so that older `common.py` /
`evaluate.py` modules from v48/v49/v50 can never shadow or be shadowed by v51
code inside spawned worker processes.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fish_env import FishEscapeEnv  # noqa: E402

NUM_FISH = 96
MAX_TIMESTEPS = 500

# ---------------------------------------------------------------------------
# Fresh v51 banks, frozen in code before any evaluation is seen.
# ---------------------------------------------------------------------------
SCENARIO_RNG_SEED = 510102          # 40 fresh evaluation scenarios
DEBUG_SCENARIO_RNG_SEED = 510101    # 2-episode debug / timing only
N_SCENARIOS = 40
N_DEBUG_SCENARIOS = 2

# Action-stream RNGs. For the real run these are derived per (episode, replicate)
# from a SeedSequence seeded with ACTION_STREAM_MASTER_SEED; a fixed list of
# replicate indices gives independent streams. DEBUG uses the single literal
# stream seed as the master.
ACTION_STREAM_MASTER_SEED = 510201
DEBUG_ACTION_STREAM_MASTER_SEED = 510101
N_ACTION_STREAMS = 3
REPLICATE_INDEX = (0, 1, 2)         # fixed action-stream replicate ids

# ---------------------------------------------------------------------------
# Frozen v49 model manifest (paths repo-relative to the fish_rl root).
# "stage" is the v49 label; actual PPO updates are read from the zip and must
# match `expected_updates` or the run aborts (fail-closed, not silent).
# ---------------------------------------------------------------------------
MANIFEST_MODELS = {
    "run1_sel": {
        "path": "experiments/v49/artifacts/runs/seed4901001/checkpoints/model_iter_150.zip",
        "run_seed": 4901001, "stage": "i150", "expected_updates": 149,
        "phase": "selection",
    },
    "run2_sel": {
        "path": "experiments/v49/artifacts/runs/seed4901002/checkpoints/model_iter_100.zip",
        "run_seed": 4901002, "stage": "i100", "expected_updates": 99,
        "phase": "selection",
    },
    "run3_sel": {
        "path": "experiments/v49/artifacts/runs/seed4901003/checkpoints/model_iter_50.zip",
        "run_seed": 4901003, "stage": "i50", "expected_updates": 49,
        "phase": "selection",
    },
    "run1_final": {
        "path": "experiments/v49/artifacts/runs/seed4901001/checkpoints/model_final.zip",
        "run_seed": 4901001, "stage": "final", "expected_updates": 200,
        "phase": "final",
    },
    "run2_final": {
        "path": "experiments/v49/artifacts/runs/seed4901002/checkpoints/model_final.zip",
        "run_seed": 4901002, "stage": "final", "expected_updates": 200,
        "phase": "final",
    },
    "run3_final": {
        "path": "experiments/v49/artifacts/runs/seed4901003/checkpoints/model_final.zip",
        "run_seed": 4901003, "stage": "final", "expected_updates": 200,
        "phase": "final",
    },
}
MODEL_NAMES = tuple(MANIFEST_MODELS)
RUNS = ("run1", "run2", "run3")

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
    """Exact v49 common.env_config; physics/reward/observation unchanged."""
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
    return FishEscapeEnv(**env_config(include_neighbor_features=include_neighbor_features))


def episode_seeds(rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n)]


def action_stream_seed(episode_index, replicate_index,
                       master=ACTION_STREAM_MASTER_SEED):
    """Derive one action-stream seed for a (episode, replicate) from a SeedSequence.

    Independent streams are spawned deterministically: the master seed is the
    entropy, and the (episode, replicate) pair is the spawn key. Two episodes or
    two replicates never share a stream.
    """
    ss = np.random.SeedSequence([int(master), int(episode_index), int(replicate_index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def debug_action_stream_seed(replicate_index):
    """The 2-episode debug uses a single fixed action-stream seed per replicate."""
    return int(np.random.SeedSequence([DEBUG_ACTION_STREAM_MASTER_SEED,
                                       int(replicate_index)]).generate_state(1, dtype=np.uint32)[0])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dependency_snapshot():
    import importlib.metadata as md
    import platform
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


def build_manifest(models_root):
    """Freeze the six-model manifest: path, sha256, actual update count.

    Reads each zip's `data['num_timesteps']` and `data['_n_updates']`. Aborts if
    the actual update count disagrees with the pre-registered expectation.
    """
    import zipfile
    models_root = Path(models_root)
    rows = []
    for name in MODEL_NAMES:
        spec = MANIFEST_MODELS[name]
        path = models_root / spec["path"]
        if not path.exists():
            raise SystemExit(f"missing model for {name}: {path}")
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data").decode())
        actual_updates = int(data.get("_n_updates", -1)) // 10  # n_epochs = 10
        num_timesteps = int(data.get("num_timesteps", -1))
        if actual_updates != spec["expected_updates"]:
            raise SystemExit(
                f"{name}: actual updates {actual_updates} != expected "
                f"{spec['expected_updates']}; refusing to run"
            )
        rows.append({
            "name": name,
            "path": spec["path"],
            "run_seed": spec["run_seed"],
            "stage": spec["stage"],
            "phase": spec["phase"],
            "completed_ppo_updates": actual_updates,
            "num_timesteps": num_timesteps,
            "_n_updates": int(data.get("_n_updates", -1)),
            "sha256": sha256_file(path),
        })
    return {
        "schema": "v51_model_manifest_v1",
        "path_base": "fish_rl_repository_root",
        "note": ("Frozen v49 checkpoints. completed_ppo_updates read from each "
                 "zip's data['_n_updates']//10, which equals the actual number of "
                 "completed PPO updates; the v49 stage label lags one update."),
        "models": rows,
    }
