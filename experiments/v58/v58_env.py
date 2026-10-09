"""v58 post-reset joint-stress intervention harness (no training).

The scientific object is one nominal world per scenario: reset the unchanged v50
`FishEscapeEnv` (`initial_escape_boost=True, escape_boost_speed=0.8`, full predator
heading/speed bias and pre-roll). We snapshot that post-reset state, then for each
condition restore the snapshot and change ONLY the post-reset velocity vectors, then
recompute observations:

  * `nominal`         : fish factor 1.0, predator rotation 0 deg, predator speed
                        factor 1.0 — the exact identity, reproducing the raw nominal
                        post-reset state and rollout bit-for-bit;
  * `combined_stress` : the scene's frozen joint triple (fish factor in {0,0.5,1},
                        predator heading rotation in {0,90,180,270} deg, predator
                        speed factor in {0.75,1.0,1.25}), applied to the same nominal
                        world.

Only the velocity vectors change: fish positions/velocities are restored then the
fish velocity vector is scaled; predator position is restored then the predator
velocity vector is rotated (exact integer matrix) and scaled (positive scalar). All
other state — alive flags, death timesteps, timestep, the pre-roll trace and the env
RNG stream — is restored bit-identically, so every condition shares one initial
world. Subsequent gravity / bounce / collision physics are unchanged.

Both the rotation and the scale are exact where required: the integer matrices have
entries in {0, +-1} (norm preserved exactly), and factor 1.0 / rotation 0 deg is the
exact identity. The nominal condition therefore reproduces the raw nominal rollout
bit-for-bit. Exercised by the semantic tests, not assumed.
"""

from typing import Dict, Tuple

import numpy as np

try:  # package-style import during tests/eval
    from . import v58_common as common
except ImportError:  # script-style import
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v58_common as common  # type: ignore


def capture_reset_state(env) -> Dict[str, object]:
    """Snapshot the full post-reset world + RNG state so it can be restored."""
    return {
        "fish_positions": env.fish_positions.copy(),
        "fish_velocities": env.fish_velocities.copy(),
        "fish_alive": env.fish_alive.copy(),
        "fish_death_timesteps": env.fish_death_timesteps.copy(),
        "step_one_death_records": [dict(r) for r in (env.step_one_death_records or [])],
        "predator_pos": env.predator_pos.copy(),
        "predator_vel": env.predator_vel.copy(),
        "timestep": int(env.timestep),
        "last_pre_roll_stats": (
            None if env._last_pre_roll_stats is None
            else {k: (list(v) if isinstance(v, list) else v)
                  for k, v in env._last_pre_roll_stats.items()}
        ),
        "np_random_state": env.np_random.bit_generator.state,
    }


def apply_condition(env, snap: Dict[str, object], triple
                    ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Restore the snapshot, apply the frozen joint triple, recompute obs.

    ``triple`` is ``(fish_factor, rotation_deg, speed_factor)``. For the nominal
    condition it is ``(1.0, 0, 1.0)``, the exact identity. Returns ``(obs,
    cond_info)`` where ``cond_info`` records the effective factors applied from the
    state, never assumed.
    """
    fish_factor, rot_deg, speed_factor = triple
    fish_factor = float(fish_factor)
    rot_deg = int(rot_deg)
    speed_factor = float(speed_factor)
    if fish_factor not in common.FISH_FACTORS:
        raise KeyError(f"unknown fish factor: {fish_factor}")
    if rot_deg not in common.ROTATION_DEGREES:
        raise KeyError(f"unknown rotation: {rot_deg}")
    if speed_factor not in common.SPEED_FACTORS:
        raise KeyError(f"unknown speed factor: {speed_factor}")

    nominal_pred_vel = np.asarray(snap["predator_vel"], dtype=np.float32)
    nominal_pred_speed = float(np.linalg.norm(nominal_pred_vel))

    env.fish_positions = snap["fish_positions"].copy()
    # fish velocities: restore then scale exactly, changing ONLY the velocity vector
    env.fish_velocities = (
        np.asarray(snap["fish_velocities"], dtype=np.float32) * np.float32(fish_factor)
    ).astype(np.float32)
    env.fish_alive = snap["fish_alive"].copy()
    env.fish_death_timesteps = snap["fish_death_timesteps"].copy()
    env.step_one_death_records = [dict(r) for r in snap["step_one_death_records"]]
    env.predator_pos = snap["predator_pos"].copy()
    env.predator_vel = common.rotate_predator_velocity(nominal_pred_vel, rot_deg)
    if speed_factor != 1.0:
        env.predator_vel = (
            env.predator_vel * np.float32(speed_factor)
        ).astype(np.float32)
    env.timestep = int(snap["timestep"])
    env._last_pre_roll_stats = snap["last_pre_roll_stats"]
    env.np_random.bit_generator.state = snap["np_random_state"]
    obs = env._get_observations()
    applied_pred_speed = float(np.linalg.norm(env.predator_vel))
    cond_info = {
        "fish_factor": fish_factor,
        "rotation_deg": rot_deg,
        "speed_factor": speed_factor,
        "nominal_predator_speed": nominal_pred_speed,
        "applied_predator_speed": applied_pred_speed,
    }
    return obs, cond_info


def reset_nominal(env, seed: int) -> Tuple[np.ndarray, Dict[str, object]]:
    """Reset into the nominal boosted world and return (obs, snapshot)."""
    obs, _info = env.reset(seed=seed)
    snap = capture_reset_state(env)
    return obs, snap


def reset_condition(env, seed: int, triple):
    """Convenience: reset nominal then apply a joint triple, returning (obs, snap, info)."""
    _obs, snap = reset_nominal(env, seed)
    obs, cond_info = apply_condition(env, snap, triple)
    return obs, snap, cond_info
