"""v56 post-reset predator-speed-scale intervention harness (no training).

The scientific object is a single nominal world per scenario: reset the unchanged v50
`FishEscapeEnv` (`initial_escape_boost=True, escape_boost_speed=0.8`, full predator
heading/speed bias and pre-roll). We snapshot that post-reset state, then for each
condition restore the snapshot and scale ONLY the predator velocity vector by a
deterministic positive factor of {0.75, 1.0, 1.25}, then recompute observations.

Fish positions/velocities, alive flags, predator position, timestep, the pre-roll
trace and the env RNG stream are restored bit-identically, so the three conditions
share one initial world and differ only in the predator's post-reset velocity
magnitude. Direction is preserved (positive scalar). The subsequent gravity / bounce
/ collision physics are unchanged; the velocity itself will evolve differently because
its initial magnitude differs.

This is a **post-reset predator initial-speed intervention**. It is explicitly NOT:
  * a whole-episode constant predator speed, nor a change of the predator's maximum
    speed; only the single post-reset velocity vector is scaled;
  * a resample of the natural initial-speed distribution (which would change the
    pre-roll draw sequence and RNG consumption);
  * a direction change: the scale factor is a positive scalar, so the heading is
    preserved. It is not a gravity turn; `fish_env._update_predator` keeps gravity
    fixed at `+y`.

Identity guarantee: the factor 1.0 condition is the exact scalar identity, so it
reproduces the raw nominal post-reset state and rollout bit-for-bit. Exercised by the
semantic tests, not assumed.

Zero-norm guard: if the nominal post-reset predator velocity norm is ~0 the scale is
undefined. We do NOT divide by zero: the condition keeps the zero vector and records a
``zero_predator_norm`` flag on the returned condition info.
"""

from typing import Dict, Tuple

import numpy as np

try:  # package-style import during tests
    from . import v56_common as common
except ImportError:  # script-style import
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v56_common as common  # type: ignore

# Below this norm the predator post-reset velocity is treated as effectively zero and
# the scale factor is undefined; we keep zero and flag it instead of dividing by zero.
ZERO_PREDATOR_NORM_EPS = 1e-12


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


def apply_condition(env, snap: Dict[str, object], condition: str
                    ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Restore the snapshot, scale ONLY the predator velocity, recompute obs.

    Returns ``(obs, cond_info)`` where ``cond_info`` carries the effective factor,
    the nominal and applied predator speeds and the zero-norm flag.
    """
    if condition not in common.SPEED_FACTORS:
        raise KeyError(f"unknown speed condition: {condition}")
    factor = float(common.SPEED_FACTORS[condition])
    nominal_vel = np.asarray(snap["predator_vel"], dtype=np.float32)
    nominal_speed = float(np.linalg.norm(nominal_vel))
    zero_norm = bool(nominal_speed <= ZERO_PREDATOR_NORM_EPS)

    env.fish_positions = snap["fish_positions"].copy()
    env.fish_velocities = snap["fish_velocities"].copy()
    env.fish_alive = snap["fish_alive"].copy()
    env.fish_death_timesteps = snap["fish_death_timesteps"].copy()
    env.step_one_death_records = [dict(r) for r in snap["step_one_death_records"]]
    env.predator_pos = snap["predator_pos"].copy()
    if zero_norm:
        # Undefined scale: keep the zero vector, never divide by zero.
        env.predator_vel = nominal_vel.copy()
    else:
        env.predator_vel = common.scale_predator_velocity(nominal_vel, factor)
    env.timestep = int(snap["timestep"])
    env._last_pre_roll_stats = snap["last_pre_roll_stats"]
    env.np_random.bit_generator.state = snap["np_random_state"]
    obs = env._get_observations()
    applied_speed = float(np.linalg.norm(env.predator_vel))
    cond_info = {
        "condition": condition,
        "factor": factor,
        "zero_predator_norm": zero_norm,
        "nominal_predator_speed": nominal_speed,
        "applied_predator_speed": applied_speed,
    }
    return obs, cond_info


def reset_nominal(env, seed: int) -> Tuple[np.ndarray, Dict[str, object]]:
    """Reset into the nominal world and return (obs, snapshot)."""
    obs, _info = env.reset(seed=seed)
    snap = capture_reset_state(env)
    return obs, snap


def reset_condition(env, seed: int, condition: str):
    """Convenience: reset nominal then apply a condition, returning (obs, snap)."""
    _obs, snap = reset_nominal(env, seed)
    obs, cond_info = apply_condition(env, snap, condition)
    return obs, snap, cond_info
