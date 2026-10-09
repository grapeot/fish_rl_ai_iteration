"""v55 post-reset predator-velocity rotation intervention harness (no training).

The scientific object is a single nominal world per scenario: reset the unchanged
v50 `FishEscapeEnv` (`initial_escape_boost=True, escape_boost_speed=0.8`, full
predator heading/speed bias and pre-roll). We snapshot that post-reset state, then
for each condition restore the snapshot and rotate ONLY the predator velocity
vector by a deterministic rotation of 0/90/180/270 degrees using exact integer
matrices, and recompute observations. Fish positions/velocities, alive flags, the
timestep, the pre-roll trace and the env RNG stream are restored bit-identically,
so the four conditions share one initial world and differ only in the predator's
post-reset heading.

This is a **post-reset direction intervention**: it is explicitly NOT a resample
of the natural initial-heading distribution (which would change the pre-roll draw
sequence and RNG consumption) and it does NOT change the predator speed (norm is
preserved exactly). It is also NOT a gravity turn: `fish_env._update_predator`
keeps gravity fixed at `+y` (`predator_vel[1] += g*dt`); only the single post-reset
velocity vector is rotated.

Zero-effect / identity guarantee: the `deg0` rotation is the exact 2x2 identity, so
the `deg0` condition reproduces the raw nominal post-reset state and rollout
bit-for-bit. Exercised by the semantic tests, not assumed.
"""

from typing import Dict

import numpy as np

try:  # package-style import during tests
    from . import v55_common as common
except ImportError:  # script-style import
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v55_common as common  # type: ignore


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


def apply_condition(env, snap: Dict[str, object], condition: str):
    """Restore the snapshot, rotate ONLY the predator velocity, recompute obs."""
    env.fish_positions = snap["fish_positions"].copy()
    env.fish_velocities = snap["fish_velocities"].copy()
    env.fish_alive = snap["fish_alive"].copy()
    env.fish_death_timesteps = snap["fish_death_timesteps"].copy()
    env.step_one_death_records = [dict(r) for r in snap["step_one_death_records"]]
    env.predator_pos = snap["predator_pos"].copy()
    env.predator_vel = common.rotate_predator_velocity(snap["predator_vel"], condition)
    env.timestep = int(snap["timestep"])
    env._last_pre_roll_stats = snap["last_pre_roll_stats"]
    env.np_random.bit_generator.state = snap["np_random_state"]
    obs = env._get_observations()
    return obs, {}


def reset_nominal(env, seed: int):
    """Reset into the nominal world and return (obs, snapshot)."""
    obs, _info = env.reset(seed=seed)
    snap = capture_reset_state(env)
    return obs, snap


def reset_condition(env, seed: int, condition: str):
    """Convenience: reset nominal then apply a condition, returning (obs, snap)."""
    _obs, snap = reset_nominal(env, seed)
    obs, _ = apply_condition(env, snap, condition)
    return obs, snap
