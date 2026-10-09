"""v54 post-reset initial-velocity intervention harness (no training).

The scientific object is a single nominal world per scenario: reset the
unchanged v50 `FishEscapeEnv` with `initial_escape_boost=True,
escape_boost_speed=0.8` (each fish gets a near-radial initial velocity of
magnitude ``FISH_MAX_SPEED*0.8 = 1.6``). We snapshot that post-reset state, then
for each condition scale ONLY the fish velocity vector by
``{1.0, 0.5, 0.0}`` and recompute observations. Positions, alive flags, predator
position/velocity, the pre-roll trace and the env RNG stream are restored
bit-identically, so the three conditions share one initial world.

This is a **post-reset initial-velocity intervention**. It is explicitly NOT
``initial_escape_boost=False``: that would change the initial-draw distribution
and the RNG consumption sequence, whereas here the nominal world is always the
boosted one and only the post-reset velocity vector is rescaled.

Zero-velocity joint consequence (from `fish_env._update_fish`): with
``speed <= 0.01`` the heading falls back to ``[1, 0]``; action 1/2 (turn) sets
``vel = new_dir * speed`` which stays 0 (a no-op), action 3 (decelerate) keeps it
0, action 4 (hold) keeps it 0, and only action 0 (forward) starts moving, always
in +x. The velocity observation channel (``obs[2], obs[3]``) is also zero. These
facts are exercised by the semantic tests, not assumed.
"""

from typing import Dict, Optional

import numpy as np

try:  # package-style import during tests
    from . import v54_common as common
except ImportError:  # script-style import
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v54_common as common  # type: ignore

HOLD_ACTION = 4


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


def apply_condition(env, snap: Dict[str, object], factor: float):
    """Restore the snapshot, scale only the fish velocities, recompute obs."""
    env.fish_positions = snap["fish_positions"].copy()
    env.fish_velocities = (snap["fish_velocities"] * np.float32(factor)).astype(np.float32)
    env.fish_alive = snap["fish_alive"].copy()
    env.fish_death_timesteps = snap["fish_death_timesteps"].copy()
    env.step_one_death_records = [dict(r) for r in snap["step_one_death_records"]]
    env.predator_pos = snap["predator_pos"].copy()
    env.predator_vel = snap["predator_vel"].copy()
    env.timestep = int(snap["timestep"])
    env._last_pre_roll_stats = snap["last_pre_roll_stats"]
    env.np_random.bit_generator.state = snap["np_random_state"]
    obs = env._get_observations()
    return obs, {}


def reset_nominal(env, seed: int):
    """Reset into the nominal boosted world and return (obs, snapshot)."""
    obs, _info = env.reset(seed=seed)
    snap = capture_reset_state(env)
    return obs, snap


def reset_condition(env, seed: int, factor: float):
    """Convenience: reset nominal then apply a condition, returning (obs, snap)."""
    _obs, snap = reset_nominal(env, seed)
    obs, _ = apply_condition(env, snap, factor)
    return obs, snap
