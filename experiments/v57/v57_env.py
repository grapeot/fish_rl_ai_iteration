"""V57SingleFishEnv: the accepted v50 fixed-identity single-focal wrapper plus an
optional per-episode *initial-velocity augmentation* (the only training change in
round 9).

Control behaviour (`augment_velocity=False`) is byte-identical to
`V50SingleFishEnv`/`V53SingleFishEnv` under `timeout_bootstrap`: the step-500
horizon is a TimeLimit truncation (SB3 bootstrap), death anywhere is terminal
-50, alive steps reward +0.7, the focal id is fixed for the episode and every
other fish HOLDs.

Treatment behaviour (`augment_velocity=True`) changes exactly one thing at reset
time:

  1. an independent augmentation RNG (`np.random.Generator`) — seeded once at
     construction from an explicit per-worker seed, never from and never touching
     the base world RNG — draws one factor uniformly from {1.0, 0.5, 0.0};
  2. EVERY fish initial velocity vector is multiplied by that factor;
  3. the observation is recomputed so the focal obs velocity channel reflects the
     augmentation.

Because the augmentation RNG is a separate stream, the base world RNG stream is
untouched (worlds, predator, pre-roll are identical to the un-augmented run for the
same worker seed), and the whole (world, factor) sequence is a deterministic
function of the worker seed and the sequence of reset calls. The draw is
independent per reset, so two consecutive episodes can share the same factor; no
non-repetition is claimed.

Factor 0 recovers the v54 zero-velocity semantics (heading falls back to +x; turn
is a no-op at exactly zero) — the game is not changed, only the initial condition.

Like v53, an explicit `seed=` is honoured for the world; otherwise the wrapper
consumes its `worker_seed` on the FIRST reset only and later resets continue the
base env RNG, giving fresh episodes across auto-resets (the audited v50 P0 fix).
"""

from typing import Optional

import gymnasium as gym  # noqa: F401  (API parity with v50_env)
import numpy as np

try:  # package-style import during training
    from . import v57_common as common
    from .v50_env import V50SingleFishEnv
except ImportError:  # script-style import during tests/eval
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v57_common as common  # type: ignore
    from v50_env import V50SingleFishEnv  # type: ignore

HOLD_ACTION = 4


class V57SingleFishEnv(V50SingleFishEnv):
    """v50 wrapper + optional per-episode train-time velocity augmentation."""

    def __init__(
        self,
        include_neighbor_features: bool = False,
        reward_mode: str = common.REWARD_MODE,
        worker_seed: Optional[int] = None,
        augment_velocity: bool = False,
    ):
        super().__init__(
            include_neighbor_features=include_neighbor_features,
            reward_mode=reward_mode,
            worker_seed=worker_seed,
        )
        self.augment_velocity = bool(augment_velocity)
        self.aug_rng = np.random.default_rng(common.augmentation_seed(worker_seed or 0))
        self._aug_resets = 0
        self._last_aug_factor: Optional[float] = None
        self._aug_factor_counts = {f: 0 for f in common.AUG_FACTORS}

    def _recompute_focal_obs(self) -> np.ndarray:
        obs = self.base_env._get_observations()
        alive = self._alive_world()
        row = self._row_of(self.focal_id, alive)
        if row < 0:
            return self._zeros_obs()
        return obs[row].astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        if self.augment_velocity:
            idx = int(self.aug_rng.integers(0, len(common.AUG_FACTORS)))
            factor = float(common.AUG_FACTORS[idx])
            self.base_env.fish_velocities = (
                self.base_env.fish_velocities * np.float32(factor)
            ).astype(np.float32)
            obs = self._recompute_focal_obs()
            self._aug_resets += 1
            self._last_aug_factor = factor
            self._aug_factor_counts[factor] += 1
            info = dict(info)
            info["aug_factor"] = factor
        return obs, info
