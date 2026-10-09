"""V50SingleFishEnv: fixed-identity single-focal wrapper with selectable reward.

This is the v49 `SingleFishControlEnv` (correct identity: fixed focal id per
episode, action to the focal fish only, focal's own reward, focal-death
termination, horizon truncation with terminal_observation for bootstrap) plus a
`reward_mode` switch and an explicit per-worker env seed.

Reward modes (physics, observation and termination are IDENTICAL in both; only
the scalar handed to PPO changes):
  * ``original``      : the focal fish's own entry of the unchanged
                        FishEscapeEnv reward vector: survival +2, in-vision
                        distance term, boundary penalty, density penalty 0.05,
                        all scaled by REWARD_SCALE=0.1, plus a one-shot -50 on the
                        death step. Byte-for-byte the v49 reward.
  * ``survival_only`` : +0.7 on every step the focal fish is alive, and one-shot
                        -50 on the focal death step. This drops the distance,
                        boundary and density terms. The +0.7 matches the original
                        *out-of-vision* baseline (survival 2 + far-distance 5 =
                        7, times scale 0.1), i.e. the original reward shape with
                        the in-vision/boundary/density shaping removed.

The comparison v50 runs is therefore a whole-reward-bundle contrast, not a
single-term ablation: `survival_only` differs from `original` in every shaping
term at once.

Seed handling: `worker_seed` is an explicit *initial* seed for this worker. It is
consumed on the FIRST `reset()` only (a genuine initialization point that pins
the run's starting episode bank); every subsequent reset passes `seed=None` so
`FishEscapeEnv` continues its own RNG stream and produces a fresh episode. In
Gymnasium, `reset(seed=None)` never re-seeds, it just draws the next state from
the (already seeded) `np_random`. This keeps a run reproducible from its worker
seed while never replaying the same world/focal on every episode.

The wrapper also honours an explicit `seed=` argument at any time (standard Gym
semantics): if `seed` is provided it is forwarded to the env, which re-seeds and
produces that exact episode again.
"""

from typing import Optional

import gymnasium as gym
import numpy as np

try:  # package-style import during training
    from . import v50_common as common
except ImportError:  # script-style import during tests/eval
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v50_common as common  # type: ignore

HOLD_ACTION = 4


class V50SingleFishEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        include_neighbor_features: bool = False,
        reward_mode: str = "original",
        worker_seed: Optional[int] = None,
    ):
        super().__init__()
        if reward_mode not in common.REWARD_MODES:
            raise ValueError(f"unknown reward_mode: {reward_mode}")
        self.reward_mode = reward_mode
        self.worker_seed = worker_seed
        self._worker_seed_consumed = False
        self._worker_seed_used: Optional[int] = None
        self.base_env = common.make_base_env(include_neighbor_features=include_neighbor_features)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.focal_id: int = 0
        self._focal_obs: Optional[np.ndarray] = None
        self._ep_return: float = 0.0

    # -- helpers ---------------------------------------------------------
    def _alive_world(self) -> np.ndarray:
        return np.where(self.base_env.fish_alive)[0]

    @staticmethod
    def _row_of(focal_id: int, alive_world: np.ndarray) -> int:
        pos = int(np.searchsorted(alive_world, focal_id))
        if pos < len(alive_world) and int(alive_world[pos]) == focal_id:
            return pos
        return -1

    def _zeros_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Resolve the seed for this reset:
        #  * an explicit `seed=` argument is honoured (standard Gym semantics,
        #    lets an episode be reproduced exactly);
        #  * otherwise, if this worker still has an unconsumed `worker_seed`,
        #    consume it once to pin the run's starting episode;
        #  * otherwise pass None so the env continues its own RNG stream.
        if seed is not None:
            eff_seed = int(seed)
        elif self.worker_seed is not None and not self._worker_seed_consumed:
            eff_seed = int(self.worker_seed)
            self._worker_seed_consumed = True
            self._worker_seed_used = eff_seed
        else:
            eff_seed = None
        obs, info = self.base_env.reset(seed=eff_seed, options=options)
        alive = self._alive_world()
        if len(alive) == 0:
            self.focal_id = 0
            return self._zeros_obs(), {"is_agent": True, "focal_id": 0, "agent_death": True}
        self.focal_id = int(alive[int(self.base_env.np_random.integers(0, len(alive)))])
        row = self._row_of(self.focal_id, alive)
        self._focal_obs = obs[row].astype(np.float32)
        self._ep_return = 0.0
        out = dict(info)
        out.update({"is_agent": True, "focal_id": self.focal_id, "agent_death": False})
        return self._focal_obs, out

    def step(self, action: int):
        action = int(np.asarray(action).reshape(-1)[0])
        alive_before = self._alive_world()
        row = self._row_of(self.focal_id, alive_before)
        actions = np.full(len(alive_before), HOLD_ACTION, dtype=np.int64)
        if row >= 0:
            actions[row] = action
        obs, rewards, _base_term, _base_trunc, info = self.base_env.step(actions)

        focal_alive = bool(self.base_env.fish_alive[self.focal_id])
        terminated = not focal_alive
        truncated = (not terminated) and (self.base_env.timestep >= self.base_env.MAX_TIMESTEPS)

        if self.reward_mode == "original":
            reward = float(rewards[self.focal_id])
        else:  # survival_only
            reward = common.DEATH_STEP_PENALTY if terminated else common.SURVIVAL_STEP_REWARD

        if terminated:
            next_obs = self._zeros_obs()
        else:
            alive_after = self._alive_world()
            next_obs = obs[self._row_of(self.focal_id, alive_after)].astype(np.float32)

        out = dict(info)
        out.update(
            {
                "is_agent": True,
                "focal_id": self.focal_id,
                "agent_death": terminated,
                "agent_truncated": truncated,
            }
        )
        if terminated or truncated:
            out["episode"] = {
                "r": float(self._ep_return + reward),
                "l": int(self.base_env.timestep),
            }
        else:
            self._ep_return += reward
        return next_obs, reward, terminated, truncated, out

    def close(self):
        self.base_env.close()
