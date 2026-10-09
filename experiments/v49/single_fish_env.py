"""SingleFishControlEnv: a correct single-agent wrapper over FishEscapeEnv.

Why this exists (v49 round 1)
----------------------------
The legacy training wrapper `SingleFishEnv` (v45/v47 train.py) returns a
round-robin *different* fish's observation each step, broadcasts one action to
all fish, and returns the mean reward over all fish. One PPO rollout row is then
a mix of (fish_i obs_t, broadcast action, mean reward, fish_j obs_{t+1}), and
GAE/returns bootstrap across fish identities and across deaths.

`experiments/v48/tests/test_env_decomposability.py` establishes, against the
installed `fish_env.py`, that:
  * the predator trajectory is independent of fish actions and fish state,
  * fish do not collide with each other (death only at fish-predator dist<0.7).

This yields a bounded *physical* decoupling, not a strict equivalence: given the
same initial state and the same focal action sequence, other fish's actions do
not change the focal fish's physical trajectory or death, and with neighbor
features OFF the focal observation on that trajectory is likewise unaffected.
That is enough to explain why the fixed-focal wrapper removes the legacy
identity defect. It does NOT make the problem a strict single-agent MDP: the
11-dim observation hides the predator when it is out of vision and carries no
timestep, and the unchanged density penalty (coef 0.05) still reads the number
of other alive fish, so the training reward context (others HOLD) differs from
the evaluation reward context (others follow the policy). Shared reward
function is not the same as equal reward distributions. See dev_v49.md for the
stated bounds.

Semantics guaranteed here
-------------------------
- The focal fish id is fixed for the whole episode; obs always belongs to it.
- `step(action)` applies `action` to the focal fish only; every other alive fish
  holds (no broadcast).
- reward is the focal fish's own per-fish reward (unchanged env reward function).
- `terminated=True` exactly when the focal fish dies (death penalty counted
  once, value bootstraps to 0). `truncated=True` exactly at the horizon while
  the focal fish is alive. The horizon truncation plus `gamma*V` bootstrap is a
  stated training-target convention, not the only correct termination for the
  native finite 500-step task.
- observation_space / action_space are the 11-dim own-observation and the
  5-action Discrete space, identical to what the v49 evaluator feeds the policy.

Deviations from the v48 evaluation config (documented, not silent):
- neighbor features are OFF in v49 train *and* eval; v48's 18-dim numbers are
  historical reference only.
- the reward keeps the unchanged density penalty (coef 0.05). That term depends
  on other fish (which hold during training), so the focal reward has a
  reward-context dependence on siblings with no proven error bound. Dynamics,
  observation and termination do not depend on them.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

try:  # package-style import during training
    from . import common
except ImportError:  # script-style import during tests/eval
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common  # type: ignore

HOLD_ACTION = 4


class SingleFishControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, include_neighbor_features: bool = False):
        super().__init__()
        self.base_env = common.make_env(include_neighbor_features=include_neighbor_features)
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
        obs, info = self.base_env.reset(seed=seed, options=options)
        alive = self._alive_world()
        if len(alive) == 0:
            self.focal_id = 0
            return self._zeros_obs(), {"is_agent": True, "focal_id": 0, "agent_death": True}
        # Draw the focal id from the episode RNG so that successive unseeded
        # resets (SB3 calls reset() generically after the seed budget is spent)
        # diversify the focal fish and its spawn. Deterministic given the seed.
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
        # apply action to the focal fish only; all other alive fish hold.
        actions = np.full(len(alive_before), HOLD_ACTION, dtype=np.int64)
        if row >= 0:
            actions[row] = action
        obs, rewards, _base_term, _base_trunc, info = self.base_env.step(actions)

        focal_alive = bool(self.base_env.fish_alive[self.focal_id])
        # `rewards` is world-indexed (length NUM_FISH); the focal fish's own value
        # is its own per-fish reward (0 base + one -50 death penalty if it died).
        reward = float(rewards[self.focal_id])

        terminated = not focal_alive
        truncated = (not terminated) and (self.base_env.timestep >= self.base_env.MAX_TIMESTEPS)

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
            # SB3's _update_info_buffer reads info["episode"] to fill
            # ep_rew_mean / ep_len_mean. Report the focal-fish episode stats.
            out["episode"] = {
                "r": float(self._ep_return + reward),
                "l": int(self.base_env.timestep),
            }
        else:
            self._ep_return += reward
        return next_obs, reward, terminated, truncated, out

    def close(self):
        self.base_env.close()
