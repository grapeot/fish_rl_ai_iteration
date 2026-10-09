"""V53SingleFishEnv: the accepted v50 fixed-identity single-focal wrapper with a
selectable episode-end semantics at `MAX_TIMESTEPS`.

Both modes use the SAME reward (`survival_only`: +0.7 per alive step, one-shot
-50 on the focal death step) and the SAME physics/observation. The only change
is what happens to a focal fish still alive when the base env reaches
`MAX_TIMESTEPS`:

  * ``timeout_bootstrap`` (v50 convention, the reused control): the horizon end is
    returned as ``truncated=True``. SB3 then adds ``gamma * V(terminal_observation)``
    to the last reward, i.e. a continuing / infinite-horizon value target. This is
    the semantics the accepted v50 corrected `survival_only` runs were trained with.
  * ``finite_terminal`` (the treatment): the horizon end is returned as a genuine
    ``terminated=True`` with no bootstrap; the terminal value is 0. The last step
    still earns +0.7 (the fish was alive that step), then the episode ends.

Death anywhere is terminal with -50 in both modes. An explicit ``seed=`` is
honoured; otherwise the worker seed is consumed exactly once (first reset) and
later resets continue the base env RNG, giving fresh episodes across auto-resets
(identical to the audited v50 P0 fix).

The evaluation of both arms does NOT use this wrapper: like v50, evaluation runs
the deterministic multi-fish `FishEscapeEnv` directly, so the contrast is purely
about the training-time end semantics.
"""

from typing import Optional

import gymnasium as gym  # noqa: F401  (kept for API parity with v50_env)

try:  # package-style import during training
    from . import v53_common as common
    from .v50_env import V50SingleFishEnv
except ImportError:  # script-style import during tests/eval
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v53_common as common  # type: ignore
    from v50_env import V50SingleFishEnv  # type: ignore

HOLD_ACTION = 4


class V53SingleFishEnv(V50SingleFishEnv):
    """v50 wrapper + an explicit `termination_mode` for the step-500 horizon."""

    def __init__(
        self,
        include_neighbor_features: bool = False,
        reward_mode: str = common.REWARD_MODE,
        worker_seed: Optional[int] = None,
        termination_mode: str = "finite_terminal",
    ):
        if termination_mode not in common.TERMINATION_MODES:
            raise ValueError(f"unknown termination_mode: {termination_mode}")
        super().__init__(
            include_neighbor_features=include_neighbor_features,
            reward_mode=reward_mode,
            worker_seed=worker_seed,
        )
        self.termination_mode = termination_mode

    def step(self, action: int):
        next_obs, reward, terminated, truncated, info = super().step(action)
        if self.termination_mode == "finite_terminal" and truncated and not terminated:
            # Horizon end for a surviving focal: promote to a real terminal with
            # no bootstrap. Reward is unchanged (+0.7 for the alive final step).
            terminated = True
            truncated = False
            info = dict(info)
            info["agent_truncated"] = False
            info["agent_finite_terminal"] = True
        return next_obs, reward, terminated, truncated, info
