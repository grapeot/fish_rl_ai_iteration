#!/usr/bin/env python3
"""v49 PPO baseline trainer: one shared policy on the fixed-focal wrapper.

Semantics (see single_fish_env.py): each episode tracks one focal fish from
reset to death or horizon; the action applies only to that fish; the reward is
that fish's own reward; GAE/bootstrap never cross fish or death. This fixes the
legacy wrapper's identity defect; it is a physically-decoupled fixed-focal
baseline, not a strict multi-fish equivalence (local observation, reward-context
shift under the density term). Stock SB3 only (SubprocVecEnv + PPO). No custom
policy, no gate/tail machinery.

Checkpoint timing: SB3 saves in the callback's on_rollout_end, before that
iteration's update, so label iN holds N-1 completed updates. Stage metrics are
logged one rollout behind the outcome counters and omit the final update; see
artifacts/metadata_notes.json.

Run independence: with SubprocVecEnv, env seeds are assigned seed+rank, so the
runs share env RNG streams. Use disjoint env seed segments for independent runs.

Faithful cost note: the env's density penalty (coef 0.05, unchanged from v48) is
O(num_fish^2) and dominates runtime. Kept so the reward definition matches v48.

Usage:
  experiments/v48/.venv/bin/python experiments/v49/train.py \
      --seed 4901001 --num-envs 6 --iterations 200 --checkpoint-iters 50,100,150 \
      --out-dir experiments/v49/artifacts/runs/seed4901001
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
from single_fish_env import SingleFishControlEnv  # noqa: E402


def make_env():
    import torch

    torch.set_num_threads(1)
    return SingleFishControlEnv(include_neighbor_features=False)


class _State:
    def __init__(self):
        self.episodes = 0
        self.deaths = 0
        self.truncs = 0


def build_callback(model, log_path: Path, checkpoint_dir: Path, checkpoint_iters):
    """Per-iteration metrics log + staged checkpoints + focal outcome counters.

    `on_rollout_end` fires after a rollout is collected and before that
    iteration's PPO update, so `logger.name_to_value` holds the metrics recorded
    during the previous completed train() step.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    state = _State()

    class _CB(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self._completed = 0
            self._t0 = time.time()
            self.rows = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for i, done in enumerate(dones):
                if not done:
                    continue
                info = infos[i] if i < len(infos) else {}
                state.episodes += 1
                if info.get("agent_death"):
                    state.deaths += 1
                elif info.get("agent_truncated"):
                    state.truncs += 1
            return True

        def on_rollout_end(self) -> None:
            if self._completed >= 1:
                vals = dict(self.logger.name_to_value)
                row = {
                    "iteration": self._completed,
                    "wall_sec": round(time.time() - self._t0, 3),
                    "episodes_so_far": state.episodes,
                    "deaths_so_far": state.deaths,
                    "truncs_so_far": state.truncs,
                }
                row.update({k: float(v) for k, v in vals.items()})
                self.rows.append(row)
                with open(log_path, "w", encoding="utf-8") as f:
                    for r in self.rows:
                        f.write(json.dumps(r) + "\n")
            self._completed += 1
            if self._completed in checkpoint_iters:
                model.save(str(checkpoint_dir / f"model_iter_{self._completed}.zip"))

    return _CB(), state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--net-arch", type=str, default="384,384")
    ap.add_argument("--checkpoint-iters", type=str, default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-subproc", action="store_true",
                    help="use DummyVecEnv (single process) for debugging")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_iters = [int(x) for x in args.checkpoint_iters.split(",") if x.strip()]

    n = 1 if args.no_subproc else args.num_envs
    if n == 1:
        vec_env = DummyVecEnv([make_env])
    else:
        vec_env = SubprocVecEnv([make_env for _ in range(n)])

    layers = [int(x) for x in args.net_arch.split(",") if x.strip()]
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=args.seed,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=layers, vf=list(layers))),
    )

    config = vars(args).copy()
    config["include_neighbor_features"] = False
    config["env_config"] = {
        k: v for k, v in common.env_config(False).items() if k != "predator_heading_bias"
    }
    config["dependency_snapshot"] = common.dependency_snapshot()
    config["total_steps"] = args.iterations * args.n_steps * n
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    cb_factory, state = build_callback(model, out_dir / "train_metrics.jsonl",
                                       checkpoint_dir, set(checkpoint_iters))
    callback = cb_factory

    total_steps = args.iterations * args.n_steps * n
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    wall = time.time() - t0

    model.save(str(checkpoint_dir / "model_final.zip"))
    summary = {
        "seed": args.seed,
        "iterations": args.iterations,
        "num_envs": n,
        "total_steps": total_steps,
        "wall_sec": wall,
        "steps_per_sec": total_steps / max(wall, 1e-9),
        "episodes": state.episodes,
        "deaths": state.deaths,
        "truncs": state.truncs,
        "focal_survival_rate": (state.truncs / state.episodes) if state.episodes else None,
        "checkpoints": sorted(p.name for p in checkpoint_dir.glob("*.zip")),
        "dependency_snapshot": common.dependency_snapshot(),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    vec_env.close()


if __name__ == "__main__":
    main()
