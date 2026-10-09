#!/usr/bin/env python3
"""v50 PPO trainer: one shared policy on the fixed-focal wrapper, two reward arms.

Two reward arms share physics/observation/termination/budget/optimizer; only the
scalar reward handed to PPO differs (see v50_env.py):
  * original      : the v49 per-fish reward (survival+distance+boundary+density,
                    scale 0.1, death -50).
  * survival_only : +0.7/alive-step, -50/death-step; no distance/boundary/density.

Infrastructure fixes over v49 (the load-bearing part of this round):
  * per-run, non-overlapping worker env seeds set AFTER PPO construction (SB3's
    `PPO.__init__` otherwise seeds the VecEnv `[seed+rank]`, the v49 overlap bug);
  * model/action randomness seeded per replicate (same seed across the two arms
    of a replicate = matched comparator; independent across replicates);
  * each stage checkpoint is saved AFTER the corresponding PPO update, with a
    sidecar recording exact `updates_completed` and `collected_steps`;
  * train metrics are written for every completed update 1..J, including the
    final one, with the episode/death counters aligned to that same update.

Usage:
  experiments/v48/.venv/bin/python experiments/v50/v50_train.py \
      --replicate 0 --arm original --iterations 200 --num-envs 6 \
      --checkpoint-iters 50,100,150 \
      --out-dir experiments/v50/artifacts/runs/rep0_original
"""

import argparse
import hashlib
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

import v50_common as common  # noqa: E402
from v50_env import V50SingleFishEnv  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_state_hash(model) -> str:
    """Deterministic content hash of the policy/value/optimizer tensors.

    Stable across re-saves (unlike the zip file SHA, which embeds timestamps), so
    it is the portable identity used to prove the two arms start from the same
    weights within a replicate.
    """
    import torch

    h = hashlib.sha256()
    state = model.policy.state_dict()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_env_fns(reward_mode, worker_seeds):
    fns = []
    for idx, ws in enumerate(worker_seeds):
        def _f(idx=idx, ws=ws):
            import torch

            torch.set_num_threads(1)
            return V50SingleFishEnv(
                include_neighbor_features=False,
                reward_mode=reward_mode,
                worker_seed=ws,
            )

        fns.append(_f)
    return fns


def make_ppo_subclass():
    """PPO subclass that snapshots metrics and checkpoints AFTER each PPO update.

    v49 logged on `on_rollout_end`, i.e. the metrics of update k-1 while the
    episode counters already included rollout k, and the final update was never
    logged. Overriding `train()` (called once per update by `learn`) fixes both.
    The checkpoint/snapshot state is kept in plain picklable instance attributes
    so that a model save never has to serialize the SubprocVecEnv.
    """
    from stable_baselines3 import PPO

    class _V50PPO(PPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._v50_updates = []          # per-update (rollout) metric snapshots
            self._v50_ckpt_meta = []
            self._v50_t0 = 0.0
            self._v50_rollout = 0           # completed rollouts / PPO update calls
            self._v50_checkpoint_iters = set()   # rollout indices
            self._v50_checkpoint_dir = "."

        def train(self) -> None:
            super().train()
            self._v50_rollout += 1
            k = int(self._v50_rollout)
            snap = {
                "update": k,
                "epochs_done": int(self._n_updates),
                "num_timesteps": int(self.num_timesteps),
                "wall_sec": round(time.time() - self._v50_t0, 3),
                "diag": {key: float(val) for key, val in self.logger.name_to_value.items()},
            }
            self._v50_updates.append(snap)
            if k in self._v50_checkpoint_iters:
                ckpt = Path(self._v50_checkpoint_dir) / f"model_updates_{k}.zip"
                self.save(str(ckpt))
                self._v50_ckpt_meta.append(
                    {
                        "updates_completed": k,
                        "collected_steps": int(self.num_timesteps),
                        "path": ckpt.name,
                    }
                )

    return _V50PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--arm", choices=common.ARMS, required=True)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=common.NUM_ENVS)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--net-arch", type=str, default="384,384")
    ap.add_argument("--checkpoint-iters", type=str, default="50,100,150")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-subproc", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)

    model_seed = common.model_seed(args.replicate)
    worker_seeds = common.env_bank_worker_seeds(args.replicate, args.num_envs)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_iters = sorted(int(x) for x in args.checkpoint_iters.split(",") if x.strip())

    env_fns = build_env_fns(args.arm, worker_seeds)
    n = 1 if args.no_subproc else args.num_envs
    if n == 1:
        vec_env = DummyVecEnv(env_fns[:1])
    else:
        vec_env = SubprocVecEnv(env_fns)

    layers = [int(x) for x in args.net_arch.split(",") if x.strip()]
    V50PPO = make_ppo_subclass()
    model = V50PPO(
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
        seed=model_seed,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=layers, vf=list(layers))),
    )

    # --- explicit seed control AFTER PPO construction ----------------------
    # SB3 `PPO.__init__` -> set_random_seed(model_seed) -> vec_env.seed(model_seed)
    # sets `_seeds = [model_seed + rank]`; if left alone every worker would be
    # seeded from the model seed (the v49 overlap defect) and, worse for the
    # training distribution, the wrapper would replay one world per worker. We
    # therefore tell the VecEnv to seed nothing (`_seeds = [None]*n`): the wrapper
    # consumes its own non-overlapping `worker_seed` on the FIRST reset only and
    # then continues the env RNG stream for every later episode.
    effective_worker_seeds = list(worker_seeds[:n])
    vec_env._seeds = [None] * n
    initial_ckpt = checkpoint_dir / "model_init.zip"
    model.save(str(initial_ckpt))
    initial_policy_hash = policy_state_hash(model)

    config = vars(args).copy()
    config.update(
        {
            "model_seed": model_seed,
            "env_bank_worker_seeds": effective_worker_seeds,
            "reward_mode": args.arm,
            "survival_step_reward": common.SURVIVAL_STEP_REWARD,
            "death_step_penalty": common.DEATH_STEP_PENALTY,
            "include_neighbor_features": False,
            "env_config": {
                k: v for k, v in common.env_config(False).items() if k != "predator_heading_bias"
            },
            "total_steps": args.iterations * args.n_steps * n,
            "dependency_snapshot": common.dependency_snapshot(),
            "initial_policy_hash": initial_policy_hash,
            "source_sha256": {
                "v50_common.py": sha256_file(HERE / "v50_common.py"),
                "v50_env.py": sha256_file(HERE / "v50_env.py"),
                "v50_train.py": sha256_file(HERE / "v50_train.py"),
                "fish_env.py": sha256_file(ROOT / "fish_env.py"),
            },
            "command": sys.argv,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # --- episode/death counters, aligned to the rollout just collected ----
    from stable_baselines3.common.callbacks import BaseCallback

    counts = {"rollout_counts": [], "episodes": 0, "deaths": 0, "truncs": 0}

    class _CountCB(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for i, done in enumerate(dones):
                if not done:
                    continue
                info = infos[i] if i < len(infos) else {}
                counts["episodes"] += 1
                if info.get("agent_death"):
                    counts["deaths"] += 1
                elif info.get("agent_truncated"):
                    counts["truncs"] += 1
            return True

        def on_rollout_end(self) -> None:
            counts["rollout_counts"].append(
                {
                    "episodes": counts["episodes"],
                    "deaths": counts["deaths"],
                    "truncs": counts["truncs"],
                }
            )

    model._v50_updates = []
    model._v50_t0 = time.time()
    model._v50_checkpoint_iters = set(checkpoint_iters)
    model._v50_checkpoint_dir = str(checkpoint_dir)
    model._v50_ckpt_meta = []

    total_steps = args.iterations * args.n_steps * n
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=_CountCB(), progress_bar=False)
    wall = time.time() - t0

    checkpoints_meta = list(model._v50_ckpt_meta)
    final_ckpt = checkpoint_dir / "model_final.zip"
    model.save(str(final_ckpt))
    checkpoints_meta.append(
        {
            "updates_completed": int(model._v50_rollout),
            "collected_steps": int(model.num_timesteps),
            "path": final_ckpt.name,
        }
    )

    # --- metrics: one row per completed update 1..J ------------------------
    # `diag` keys are SB3 logger names already namespaced `train/...`; keep them
    # verbatim instead of re-prefixing (the v49 double-prefix slip).
    rollout_counts = counts["rollout_counts"]
    rows = []
    for i, snap in enumerate(model._v50_updates):
        rc = rollout_counts[i] if i < len(rollout_counts) else (
            rollout_counts[-1] if rollout_counts else {"episodes": 0, "deaths": 0, "truncs": 0}
        )
        row = {
            "iteration": snap["update"],
            "num_timesteps": snap["num_timesteps"],
            "wall_sec": snap["wall_sec"],
            "episodes": rc["episodes"],
            "deaths": rc["deaths"],
            "truncs": rc["truncs"],
        }
        row.update(snap["diag"])
        rows.append(row)
    metrics_path = out_dir / "train_metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ep = counts["episodes"]
    summary = {
        "replicate": args.replicate,
        "arm": args.arm,
        "model_seed": model_seed,
        "num_envs": n,
        "env_bank_worker_seeds": effective_worker_seeds,
        "iterations": args.iterations,
        "total_steps": total_steps,
        "wall_sec": wall,
        "steps_per_sec": total_steps / max(wall, 1e-9),
        "episodes": ep,
        "deaths": counts["deaths"],
        "truncs": counts["truncs"],
        "focal_survival_rate": (counts["truncs"] / ep) if ep else None,
        "updates_completed": int(model._v50_rollout),
        "train_metrics_rows": len(rows),
        "checkpoints": checkpoints_meta,
        "initial_policy_hash": initial_policy_hash,
        "dependency_snapshot": common.dependency_snapshot(),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    vec_env.close()


if __name__ == "__main__":
    main()
