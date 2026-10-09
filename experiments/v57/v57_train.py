#!/usr/bin/env python3
"""v57 PPO trainer: the accepted v50 corrected `survival_only` recipe with a
per-episode train-time initial-velocity augmentation and nothing else changed.

Held fixed (identical to the reused v50 control):
  * survival_only reward (+0.7 alive / -50 death, one-shot);
  * physics/observation/action (96 fish, 11-dim local obs, other fish HOLD, fixed
    focal, neighbor off);
  * the 500-step timeout-bootstrap horizon (the v50/v53 control convention);
  * PPO hyperparameters, 200 iterations x 512 steps x 6 envs = 614,400 steps;
  * per-replicate continuous worker bank and model seed (paired with the reused
    v50 control), so both arms of a replicate start from the same initial policy
    and the same starting episodes;
  * post-update stage checkpoints and metrics 1..200.

The single change is `augment_velocity=True` in the wrapper: each episode reset
draws a factor uniformly from {1.0,0.5,0.0} on an augmentation RNG that is
independent of the base world RNG, and scales all fish initial velocities.

Usage:
  experiments/v48/.venv/bin/python experiments/v57/v57_train.py \
      --replicate 0 --augment-velocity --iterations 200 \
      --num-envs 6 --checkpoint-iters 50,100,150 \
      --out-dir experiments/v57/artifacts/runs/rep0_augmented_velocity
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
for p in (str(ROOT), str(HERE), str(ROOT / "experiments" / "v50")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v57_common as common  # noqa: E402
from v57_env import V57SingleFishEnv  # noqa: E402


def policy_state_hash(model) -> str:
    """Deterministic content hash of the policy/value tensors (timestamp-free)."""
    import torch

    h = hashlib.sha256()
    state = model.policy.state_dict()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_env_fns(worker_seeds, augment_velocity):
    fns = []
    for ws in worker_seeds:
        def _f(ws=ws):
            import torch

            torch.set_num_threads(1)
            return V57SingleFishEnv(
                include_neighbor_features=False,
                reward_mode=common.REWARD_MODE,
                worker_seed=ws,
                augment_velocity=augment_velocity,
            )

        fns.append(_f)
    return fns


def make_ppo_subclass():
    """PPO subclass that snapshots metrics/checkpoints AFTER each PPO update."""
    from stable_baselines3 import PPO

    class _V57PPO(PPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._v57_updates = []
            self._v57_ckpt_meta = []
            self._v57_t0 = 0.0
            self._v57_rollout = 0
            self._v57_checkpoint_iters = set()
            self._v57_checkpoint_dir = "."

        def train(self) -> None:
            super().train()
            self._v57_rollout += 1
            k = int(self._v57_rollout)
            snap = {
                "update": k,
                "epochs_done": int(self._n_updates),
                "num_timesteps": int(self.num_timesteps),
                "wall_sec": round(time.time() - self._v57_t0, 3),
                "diag": {key: float(val) for key, val in self.logger.name_to_value.items()},
            }
            self._v57_updates.append(snap)
            if k in self._v57_checkpoint_iters:
                ckpt = Path(self._v57_checkpoint_dir) / f"model_updates_{k}.zip"
                self.save(str(ckpt))
                self._v57_ckpt_meta.append(
                    {
                        "updates_completed": k,
                        "collected_steps": int(self.num_timesteps),
                        "path": ckpt.name,
                    }
                )

    return _V57PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--augment-velocity", action="store_true",
                    help="enable the per-episode train-time velocity augmentation")
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

    env_fns = build_env_fns(worker_seeds, args.augment_velocity)
    n = 1 if args.no_subproc else args.num_envs
    vec_env = DummyVecEnv(env_fns[:1]) if n == 1 else SubprocVecEnv(env_fns)

    layers = [int(x) for x in args.net_arch.split(",") if x.strip()]
    V57PPO = make_ppo_subclass()
    model = V57PPO(
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

    # Same explicit seed control as v50/v53: the wrapper owns the non-overlapping
    # worker seed, SB3 seeds nothing.
    effective_worker_seeds = list(worker_seeds[:n])
    vec_env._seeds = [None] * n
    initial_ckpt = checkpoint_dir / "model_init.zip"
    model.save(str(initial_ckpt))
    initial_policy_hash = policy_state_hash(model)

    aug = {
        "augment_velocity": args.augment_velocity,
        "aug_factors": list(common.AUG_FACTORS),
        "aug_seed_offset": common.AUG_SEED_OFFSET,
        "augmentation_seed_by_worker": {
            int(ws): common.augmentation_seed(ws) for ws in effective_worker_seeds
        },
    }

    config = vars(args).copy()
    config.update(
        {
            "model_seed": model_seed,
            "env_bank_worker_seeds": effective_worker_seeds,
            "reward_mode": common.REWARD_MODE,
            "survival_step_reward": common.SURVIVAL_STEP_REWARD,
            "death_step_penalty": common.DEATH_STEP_PENALTY,
            "include_neighbor_features": False,
            "env_config": common.env_config(False),   # FULL config incl heading bias
            "termination_mode": "timeout_bootstrap",
            "total_steps": args.iterations * args.n_steps * n,
            "dependency_snapshot": common.dependency_snapshot(),
            "initial_policy_hash": initial_policy_hash,
            "augmentation": aug,
            # run-time training source snapshot; never refreshed afterwards.
            "source_sha256": common.training_source_sha256(),
            "command": sys.argv,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    from stable_baselines3.common.callbacks import BaseCallback

    # SB3's VecEnv auto-reset does not forward the reset info to the callback, so
    # the per-episode augmentation coverage is read from the worker wrappers after
    # training (each wrapper counts its own resets/factors). The callback still
    # counts episodes/deaths/truncs from the step info, aligned to v50.
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

    model._v57_updates = []
    model._v57_t0 = time.time()
    model._v57_checkpoint_iters = set(checkpoint_iters)
    model._v57_checkpoint_dir = str(checkpoint_dir)
    model._v57_ckpt_meta = []

    total_steps = args.iterations * args.n_steps * n
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=_CountCB(), progress_bar=False)
    wall = time.time() - t0

    checkpoints_meta = list(model._v57_ckpt_meta)
    final_ckpt = checkpoint_dir / "model_final.zip"
    model.save(str(final_ckpt))
    checkpoints_meta.append(
        {
            "updates_completed": int(model._v57_rollout),
            "collected_steps": int(model.num_timesteps),
            "path": final_ckpt.name,
        }
    )

    # Read per-worker augmentation coverage from the live wrappers (auto-reset
    # info is not forwarded through the VecEnv to the callback).
    aug_factor_counts = {str(f): 0 for f in common.AUG_FACTORS}
    aug_resets_total = 0
    if args.augment_velocity:
        try:
            per_env = vec_env.get_attr("_aug_factor_counts")
            resets = vec_env.get_attr("_aug_resets")
        except Exception as exc:  # pragma: no cover - defensive
            per_env, resets = [], []
            print(f"[warn] could not read augmentation coverage from workers: {exc}")
        for d in per_env:
            for k, v in d.items():
                aug_factor_counts[str(float(k))] = aug_factor_counts.get(str(float(k)), 0) + int(v)
        aug_resets_total = int(sum(int(x) for x in resets)) if resets else 0

    rollout_counts = counts["rollout_counts"]
    rows = []
    for i, snap in enumerate(model._v57_updates):
        rc = rollout_counts[i] if i < len(rollout_counts) else (
            rollout_counts[-1] if rollout_counts else
            {"episodes": 0, "deaths": 0, "truncs": 0}
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
    (out_dir / "train_metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    ep = counts["episodes"]
    summary = {
        "replicate": args.replicate,
        "arm": "treatment",
        "augment_velocity": args.augment_velocity,
        "aug_factors": list(common.AUG_FACTORS),
        "augment_factor_counts": aug_factor_counts,
        "augment_resets_total": aug_resets_total,
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
        "updates_completed": int(model._v57_rollout),
        "train_metrics_rows": len(rows),
        "checkpoints": checkpoints_meta,
        "initial_policy_hash": initial_policy_hash,
        "runtime_source_sha256": common.training_source_sha256(),
        "dependency_snapshot": common.dependency_snapshot(),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    vec_env.close()


if __name__ == "__main__":
    main()
