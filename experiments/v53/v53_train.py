#!/usr/bin/env python3
"""v53 PPO trainer: the accepted v50 `survival_only` training recipe with only the
step-500 episode-end semantics switched (see v53_env.py).

Everything the v50 corrected runs froze is held fixed here:
  * the survival_only reward (+0.7 alive / -50 death, one-shot);
  * physics/observation/action (96 fish, 11-dim local obs, other fish HOLD,
    fixed focal, neighbor off);
  * PPO hyperparameters, 200 iterations x 512 steps x 6 envs = 614,400 steps;
  * per-replicate continuous worker bank and model seed (paired with the reused
    v50 control), so both arms of a replicate start from the same initial policy
    and the same starting episodes;
  * post-update stage checkpoints and metrics 1..200.

The single change is `--termination-mode finite_terminal`: a focal fish alive at
step 500 is a genuine terminal (no bootstrap) instead of a TimeLimit truncation.

Usage:
  experiments/v48/.venv/bin/python experiments/v53/v53_train.py \
      --replicate 0 --termination-mode finite_terminal --iterations 200 \
      --num-envs 6 --checkpoint-iters 50,100,150 \
      --out-dir experiments/v53/artifacts/runs/rep0_finite_terminal
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

import v53_common as common  # noqa: E402
from v53_env import V53SingleFishEnv  # noqa: E402


def policy_state_hash(model) -> str:
    """Deterministic content hash of the policy/value tensors (timestamp-free)."""
    import torch

    h = hashlib.sha256()
    state = model.policy.state_dict()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_env_fns(worker_seeds, termination_mode):
    fns = []
    for ws in worker_seeds:
        def _f(ws=ws):
            import torch

            torch.set_num_threads(1)
            return V53SingleFishEnv(
                include_neighbor_features=False,
                reward_mode=common.REWARD_MODE,
                worker_seed=ws,
                termination_mode=termination_mode,
            )

        fns.append(_f)
    return fns


def make_ppo_subclass():
    """PPO subclass that snapshots metrics/checkpoints AFTER each PPO update."""
    from stable_baselines3 import PPO

    class _V53PPO(PPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._v53_updates = []
            self._v53_ckpt_meta = []
            self._v53_t0 = 0.0
            self._v53_rollout = 0
            self._v53_checkpoint_iters = set()
            self._v53_checkpoint_dir = "."

        def train(self) -> None:
            super().train()
            self._v53_rollout += 1
            k = int(self._v53_rollout)
            snap = {
                "update": k,
                "epochs_done": int(self._n_updates),
                "num_timesteps": int(self.num_timesteps),
                "wall_sec": round(time.time() - self._v53_t0, 3),
                "diag": {key: float(val) for key, val in self.logger.name_to_value.items()},
            }
            self._v53_updates.append(snap)
            if k in self._v53_checkpoint_iters:
                ckpt = Path(self._v53_checkpoint_dir) / f"model_updates_{k}.zip"
                self.save(str(ckpt))
                self._v53_ckpt_meta.append(
                    {
                        "updates_completed": k,
                        "collected_steps": int(self.num_timesteps),
                        "path": ckpt.name,
                    }
                )

    return _V53PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--termination-mode", choices=common.TERMINATION_MODES,
                    default="finite_terminal")
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

    env_fns = build_env_fns(worker_seeds, args.termination_mode)
    n = 1 if args.no_subproc else args.num_envs
    vec_env = DummyVecEnv(env_fns[:1]) if n == 1 else SubprocVecEnv(env_fns)

    layers = [int(x) for x in args.net_arch.split(",") if x.strip()]
    V53PPO = make_ppo_subclass()
    model = V53PPO(
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

    # Same explicit seed control as v50: the wrapper owns the non-overlapping
    # worker seed, SB3 seeds nothing.
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
            "reward_mode": common.REWARD_MODE,
            "survival_step_reward": common.SURVIVAL_STEP_REWARD,
            "death_step_penalty": common.DEATH_STEP_PENALTY,
            "include_neighbor_features": False,
            "env_config": common.env_config(False),   # FULL config incl heading bias
            "termination_mode": args.termination_mode,
            "total_steps": args.iterations * args.n_steps * n,
            "dependency_snapshot": common.dependency_snapshot(),
            "initial_policy_hash": initial_policy_hash,
            # run-time training source snapshot; never refreshed afterwards.
            "source_sha256": common.training_source_sha256(),
            "command": sys.argv,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    from stable_baselines3.common.callbacks import BaseCallback

    counts = {"rollout_counts": [], "episodes": 0, "deaths": 0, "truncs": 0,
              "finite_terminals": 0}

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
                elif info.get("agent_finite_terminal"):
                    counts["finite_terminals"] += 1
            return True

        def on_rollout_end(self) -> None:
            counts["rollout_counts"].append(
                {
                    "episodes": counts["episodes"],
                    "deaths": counts["deaths"],
                    "truncs": counts["truncs"],
                    "finite_terminals": counts["finite_terminals"],
                }
            )

    model._v53_updates = []
    model._v53_t0 = time.time()
    model._v53_checkpoint_iters = set(checkpoint_iters)
    model._v53_checkpoint_dir = str(checkpoint_dir)
    model._v53_ckpt_meta = []

    total_steps = args.iterations * args.n_steps * n
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=_CountCB(), progress_bar=False)
    wall = time.time() - t0

    checkpoints_meta = list(model._v53_ckpt_meta)
    final_ckpt = checkpoint_dir / "model_final.zip"
    model.save(str(final_ckpt))
    checkpoints_meta.append(
        {
            "updates_completed": int(model._v53_rollout),
            "collected_steps": int(model.num_timesteps),
            "path": final_ckpt.name,
        }
    )

    rollout_counts = counts["rollout_counts"]
    rows = []
    for i, snap in enumerate(model._v53_updates):
        rc = rollout_counts[i] if i < len(rollout_counts) else (
            rollout_counts[-1] if rollout_counts else
            {"episodes": 0, "deaths": 0, "truncs": 0, "finite_terminals": 0}
        )
        row = {
            "iteration": snap["update"],
            "num_timesteps": snap["num_timesteps"],
            "wall_sec": snap["wall_sec"],
            "episodes": rc["episodes"],
            "deaths": rc["deaths"],
            "truncs": rc["truncs"],
            "finite_terminals": rc["finite_terminals"],
        }
        row.update(snap["diag"])
        rows.append(row)
    (out_dir / "train_metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    ep = counts["episodes"]
    summary = {
        "replicate": args.replicate,
        "arm": args.termination_mode,
        "termination_mode": args.termination_mode,
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
        "finite_terminals": counts["finite_terminals"],
        "focal_survival_rate": ((counts["truncs"] + counts["finite_terminals"]) / ep) if ep else None,
        "updates_completed": int(model._v53_rollout),
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
