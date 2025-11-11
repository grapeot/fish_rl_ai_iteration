import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fish_env import FishEscapeEnv

EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
LOG_DIR = ARTIFACTS_DIR / "logs"
TB_LOG_DIR = ARTIFACTS_DIR / "tb_logs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MEDIA_DIR = ARTIFACTS_DIR / "media"

for _dir in (CHECKPOINT_DIR, LOG_DIR, TB_LOG_DIR, PLOTS_DIR, MEDIA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class SingleFishEnv(gym.Env):
    """Wrap FishEscapeEnv so each vectorized env controls a single fish."""

    metadata = FishEscapeEnv.metadata

    def __init__(self, num_fish: int = 1, seed: int | None = None):
        super().__init__()
        self.num_fish = num_fish
        self.base_env = FishEscapeEnv(num_fish=num_fish)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self._last_obs = None
        self._survival_sum = 0.0
        self._steps = 0
        self._initial_seed = seed

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is None:
            seed = self._initial_seed
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._survival_sum = 0.0
        self._steps = 0
        if len(obs) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32), info
        return obs[0], info

    def step(self, action):
        if self._last_obs is not None:
            alive = len(self._last_obs)
        else:
            alive = self.num_fish
        actions = [action] * max(alive, 1)
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        self._last_obs = obs

        reward = float(rewards[0]) if len(rewards) > 0 else -100.0
        if len(obs) == 0:
            single_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            single_obs = obs[0]

        # 额外指标，方便 VecMonitor 读取
        info = dict(info)
        current_sr = info.get("survival_rate", 0.0)
        self._survival_sum += current_sr
        self._steps += 1
        avg_survival = self._survival_sum / max(self._steps, 1)
        info["avg_survival_rate"] = avg_survival
        info["avg_num_alive"] = avg_survival * self.num_fish
        info["single_reward"] = reward
        if terminated or truncated:
            info["final_num_alive"] = info.get("num_alive", 0)
        else:
            info["final_num_alive"] = None
        return single_obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


class IterationStatsCallback(BaseCallback):
    """Collect stats per rollout iteration and persist checkpoints/plots."""

    def __init__(self, *, stats_path: Path, log_path: Path, checkpoint_dir: Path, save_every: int = 5):
        super().__init__(verbose=1)
        self.stats_path = stats_path
        self.log_path = log_path
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.iteration = 0
        self.stats = {
            "iterations": [],
            "survival_rate": [],
            "avg_reward": [],
            "episode_length": [],
            "avg_num_alive": [],
            "final_num_alive": []
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8")

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self._log_file.write(line)
        self._log_file.flush()
        print(line, end="")

    def _extract_metric(self, episodes: List[Dict], key: str, default: float = 0.0):
        values = [ep[key] for ep in episodes if key in ep]
        if not values:
            return default
        return float(np.mean(values))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.iteration += 1
        episodes = list(self.model.ep_info_buffer)
        if not episodes:
            return True

        survival_rate = self._extract_metric(episodes, "survival_rate")
        final_num_alive = self._extract_metric(episodes, "final_num_alive")
        avg_num_alive = self._extract_metric(episodes, "avg_num_alive", final_num_alive)
        avg_reward = self._extract_metric(episodes, "r")
        avg_length = self._extract_metric(episodes, "l")

        self.stats["iterations"].append(self.iteration)
        self.stats["survival_rate"].append(survival_rate)
        self.stats["avg_reward"].append(avg_reward)
        self.stats["episode_length"].append(avg_length)
        self.stats["avg_num_alive"].append(avg_num_alive)
        self.stats["final_num_alive"].append(final_num_alive)

        self.logger.record("custom/avg_survival_rate", survival_rate)
        self.logger.record("custom/final_num_alive", final_num_alive)
        self.logger.record("custom/avg_num_alive", avg_num_alive)

        self._log(
            f"iter={self.iteration:03d} sr={survival_rate:6.2%} "
            f"final_alive={final_num_alive:5.2f} reward={avg_reward:7.2f} steps={avg_length:6.1f}"
        )

        if self.iteration % self.save_every == 0:
            self._save_checkpoint(self.iteration)
            self._dump_stats()

        return True

    def _save_checkpoint(self, iteration: int):
        path = self.checkpoint_dir / f"model_iter_{iteration}"
        self.model.save(path)
        self._log(f"checkpoint_saved iteration={iteration}")

    def _dump_stats(self):
        with open(self.stats_path, "wb") as f:
            pickle.dump(self.stats, f)

    def _on_training_end(self) -> None:
        self._dump_stats()
        self._log_file.close()


def make_env_fns(num_envs: int, num_fish: int, base_seed: int | None):
    def _factory(rank: int):
        def _init():
            seed = None if base_seed is None else base_seed + rank
            env = SingleFishEnv(num_fish=num_fish, seed=seed)
            return env

        return _init

    return [_factory(r) for r in range(num_envs)]


def plot_stats(stats: Dict[str, List[float]], run_name: str, plot_path: Path):
    if not stats["iterations"]:
        return
    iterations = stats["iterations"]
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, stats["survival_rate"], label="survival_rate")
    plt.plot(iterations, stats["final_num_alive"], label="final_num_alive")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(f"Fish RL dev_v3 - {run_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def stats_gif(stats: Dict[str, List[float]], run_name: str, media_path: Path):
    if not stats["iterations"]:
        return
    frames = []
    iterations = stats["iterations"]
    survival = stats["survival_rate"]
    final_alive = stats["final_num_alive"]
    for idx in range(1, len(iterations) + 1):
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(iterations[:idx], survival[:idx], color="tab:blue", label="survival")
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Survival Rate")
        ax2 = ax1.twinx()
        ax2.plot(iterations[:idx], final_alive[:idx], color="tab:orange", label="final_alive")
        ax2.set_ylabel("Final Alive")
        ax1.set_title(f"dev_v3 progression ({run_name})")
        fig.tight_layout()
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        frame = buffer[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    media_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(media_path, frames, duration=0.4)


def evaluate_model(model: PPO, num_fish: int, episodes: int = 5):
    results = []
    for ep in range(episodes):
        env = SingleFishEnv(num_fish=num_fish)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        final_alive = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            final_alive = info.get("num_alive", final_alive)
        env.close()
        results.append({
            "episode": int(ep),
            "reward": float(total_reward),
            "steps": int(steps),
            "final_num_alive": int(final_alive)
        })
    return results


def train(args):
    run_name = args.run_name or datetime.now().strftime("dev_v3_%Y%m%d_%H%M%S")
    run_checkpoint_dir = CHECKPOINT_DIR / run_name
    run_log_path = LOG_DIR / f"{run_name}.log"
    stats_path = run_checkpoint_dir / "training_stats.pkl"
    tb_run_dir = TB_LOG_DIR / run_name

    env_fns = make_env_fns(args.num_envs, args.num_fish, args.seed)
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(
        vec_env,
        info_keywords=("survival_rate", "num_alive", "final_num_alive", "avg_survival_rate", "avg_num_alive")
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(tb_run_dir),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )

    callback = IterationStatsCallback(
        stats_path=stats_path,
        log_path=run_log_path,
        checkpoint_dir=run_checkpoint_dir,
        save_every=args.checkpoint_interval
    )

    total_timesteps = args.total_iterations * args.n_steps * args.num_envs
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    final_model_path = run_checkpoint_dir / "model_final"
    model.save(final_model_path)

    vec_env.close()

    plot_path = PLOTS_DIR / f"{run_name}_survival.png"
    gif_path = MEDIA_DIR / f"{run_name}_curve.gif"
    plot_stats(callback.stats, run_name, plot_path)
    stats_gif(callback.stats, run_name, gif_path)

    eval_results = evaluate_model(model, num_fish=args.num_fish, episodes=args.eval_episodes)
    eval_path = run_checkpoint_dir / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"results": eval_results}, f, indent=2)

    print("Training finished")
    print(f"Run name: {run_name}")
    print(f"Checkpoints: {run_checkpoint_dir}")
    print(f"TensorBoard: {tb_run_dir}")
    print(f"Plot saved to: {plot_path}")
    print(f"GIF saved to: {gif_path}")
    print(f"Eval results: {eval_path}")

    return {
        "run_name": run_name,
        "stats": callback.stats,
        "plot_path": plot_path,
        "gif_path": gif_path,
        "checkpoint_dir": run_checkpoint_dir,
        "eval_path": eval_path,
        "log_path": run_log_path,
        "tb_log_dir": tb_run_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Fish RL dev_v3")
    parser.add_argument("--total_iterations", type=int, default=20, help="迭代次数 (rollout count)")
    parser.add_argument("--num_envs", type=int, default=128, help="并行环境数量 (>=64 per SOP)")
    parser.add_argument("--num_fish", type=int, default=1, help="每个环境中的鱼数量")
    parser.add_argument("--n_steps", type=int, default=512, help="每个环境的 rollout 步数")
    parser.add_argument("--batch_size", type=int, default=1024, help="PPO 批大小")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="保存 checkpoint 的 iteration 间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="训练结束后 deterministic 评估 EP 数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--run_name", type=str, default=None, help="自定义 run 名称 (可选)")

    args = parser.parse_args()
    train(args)
