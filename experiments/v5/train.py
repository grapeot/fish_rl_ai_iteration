import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

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
    """Wrap FishEscapeEnv to expose one observation per step while sampling all fish."""

    metadata = FishEscapeEnv.metadata

    def __init__(
        self,
        num_fish: int = 1,
        seed: Optional[int] = None,
        sampling_mode: Literal["round_robin", "random"] = "round_robin"
    ):
        super().__init__()
        self.num_fish = num_fish
        self.base_env = FishEscapeEnv(num_fish=num_fish)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self._last_obs: Optional[np.ndarray] = None
        self._survival_sum = 0.0
        self._steps = 0
        self._initial_seed = seed
        self._sampling_mode = sampling_mode
        self._fish_pointer = 0
        self._rng = np.random.default_rng(seed)
        self._last_sampled_idx = -1

    def _select_index(self, alive_count: int) -> int:
        if alive_count <= 0:
            return -1
        if self._sampling_mode == "random":
            return int(self._rng.integers(0, alive_count))
        idx = self._fish_pointer % alive_count
        self._fish_pointer = (idx + 1) % max(alive_count, 1)
        return idx

    def _single_observation(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        idx = self._select_index(len(obs))
        self._last_sampled_idx = idx
        return obs[idx]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        rng_seed = seed if seed is not None else self._initial_seed
        self._rng = np.random.default_rng(rng_seed)
        self._fish_pointer = 0
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._survival_sum = 0.0
        self._steps = 0
        single_obs = self._single_observation(obs)
        info = dict(info)
        info["sampled_fish_index"] = getattr(self, "_last_sampled_idx", -1)
        return single_obs, info

    def step(self, action):
        alive = len(self._last_obs) if self._last_obs is not None else self.num_fish
        actions = [action] * max(alive, 1)
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        self._last_obs = obs

        reward = float(np.mean(rewards)) if len(rewards) > 0 else -100.0
        single_obs = self._single_observation(obs)

        # 额外指标，方便 VecMonitor 读取
        info = dict(info)
        info["sampled_fish_index"] = getattr(self, "_last_sampled_idx", -1)
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


def make_env_fns(num_envs: int, num_fish: int, base_seed: Optional[int], sampling_mode: str):
    def _factory(rank: int):
        def _init():
            seed = None if base_seed is None else base_seed + rank
            env = SingleFishEnv(num_fish=num_fish, seed=seed, sampling_mode=sampling_mode)
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
    plt.title(f"Fish RL dev_v5 - {run_name}")
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
        ax1.set_title(f"dev_v5 progression ({run_name})")
        fig.tight_layout()
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        frame = buffer[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    media_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(media_path, frames, duration=0.4)


def evaluate_single_fish(
    model: PPO,
    num_fish: int,
    episodes: int = 5,
    sampling_mode: str = "round_robin"
):
    results = []
    for ep in range(episodes):
        env = SingleFishEnv(num_fish=num_fish, sampling_mode=sampling_mode)
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


def _predict_actions(model: PPO, obs_batch):
    actions = []
    for obs in obs_batch:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
    return actions


def evaluate_multi_fish(model: PPO, num_fish: int, episodes: int = 3):
    """Use a single-fish policy to control all fish in the base env."""

    results = []
    for ep in range(episodes):
        env = FishEscapeEnv(num_fish=num_fish)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        survival_trace = []
        final_alive = num_fish
        while not done:
            actions = _predict_actions(model, obs) if len(obs) > 0 else []
            obs, rewards, terminated, truncated, info = env.step(actions)
            avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
            total_reward += avg_reward
            steps += 1
            survival = float(info.get("survival_rate", 0.0))
            survival_trace.append(survival)
            done = terminated or truncated
            final_alive = int(info.get("num_alive", 0))
        env.close()
        results.append({
            "episode": int(ep),
            "avg_reward": float(total_reward / max(steps, 1)),
            "total_reward": float(total_reward),
            "steps": int(steps),
            "final_num_alive": final_alive,
            "min_survival_rate": float(min(survival_trace) if survival_trace else 0.0)
        })
    return results


def record_multi_fish_video(
    model: PPO,
    num_fish: int,
    video_path: Path,
    max_steps: int = 500,
    fps: int = 20
):
    """Render a deterministic multi-fish rollout to mp4 for qualitative review."""

    video_path.parent.mkdir(parents=True, exist_ok=True)
    env = FishEscapeEnv(num_fish=num_fish, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for _ in range(max_steps):
        actions = _predict_actions(model, obs) if len(obs) > 0 else []
        obs, _, terminated, truncated, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    env.close()
    if not frames:
        return None

    imageio.mimsave(video_path, frames, fps=fps)
    return video_path


def train(args):
    if args.num_envs < 64:
        raise ValueError("SOP requires num_envs >= 64; please increase --num_envs.")

    run_name = args.run_name or datetime.now().strftime("dev_v5_%Y%m%d_%H%M%S")
    run_checkpoint_dir = CHECKPOINT_DIR / run_name
    run_log_path = LOG_DIR / f"{run_name}.log"
    stats_path = run_checkpoint_dir / "training_stats.pkl"
    tb_run_dir = TB_LOG_DIR / run_name

    env_fns = make_env_fns(args.num_envs, args.num_fish, args.seed, args.obs_sampling)
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(
        vec_env,
        info_keywords=(
            "survival_rate",
            "num_alive",
            "final_num_alive",
            "avg_survival_rate",
            "avg_num_alive",
            "sampled_fish_index"
        )
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

    eval_results_single = evaluate_single_fish(
        model,
        num_fish=args.num_fish,
        episodes=args.eval_episodes,
        sampling_mode=args.obs_sampling
    )
    eval_path_single = run_checkpoint_dir / "eval_single_fish.json"
    with open(eval_path_single, "w", encoding="utf-8") as f:
        json.dump({"results": eval_results_single}, f, indent=2)

    eval_multi = None
    eval_path_multi = None
    if args.eval_multi_fish and args.eval_multi_fish > 1:
        eval_multi = evaluate_multi_fish(
            model,
            num_fish=args.eval_multi_fish,
            episodes=args.eval_multi_episodes
        )
        eval_path_multi = run_checkpoint_dir / "eval_multi_fish.json"
        with open(eval_path_multi, "w", encoding="utf-8") as f:
            json.dump({
                "num_fish": args.eval_multi_fish,
                "results": eval_multi
            }, f, indent=2)

    video_path = None
    if args.video_num_fish and args.video_num_fish > 0:
        video_path = MEDIA_DIR / f"{run_name}_multi_fish_eval.mp4"
        record_multi_fish_video(
            model,
            num_fish=args.video_num_fish,
            video_path=video_path,
            max_steps=args.video_max_steps,
            fps=args.video_fps
        )

    print("Training finished")
    print(f"Run name: {run_name}")
    print(f"Checkpoints: {run_checkpoint_dir}")
    print(f"TensorBoard: {tb_run_dir}")
    print(f"Plot saved to: {plot_path}")
    print(f"GIF saved to: {gif_path}")
    print(f"Single-fish eval: {eval_path_single}")
    if eval_path_multi:
        print(f"Multi-fish eval: {eval_path_multi}")
    if video_path:
        print(f"Multi-fish video: {video_path}")

    return {
        "run_name": run_name,
        "stats": callback.stats,
        "plot_path": plot_path,
        "gif_path": gif_path,
        "checkpoint_dir": run_checkpoint_dir,
        "eval_single": eval_path_single,
        "eval_multi": eval_path_multi,
        "log_path": run_log_path,
        "tb_log_dir": tb_run_dir,
        "video_path": video_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Fish RL dev_v5")
    parser.add_argument("--total_iterations", type=int, default=24, help="迭代次数 (rollout count)")
    parser.add_argument("--num_envs", type=int, default=128, help="并行环境数量 (>=64 per SOP)")
    parser.add_argument("--num_fish", type=int, default=25, help="训练时每个环境中的鱼数量")
    parser.add_argument("--n_steps", type=int, default=512, help="每个环境的 rollout 步数")
    parser.add_argument("--batch_size", type=int, default=1024, help="PPO 批大小")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="保存 checkpoint 的 iteration 间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="训练结束后 deterministic 评估 EP 数")
    parser.add_argument("--eval_multi_fish", type=int, default=25, help="多鱼评估时的鱼数量 (>1 启用)")
    parser.add_argument("--eval_multi_episodes", type=int, default=3, help="多鱼 deterministic 评估 EP 数")
    parser.add_argument(
        "--obs_sampling",
        type=str,
        default="round_robin",
        choices=("round_robin", "random"),
        help="单策略训练时的观测采样方式"
    )
    parser.add_argument("--video_num_fish", type=int, default=25, help="录制 mp4 时的鱼数量 (<=0 跳过)")
    parser.add_argument("--video_max_steps", type=int, default=500, help="视频最多录制的步数")
    parser.add_argument("--video_fps", type=int, default=20, help="输出 mp4 的帧率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--run_name", type=str, default=None, help="自定义 run 名称 (可选)")

    args = parser.parse_args()
    train(args)
