import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

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

MAX_STEPS = 500


class SingleFishEnv(gym.Env):
    """Wrap FishEscapeEnv to expose one observation per step while sampling all fish."""

    metadata = FishEscapeEnv.metadata

    def __init__(
        self,
        num_fish: int = 1,
        seed: Optional[int] = None,
        sampling_mode: Literal["round_robin", "random"] = "round_robin",
        include_neighbor_features: bool = False,
        neighbor_radius: float = 2.0,
        neighbor_average_count: int = 6,
        initial_escape_boost: bool = False,
        escape_boost_speed: float = 0.6,
        escape_jitter_std: float = np.pi / 12,
        divergence_reward_coef: float = 0.0,
        density_penalty_coef: float = 0.0,
        density_target: float = 0.4,
    ):
        super().__init__()
        self.num_fish = num_fish
        self.base_env = FishEscapeEnv(
            num_fish=num_fish,
            include_neighbor_features=include_neighbor_features,
            neighbor_radius=neighbor_radius,
            neighbor_average_count=neighbor_average_count,
            initial_escape_boost=initial_escape_boost,
            escape_boost_speed=escape_boost_speed,
            escape_jitter_std=escape_jitter_std,
            divergence_reward_coef=divergence_reward_coef,
            density_penalty_coef=density_penalty_coef,
            density_target=density_target,
        )
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
        # 多阶段训练时保持日志句柄，最终由 finalize() 关闭
        pass

    def finalize(self):
        self._dump_stats()
        if not self._log_file.closed:
            self._log_file.close()


def make_env_fns(
    num_envs: int,
    num_fish: int,
    base_seed: Optional[int],
    sampling_mode: str,
    include_neighbor_features: bool,
    neighbor_radius: float,
    neighbor_average_count: int,
    initial_escape_boost: bool,
    escape_boost_speed: float,
    escape_jitter_std: float,
    divergence_reward_coef: float,
    density_penalty_coef: float,
    density_target: float,
):
    def _factory(rank: int):
        def _init():
            seed = None if base_seed is None else base_seed + rank
            env = SingleFishEnv(
                num_fish=num_fish,
                seed=seed,
                sampling_mode=sampling_mode,
                include_neighbor_features=include_neighbor_features,
                neighbor_radius=neighbor_radius,
                neighbor_average_count=neighbor_average_count,
                initial_escape_boost=initial_escape_boost,
                escape_boost_speed=escape_boost_speed,
                escape_jitter_std=escape_jitter_std,
                divergence_reward_coef=divergence_reward_coef,
                density_penalty_coef=density_penalty_coef,
                density_target=density_target,
            )
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
    plt.title(f"Fish RL dev_v8 - {run_name}")
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
        ax1.set_title(f"dev_v8 progression ({run_name})")
        fig.tight_layout()
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        frame = buffer[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    media_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(media_path, frames, duration=0.4)


def make_lr_schedule(base_lr: float, mode: str):
    if mode == "constant":
        return base_lr
    if mode == "cosine":
        def _cosine_lr(progress_remaining: float) -> float:
            cosine_component = 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
            return base_lr * cosine_component

        return _cosine_lr

    if mode == "warm_cosine":
        warm_frac = 0.2

        def _warm_cosine(progress_remaining: float) -> float:
            progress_elapsed = 1.0 - progress_remaining
            if progress_elapsed <= warm_frac:
                warm_ratio = progress_elapsed / max(warm_frac, 1e-6)
                # 线性降速：base_lr -> 0.5 * base_lr
                return base_lr * (1.0 - 0.5 * warm_ratio)
            cosine_progress = (progress_elapsed - warm_frac) / max(1 - warm_frac, 1e-6)
            cosine_progress = np.clip(cosine_progress, 0.0, 1.0)
            cosine_component = 0.5 * (1 + np.cos(np.pi * cosine_progress))
            return base_lr * 0.5 * cosine_component

        return _warm_cosine

    raise ValueError(f"Unsupported lr_schedule mode: {mode}")


def parse_curriculum(
    curriculum: str,
    total_iterations: int,
    fallback_num_fish: int,
) -> List[Tuple[int, int]]:
    if not curriculum:
        return [(fallback_num_fish, total_iterations)]

    phases: List[Tuple[int, int]] = []
    for chunk in curriculum.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"curriculum chunk '{chunk}' must be formatted as num_fish:iterations")
        fish_str, iter_str = chunk.split(":", maxsplit=1)
        num_fish = int(fish_str)
        iterations = int(iter_str)
        if num_fish <= 0 or iterations <= 0:
            raise ValueError("num_fish and iterations in curriculum must be positive")
        phases.append((num_fish, iterations))

    total = sum(it for _, it in phases)
    if total != total_iterations:
        raise ValueError(
            f"curriculum iterations sum ({total}) does not match --total_iterations ({total_iterations})"
        )
    return phases


def build_vectorized_env(
    num_envs: int,
    num_fish: int,
    seed: Optional[int],
    sampling_mode: str,
    include_neighbor_features: bool,
    neighbor_radius: float,
    neighbor_average_count: int,
    initial_escape_boost: bool,
    escape_boost_speed: float,
    escape_jitter_std: float,
    divergence_reward_coef: float,
    density_penalty_coef: float,
    density_target: float,
) -> VecMonitor:
    env_fns = make_env_fns(
        num_envs,
        num_fish,
        seed,
        sampling_mode,
        include_neighbor_features,
        neighbor_radius,
        neighbor_average_count,
        initial_escape_boost,
        escape_boost_speed,
        escape_jitter_std,
        divergence_reward_coef,
        density_penalty_coef,
        density_target,
    )
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(
        vec_env,
        info_keywords=(
            "survival_rate",
            "num_alive",
            "final_num_alive",
            "avg_survival_rate",
            "avg_num_alive",
            "sampled_fish_index",
            "first_death_step",
        ),
    )
    return vec_env


def evaluate_single_fish(
    model: PPO,
    num_fish: int,
    episodes: int = 5,
    sampling_mode: str = "round_robin",
    include_neighbor_features: bool = False,
    neighbor_radius: float = 2.0,
    neighbor_average_count: int = 6,
    initial_escape_boost: bool = False,
    escape_boost_speed: float = 0.6,
    escape_jitter_std: float = np.pi / 12,
    divergence_reward_coef: float = 0.0,
    density_penalty_coef: float = 0.0,
    density_target: float = 0.4,
):
    results = []
    for ep in range(episodes):
        env = SingleFishEnv(
            num_fish=num_fish,
            sampling_mode=sampling_mode,
            include_neighbor_features=include_neighbor_features,
            neighbor_radius=neighbor_radius,
            neighbor_average_count=neighbor_average_count,
            initial_escape_boost=initial_escape_boost,
            escape_boost_speed=escape_boost_speed,
            escape_jitter_std=escape_jitter_std,
            divergence_reward_coef=divergence_reward_coef,
            density_penalty_coef=density_penalty_coef,
            density_target=density_target,
        )
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


def evaluate_multi_fish(
    model: PPO,
    num_fish: int,
    episodes: int = 3,
    include_neighbor_features: bool = False,
    neighbor_radius: float = 2.0,
    neighbor_average_count: int = 6,
    initial_escape_boost: bool = False,
    escape_boost_speed: float = 0.6,
    escape_jitter_std: float = np.pi / 12,
    divergence_reward_coef: float = 0.0,
    density_penalty_coef: float = 0.0,
    density_target: float = 0.4,
):
    """Use a single-fish policy to control all fish in the base env."""

    results = []
    for ep in range(episodes):
        env = FishEscapeEnv(
            num_fish=num_fish,
            include_neighbor_features=include_neighbor_features,
            neighbor_radius=neighbor_radius,
            neighbor_average_count=neighbor_average_count,
            initial_escape_boost=initial_escape_boost,
            escape_boost_speed=escape_boost_speed,
            escape_jitter_std=escape_jitter_std,
            divergence_reward_coef=divergence_reward_coef,
            density_penalty_coef=density_penalty_coef,
            density_target=density_target,
        )
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        survival_trace = []
        final_alive = num_fish
        death_timesteps = None
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
            if done:
                death_timesteps = info.get("death_timesteps")
        env.close()
        results.append({
            "episode": int(ep),
            "avg_reward": float(total_reward / max(steps, 1)),
            "total_reward": float(total_reward),
            "steps": int(steps),
            "final_num_alive": final_alive,
            "min_survival_rate": float(min(survival_trace) if survival_trace else 0.0),
            "death_timesteps": death_timesteps,
        })
    return results


def record_multi_fish_video(
    model: PPO,
    num_fish: int,
    video_path: Path,
    max_steps: int = 500,
    fps: int = 20,
    include_neighbor_features: bool = False,
    neighbor_radius: float = 2.0,
    neighbor_average_count: int = 6,
    initial_escape_boost: bool = False,
    escape_boost_speed: float = 0.6,
    escape_jitter_std: float = np.pi / 12,
    divergence_reward_coef: float = 0.0,
    density_penalty_coef: float = 0.0,
    density_target: float = 0.4,
):
    """Render a deterministic multi-fish rollout to mp4 for qualitative review."""

    video_path.parent.mkdir(parents=True, exist_ok=True)
    env = FishEscapeEnv(
        num_fish=num_fish,
        render_mode="rgb_array",
        include_neighbor_features=include_neighbor_features,
        neighbor_radius=neighbor_radius,
        neighbor_average_count=neighbor_average_count,
        initial_escape_boost=initial_escape_boost,
        escape_boost_speed=escape_boost_speed,
        escape_jitter_std=escape_jitter_std,
        divergence_reward_coef=divergence_reward_coef,
        density_penalty_coef=density_penalty_coef,
        density_target=density_target,
    )
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


def collect_death_timesteps(eval_multi_results: Optional[Sequence[Dict]], max_steps: int) -> List[int]:
    if not eval_multi_results:
        return []
    values: List[int] = []
    for episode in eval_multi_results:
        steps = episode.get("death_timesteps")
        if not steps:
            continue
        for step in steps:
            if step is None or step < 0:
                values.append(max_steps)
            else:
                values.append(min(int(step), max_steps))
    return values


def plot_death_histogram(death_timesteps: List[int], run_name: str, plot_path: Path, max_steps: int):
    if not death_timesteps:
        return
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    bins = np.arange(0, max_steps + 25, 25)
    plt.hist(death_timesteps, bins=bins, color="tab:red", alpha=0.85, edgecolor="black")
    plt.xlabel("Death timestep (<= survives)")
    plt.ylabel("Count")
    plt.title(f"Death distribution - {run_name}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def save_death_stats(death_timesteps: List[int], output_path: Path, max_steps: int):
    if not death_timesteps:
        return
    arr = np.array(death_timesteps)
    summary = {
        "total_fish_samples": int(arr.size),
        "survival_fraction": float(np.mean(arr >= max_steps)),
        "mean_death_step": float(np.mean(arr)),
        "median_death_step": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "early_death_fraction_150": float(np.mean(arr < 150)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def train(args):
    if args.num_envs < 64:
        raise ValueError("SOP requires num_envs >= 64; please increase --num_envs.")

    curriculum = parse_curriculum(args.curriculum, args.total_iterations, args.num_fish)

    run_name = args.run_name or datetime.now().strftime("dev_v8_%Y%m%d_%H%M%S")
    run_checkpoint_dir = CHECKPOINT_DIR / run_name
    run_log_path = LOG_DIR / f"{run_name}.log"
    stats_path = run_checkpoint_dir / "training_stats.pkl"
    tb_run_dir = TB_LOG_DIR / run_name

    lr_schedule = make_lr_schedule(args.learning_rate, args.lr_schedule)

    vec_env = build_vectorized_env(
        args.num_envs,
        curriculum[0][0],
        args.seed,
        args.obs_sampling,
        args.include_neighbor_features,
        args.neighbor_radius,
        args.neighbor_average_count,
        args.initial_escape_boost,
        args.escape_boost_speed,
        args.escape_jitter_std,
        args.divergence_reward_coef,
        args.density_penalty_coef,
        args.density_target,
    )

    try:
        policy_layers = [int(size.strip()) for size in args.policy_hidden_sizes.split(",") if size.strip()]
    except ValueError as exc:
        raise ValueError("policy_hidden_sizes must be comma-separated integers") from exc
    if not policy_layers:
        raise ValueError("policy_hidden_sizes must include at least one layer size")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(tb_run_dir),
        policy_kwargs=dict(net_arch=dict(pi=policy_layers, vf=list(policy_layers)))
    )

    callback = IterationStatsCallback(
        stats_path=stats_path,
        log_path=run_log_path,
        checkpoint_dir=run_checkpoint_dir,
        save_every=args.checkpoint_interval
    )

    for idx, (phase_num_fish, phase_iterations) in enumerate(curriculum):
        if idx > 0:
            vec_env.close()
            vec_env = build_vectorized_env(
                args.num_envs,
                phase_num_fish,
                args.seed,
                args.obs_sampling,
                args.include_neighbor_features,
                args.neighbor_radius,
                args.neighbor_average_count,
                args.initial_escape_boost,
                args.escape_boost_speed,
                args.escape_jitter_std,
                args.divergence_reward_coef,
                args.density_penalty_coef,
                args.density_target,
            )
            model.set_env(vec_env)
        phase_timesteps = phase_iterations * args.n_steps * args.num_envs
        print(
            f"[dev_v8] Phase {idx + 1}/{len(curriculum)}: num_fish={phase_num_fish}, iterations={phase_iterations},"
            f" timesteps={phase_timesteps}"
        )
        model.learn(
            total_timesteps=phase_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=(idx == 0)
        )

    final_model_path = run_checkpoint_dir / "model_final"
    model.save(final_model_path)
    vec_env.close()
    callback.finalize()

    plot_path = PLOTS_DIR / f"{run_name}_survival.png"
    gif_path = MEDIA_DIR / f"{run_name}_curve.gif"
    plot_stats(callback.stats, run_name, plot_path)
    stats_gif(callback.stats, run_name, gif_path)

    eval_results_single = evaluate_single_fish(
        model,
        num_fish=curriculum[-1][0],
        episodes=args.eval_episodes,
        sampling_mode=args.obs_sampling,
        include_neighbor_features=args.include_neighbor_features,
        neighbor_radius=args.neighbor_radius,
        neighbor_average_count=args.neighbor_average_count,
        initial_escape_boost=args.initial_escape_boost,
        escape_boost_speed=args.escape_boost_speed,
        escape_jitter_std=args.escape_jitter_std,
        divergence_reward_coef=args.divergence_reward_coef,
        density_penalty_coef=args.density_penalty_coef,
        density_target=args.density_target,
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
            episodes=args.eval_multi_episodes,
            include_neighbor_features=args.include_neighbor_features,
            neighbor_radius=args.neighbor_radius,
            neighbor_average_count=args.neighbor_average_count,
            initial_escape_boost=args.initial_escape_boost,
            escape_boost_speed=args.escape_boost_speed,
            escape_jitter_std=args.escape_jitter_std,
            divergence_reward_coef=args.divergence_reward_coef,
            density_penalty_coef=args.density_penalty_coef,
            density_target=args.density_target,
        )
        eval_path_multi = run_checkpoint_dir / "eval_multi_fish.json"
        with open(eval_path_multi, "w", encoding="utf-8") as f:
            json.dump({
                "num_fish": args.eval_multi_fish,
                "results": eval_multi
            }, f, indent=2)

    death_plot_path = None
    death_stats_path = None
    if eval_multi:
        death_timesteps = collect_death_timesteps(eval_multi, MAX_STEPS)
        death_plot_path = PLOTS_DIR / f"{run_name}_death_histogram.png"
        death_stats_path = run_checkpoint_dir / "death_stats.json"
        plot_death_histogram(death_timesteps, run_name, death_plot_path, MAX_STEPS)
        save_death_stats(death_timesteps, death_stats_path, MAX_STEPS)

    video_path = None
    if args.video_num_fish and args.video_num_fish > 0:
        video_path = MEDIA_DIR / f"{run_name}_multi_fish_eval.mp4"
        record_multi_fish_video(
            model,
            num_fish=args.video_num_fish,
            video_path=video_path,
            max_steps=args.video_max_steps,
            fps=args.video_fps,
            include_neighbor_features=args.include_neighbor_features,
            neighbor_radius=args.neighbor_radius,
            neighbor_average_count=args.neighbor_average_count,
            initial_escape_boost=args.initial_escape_boost,
            escape_boost_speed=args.escape_boost_speed,
            escape_jitter_std=args.escape_jitter_std,
            divergence_reward_coef=args.divergence_reward_coef,
            density_penalty_coef=args.density_penalty_coef,
            density_target=args.density_target,
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
    if death_plot_path:
        print(f"Death histogram: {death_plot_path}")
        print(f"Death stats JSON: {death_stats_path}")
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
        "death_plot": death_plot_path,
        "death_stats": death_stats_path,
        "log_path": run_log_path,
        "tb_log_dir": tb_run_dir,
        "video_path": video_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Fish RL dev_v8")
    parser.add_argument("--total_iterations", type=int, default=24, help="迭代次数 (rollout count)")
    parser.add_argument("--num_envs", type=int, default=128, help="并行环境数量 (>=64 per SOP)")
    parser.add_argument("--num_fish", type=int, default=25, help="默认训练时的鱼数量 (无 curriculum 时生效)")
    parser.add_argument("--curriculum", type=str, default="15:6,20:9,25:9", help="逗号分隔的 num_fish:iterations 阶段定义，留空表示禁用")
    parser.add_argument("--n_steps", type=int, default=512, help="每个环境的 rollout 步数")
    parser.add_argument("--batch_size", type=int, default=1024, help="PPO 批大小")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_schedule", type=str, choices=("constant", "cosine", "warm_cosine"), default="warm_cosine", help="学习率日程")
    parser.add_argument("--policy_hidden_sizes", type=str, default="256,256", help="逗号分隔的 MLP 隐层尺寸，如 '384,384'")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="PPO entropy 系数")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="保存 checkpoint 的 iteration 间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="训练结束后 deterministic 评估 EP 数")
    parser.add_argument("--eval_multi_fish", type=int, default=25, help="多鱼评估时的鱼数量 (>1 启用)")
    parser.add_argument("--eval_multi_episodes", type=int, default=5, help="多鱼 deterministic 评估 EP 数")
    parser.add_argument(
        "--obs_sampling",
        type=str,
        default="round_robin",
        choices=("round_robin", "random"),
        help="单策略训练时的观测采样方式"
    )
    parser.add_argument("--include_neighbor_features", dest="include_neighbor_features", action="store_true", help="启用邻居特征")
    parser.add_argument("--no_neighbor_features", dest="include_neighbor_features", action="store_false", help="禁用邻居特征")
    parser.set_defaults(include_neighbor_features=True)
    parser.add_argument("--neighbor_radius", type=float, default=3.0, help="邻居感知半径 (世界坐标)")
    parser.add_argument("--neighbor_average_count", type=int, default=6, help="统计平均的邻居数量上限")
    parser.add_argument("--initial_escape_boost", dest="initial_escape_boost", action="store_true", help="启用开局逃逸脉冲")
    parser.add_argument("--no_initial_escape_boost", dest="initial_escape_boost", action="store_false", help="禁用开局逃逸脉冲")
    parser.set_defaults(initial_escape_boost=True)
    parser.add_argument("--escape_boost_speed", type=float, default=0.7, help="初始逃逸脉冲系数 (乘以 FISH_MAX_SPEED)")
    parser.add_argument("--escape_jitter_std", type=float, default=0.35, help="初始逃逸方向噪声 (弧度标准差)")
    parser.add_argument("--divergence_reward_coef", type=float, default=0.0, help="邻居散度正向奖励系数")
    parser.add_argument("--density_penalty_coef", type=float, default=0.0, help="邻居密度惩罚系数 (penalize > target)")
    parser.add_argument("--density_target", type=float, default=0.4, help="密度惩罚触发的归一化阈值 (0~1)")
    parser.add_argument("--video_num_fish", type=int, default=25, help="录制 mp4 时的鱼数量 (<=0 跳过)")
    parser.add_argument("--video_max_steps", type=int, default=500, help="视频最多录制的步数")
    parser.add_argument("--video_fps", type=int, default=20, help="输出 mp4 的帧率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--run_name", type=str, default=None, help="自定义 run 名称 (可选)")

    args = parser.parse_args()
    train(args)
