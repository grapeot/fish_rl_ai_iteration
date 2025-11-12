import argparse
import json
import math
import pickle
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Sequence, Set, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
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


def append_jsonl(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def normalize_stage_type(stage_type: Optional[str]) -> Optional[str]:
    if stage_type is None:
        return None
    value = str(stage_type).strip().lower()
    return value or None


def stage_type_slug(stage_type: Optional[str]) -> Optional[str]:
    key = normalize_stage_type(stage_type)
    if key is None:
        return None
    sanitized = []
    for char in key:
        if char.isalnum() or char in {"_", "-"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized) if sanitized else None


def parse_stage_threshold_overrides(spec: Optional[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    if not spec:
        return overrides
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("penalty_gate_success_early_death_by_stage 需使用 'stage:value' 形式")
        stage_part, value_part = chunk.split(":", maxsplit=1)
        stage_key = normalize_stage_type(stage_part)
        if not stage_key:
            raise ValueError("penalty_gate_success_early_death_by_stage 包含空 stage 名称")
        try:
            overrides[stage_key] = float(value_part)
        except ValueError as exc:
            raise ValueError(f"penalty_gate_success_early_death_by_stage 阈值非法: '{value_part}'") from exc
    return overrides


def parse_stage_int_overrides(spec: Optional[str], field_name: str, *, min_value: int = 1) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    if not spec:
        return overrides
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"{field_name} 需使用 'stage:value' 形式")
        stage_part, value_part = chunk.split(":", maxsplit=1)
        stage_key = normalize_stage_type(stage_part)
        if not stage_key:
            raise ValueError(f"{field_name} 包含空 stage 名称")
        try:
            parsed_value = int(value_part)
        except ValueError as exc:
            raise ValueError(f"{field_name} 阈值非法: '{value_part}'") from exc
        if parsed_value < min_value:
            raise ValueError(f"{field_name} 中 {stage_part} 需 >= {min_value}")
        overrides[stage_key] = parsed_value
    return overrides


def _describe_scalar_series(values: Sequence[float]) -> Optional[Dict[str, float]]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return None
    arr = np.array(data, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_pre_roll_samples(eval_multi_results: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not eval_multi_results:
        return samples
    for episode in eval_multi_results:
        stats = episode.get("pre_roll_stats")
        if not stats:
            continue
        sample = {
            "episode": int(episode.get("episode", 0)),
            "spawn_radius": float(stats.get("spawn_radius", 0.0)),
            "spawn_angle_deg": float(stats.get("spawn_angle_deg", 0.0)),
            "initial_speed": float(stats.get("initial_speed", 0.0)),
            "initial_heading_deg": float(stats.get("initial_heading_deg", 0.0)),
            "final_speed": float(stats.get("final_speed", 0.0)),
            "final_heading_deg": float(stats.get("final_heading_deg", 0.0)),
            "pre_roll_steps": int(stats.get("pre_roll_steps", 0)),
            "angle_jitter_deg": [float(val) for val in stats.get("angle_jitter_deg", [])],
            "speed_scale": [float(val) for val in stats.get("speed_scale", [])],
        }
        bias_angle = stats.get("heading_bias_angle_deg")
        sample["heading_bias_angle_deg"] = None if bias_angle is None else float(bias_angle)
        sample["prewarm_velocity_override"] = bool(stats.get("prewarm_velocity_override", False))
        prewarm_velocity = stats.get("prewarm_velocity")
        if isinstance(prewarm_velocity, (list, tuple)) and len(prewarm_velocity) >= 2:
            sample["prewarm_velocity"] = [float(prewarm_velocity[0]), float(prewarm_velocity[1])]
        samples.append(sample)
    return samples


def summarize_pre_roll_samples(samples: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not samples:
        return None
    spawn_radius = [sample.get("spawn_radius") for sample in samples]
    spawn_angle = [sample.get("spawn_angle_deg") for sample in samples]
    initial_heading = [sample.get("initial_heading_deg") for sample in samples]
    final_heading = [sample.get("final_heading_deg") for sample in samples]
    initial_speed = [sample.get("initial_speed") for sample in samples]
    final_speed = [sample.get("final_speed") for sample in samples]
    angle_jitter = [angle for sample in samples for angle in sample.get("angle_jitter_deg", [])]
    speed_scale = [scale for sample in samples for scale in sample.get("speed_scale", [])]
    heading_bias_angles = [sample.get("heading_bias_angle_deg") for sample in samples if sample.get("heading_bias_angle_deg") is not None]
    prewarm_flags = [bool(sample.get("prewarm_velocity_override")) for sample in samples]
    summary = {
        "sampled_episodes": int(len(samples)),
        "spawn_radius": _describe_scalar_series(spawn_radius),
        "spawn_angle_deg": _describe_scalar_series(spawn_angle),
        "initial_heading_deg": _describe_scalar_series(initial_heading),
        "final_heading_deg": _describe_scalar_series(final_heading),
        "initial_speed": _describe_scalar_series(initial_speed),
        "final_speed": _describe_scalar_series(final_speed),
        "angle_jitter_deg": _describe_scalar_series(angle_jitter),
        "speed_scale": _describe_scalar_series(speed_scale),
        "heading_bias_angle_deg": _describe_scalar_series(heading_bias_angles) if heading_bias_angles else None,
    }
    if prewarm_flags:
        hits = sum(1 for flag in prewarm_flags if flag)
        summary["prewarm_override_count"] = int(hits)
        summary["prewarm_override_ratio"] = float(hits / len(prewarm_flags))
    angle_values = [float(val) for val in spawn_angle if val is not None]
    if angle_values:
        hist_counts, hist_edges = np.histogram(angle_values, bins=12, range=(0.0, 360.0))
        summary["spawn_angle_hist_deg"] = {
            "bins": hist_counts.tolist(),
            "edges": [float(edge) for edge in hist_edges.tolist()],
        }
    speed_values = [float(val) for val in initial_speed if val is not None]
    if speed_values:
        max_speed = max(float(max(speed_values)), 1.0)
        upper = math.ceil(max_speed / 0.2) * 0.2
        hist_counts, hist_edges = np.histogram(speed_values, bins=10, range=(0.0, upper))
        summary["initial_speed_hist"] = {
            "bins": hist_counts.tolist(),
            "edges": [float(edge) for edge in hist_edges.tolist()],
        }
    return summary


def summarize_step_one_clusters(
    records: Optional[Sequence[Dict[str, Any]]],
    angle_bin_deg: float = 30.0,
    speed_bin: float = 0.4,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    buckets: Dict[Tuple[int, int], int] = {}
    for record in records:
        velocity = record.get("predator_velocity") or []
        if len(velocity) < 2:
            continue
        vx = float(velocity[0])
        vy = float(velocity[1])
        speed = math.sqrt(vx * vx + vy * vy)
        angle = math.degrees(math.atan2(vy, vx)) % 360.0
        angle_bin = int(angle // angle_bin_deg)
        speed_bin_idx = int(speed // speed_bin)
        key = (angle_bin, speed_bin_idx)
        buckets[key] = buckets.get(key, 0) + 1
    total = sum(buckets.values())
    if total == 0:
        return None
    clusters = []
    for (angle_bin, speed_bin_idx), count in sorted(buckets.items(), key=lambda item: item[1], reverse=True):
        clusters.append(
            {
                "angle_deg_start": float(angle_bin * angle_bin_deg),
                "angle_deg_end": float((angle_bin + 1) * angle_bin_deg),
                "speed_bin_start": float(speed_bin_idx * speed_bin),
                "speed_bin_end": float((speed_bin_idx + 1) * speed_bin),
                "count": int(count),
                "share": float(count / total),
            }
        )
    return {
        "total": int(total),
        "angle_bin_deg": float(angle_bin_deg),
        "speed_bin": float(speed_bin),
        "clusters": clusters,
    }


def _ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def compute_cluster_angle_speed_share(
    clusters: Sequence[Dict[str, Any]],
    angle_min: float,
    angle_max: float,
    min_speed: float,
) -> float:
    share = 0.0
    for cluster in clusters:
        start = float(cluster.get("angle_deg_start", 0.0))
        end = float(cluster.get("angle_deg_end", 0.0))
        speed_start = float(cluster.get("speed_bin_start", 0.0))
        speed_end = float(cluster.get("speed_bin_end", speed_start))
        if not _ranges_overlap(start, end, angle_min, angle_max):
            continue
        if speed_start >= min_speed or speed_end >= min_speed:
            share += float(cluster.get("share", 0.0))
    return share


class TrainingSignals:
    """Mutable container so callbacks可以读取实时实验信号（如 escape boost gating）。"""

    def __init__(self, escape_boost_speed: float = 0.0):
        self.escape_boost_speed = float(escape_boost_speed)
        self.phase_base_escape_boost_speed = float(escape_boost_speed)
        self.phase_floor_escape_boost_speed = float(escape_boost_speed)


class AdaptiveBoostController:
    """Adjust escape boost speed based on rolling first_death_p10 heuristics."""

    def __init__(
        self,
        *,
        window: int,
        lower_p10: float,
        upper_p10: float,
        low_speed: float,
        high_speed: float,
        min_iteration: int = 1,
        use_median: bool = False,
        stage_speed_floors: Optional[Dict[int, float]] = None,
    ):
        self.window = max(int(window), 1)
        self.lower_p10 = float(lower_p10)
        self.upper_p10 = float(upper_p10)
        self.low_speed = float(low_speed)
        self.high_speed = float(high_speed)
        self.min_iteration = max(int(min_iteration), 1)
        self._history: Deque[float] = deque(maxlen=self.window)
        self._use_median = bool(use_median)
        self._stage_speed_floors = {int(k): float(v) for k, v in (stage_speed_floors or {}).items()}
        self._current_stage = 0

    def update(self, iteration: int, p10_value: Optional[float]) -> Optional[Tuple[float, float]]:
        if p10_value is None or np.isnan(p10_value):
            return None
        self._history.append(float(p10_value))
        if iteration < self.min_iteration or len(self._history) < self.window:
            return None
        if self._use_median:
            window_metric = float(np.median(self._history))
        else:
            window_metric = float(np.mean(self._history))
        target: Optional[float] = None
        if window_metric < self.lower_p10:
            target = self.low_speed
        elif window_metric > self.upper_p10:
            target = self.high_speed
        if target is None:
            return None
        return target, window_metric

    def set_stage(self, stage: Optional[int]) -> None:
        if stage is None:
            return
        self._current_stage = max(int(stage), 0)

    def stage_floor(self) -> Optional[float]:
        value = self._stage_speed_floors.get(self._current_stage)
        if value is None:
            return None
        return float(value)


class TailStageTracker:
    """Track staged tail-prewarm queues so each iteration knows remaining supply."""

    def __init__(self, stages: Sequence[Dict[str, Any]]):
        self._stages: List[Dict[str, Any]] = []
        cumulative = 0
        for idx, stage in enumerate(stages, start=1):
            capacity = int(stage.get("capacity", 0) or 0)
            if capacity <= 0:
                continue
            label = stage.get("label") or f"stage_{idx}"
            stage_type = stage.get("stage_type") or stage.get("type")
            stage_type_str = str(stage_type).strip() if stage_type is not None else None
            start = cumulative
            cumulative += capacity
            self._stages.append(
                {
                    "index": idx,
                    "label": str(label),
                    "stage_type": stage_type_str,
                    "capacity": capacity,
                    "start": start,
                    "end": cumulative,
                }
            )
        self.total_capacity = cumulative
        self._consumed = 0

    def _stage_for_consumed(self, consumed: int) -> Optional[Dict[str, Any]]:
        if not self._stages:
            return None
        stage = self._stages[-1]
        for candidate in self._stages:
            if consumed < candidate["end"]:
                stage = candidate
                break
        return stage

    def update_total_consumed(self, total: int) -> None:
        if self.total_capacity <= 0:
            return
        value = max(int(total), 0)
        self._consumed = min(value, self.total_capacity)

    def status(self) -> Optional[Dict[str, Any]]:
        if self.total_capacity <= 0 or not self._stages:
            return None
        consumed = min(max(int(self._consumed), 0), self.total_capacity)
        stage = self._stage_for_consumed(consumed)
        if stage is None:
            return None
        progress = 1.0
        if stage["capacity"] > 0:
            progress = (consumed - stage["start"]) / float(stage["capacity"])
            progress = max(0.0, min(progress, 1.0))
        remaining = max(self.total_capacity - consumed, 0)
        ratio = None
        if self.total_capacity > 0:
            ratio = remaining / float(self.total_capacity)
        remaining_by_type: Dict[str, int] = {}
        ratio_by_type: Dict[str, float] = {}
        for stage_info in self._stages:
            stage_label = stage_info.get("stage_type") or "unknown"
            stage_consumed = 0
            if consumed >= stage_info["end"]:
                stage_consumed = stage_info["capacity"]
            elif consumed > stage_info["start"]:
                stage_consumed = consumed - stage_info["start"]
            stage_remaining = max(stage_info["capacity"] - stage_consumed, 0)
            remaining_by_type[stage_label] = remaining_by_type.get(stage_label, 0) + stage_remaining
        reinjections_by_type: Dict[str, int] = {}
        stage_entries: Dict[str, int] = {}
        for stage_info in self._stages:
            stage_label = stage_info.get("stage_type") or "unknown"
            if consumed >= stage_info["start"]:
                stage_entries[stage_label] = stage_entries.get(stage_label, 0) + 1
                reinjections_by_type[stage_label] = max(stage_entries[stage_label] - 1, 0)
        if self.total_capacity > 0:
            ratio_by_type = {
                key: value / float(self.total_capacity)
                for key, value in remaining_by_type.items()
            }
        return {
            "index": int(stage["index"]),
            "label": stage["label"],
            "stage_type": stage.get("stage_type"),
            "progress": float(progress),
            "remaining": int(remaining),
            "remaining_ratio": float(ratio) if ratio is not None else None,
            "consumed": int(consumed),
            "total_capacity": int(self.total_capacity),
            "remaining_by_type": remaining_by_type,
            "remaining_ratio_by_type": ratio_by_type or None,
            "reinjections_by_type": reinjections_by_type or None,
            "ne_reinjections": reinjections_by_type.get("ne") if reinjections_by_type else None,
        }


class TailSeedCycler:
    """Persist per-env tail seed offsets across env rebuilds and track consumption."""

    def __init__(self, sequences: Sequence[Sequence[Tuple[float, float]]]):
        payloads: List[List[Tuple[float, float]]] = []
        for seq in sequences or []:
            sanitized: List[Tuple[float, float]] = []
            for item in seq or []:
                if item is None or len(item) < 2:
                    continue
                try:
                    vx = float(item[0])
                    vy = float(item[1])
                except (TypeError, ValueError):
                    continue
                sanitized.append((vx, vy))
            payloads.append(sanitized)
        self._sequences = payloads
        self._offsets: List[int] = [0 for _ in self._sequences]
        self._consumed_total = 0

    def has_data(self) -> bool:
        return any(self._sequences)

    def payloads(self) -> List[Optional[Dict[str, Any]]]:
        """Return per-env payloads with current offsets for env construction."""
        results: List[Optional[Dict[str, Any]]] = []
        for offset, seq in zip(self._offsets, self._sequences):
            if not seq:
                results.append(None)
                continue
            pointer = offset % len(seq) if seq else 0
            results.append(
                {
                    "velocities": list(seq),
                    "offset": pointer,
                }
            )
        return results

    def record_overrides(self, overrides_per_env: Sequence[int]) -> int:
        """Advance offsets using per-env override counts from the latest rollout."""
        total_added = 0
        for idx, amount in enumerate(overrides_per_env or []):
            if idx >= len(self._offsets):
                break
            hits = max(int(amount or 0), 0)
            if hits == 0:
                continue
            seq = self._sequences[idx] if idx < len(self._sequences) else []
            if seq:
                seq_len = len(seq)
                self._offsets[idx] = (self._offsets[idx] + hits) % seq_len
            total_added += hits
        self._consumed_total += total_added
        return total_added

    @property
    def total_consumed(self) -> int:
        return int(self._consumed_total)


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
        predator_spawn_jitter_radius: float = 0.0,
        predator_pre_roll_steps: int = 0,
        predator_pre_roll_angle_jitter: float = 0.0,
        predator_pre_roll_speed_jitter: float = 0.0,
        predator_heading_bias: Optional[Sequence[Dict[str, float]]] = None,
        predator_pre_roll_speed_bias: Optional[Sequence[Dict[str, float]]] = None,
        prewarm_predator_velocities: Optional[Sequence[Sequence[float]]] = None,
        tail_force_reset_steps: Optional[int] = None,
        tail_seed_offset: int = 0,
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
            predator_spawn_jitter_radius=predator_spawn_jitter_radius,
            predator_pre_roll_steps=predator_pre_roll_steps,
            predator_pre_roll_angle_jitter=predator_pre_roll_angle_jitter,
            predator_pre_roll_speed_jitter=predator_pre_roll_speed_jitter,
            predator_heading_bias=predator_heading_bias,
            predator_pre_roll_speed_bias=predator_pre_roll_speed_bias,
            prewarm_predator_velocity_queue=prewarm_predator_velocities,
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
        self._prewarm_episode_total = 0
        self._prewarm_override_total = 0
        self._prewarm_episode_since = 0
        self._prewarm_override_since = 0
        self.tail_force_reset_steps = max(int(tail_force_reset_steps or 0), 0)
        self._forced_tail_reset_since = 0
        self._forced_tail_reset_total = 0
        self._tail_seed_library: List[np.ndarray] = []
        if prewarm_predator_velocities:
            for velocity in prewarm_predator_velocities:
                if velocity is None or len(velocity) < 2:
                    continue
                try:
                    vx = float(velocity[0])
                    vy = float(velocity[1])
                except (TypeError, ValueError):
                    continue
                self._tail_seed_library.append(np.array([vx, vy], dtype=np.float32))
        self._tail_seed_pointer = max(int(tail_seed_offset or 0), 0)
        if self._tail_seed_library:
            self._tail_seed_pointer %= len(self._tail_seed_library)
        else:
            self._tail_seed_pointer = 0
        self._pending_tail_injection = False
        self._tail_injection_since = 0
        self._tail_injection_total = 0

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
        if self._tail_seed_library:
            self._prepare_tail_seed()
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._survival_sum = 0.0
        self._steps = 0
        single_obs = self._single_observation(obs)
        info = dict(info)
        info["sampled_fish_index"] = getattr(self, "_last_sampled_idx", -1)
        self._record_prewarm_usage(info.get("pre_roll_stats"))
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
        forced_reset = False
        if not (terminated or truncated) and self.tail_force_reset_steps > 0 and self._steps >= self.tail_force_reset_steps:
            truncated = True
            forced_reset = True
            info["forced_tail_reset"] = True
            if "death_timesteps" not in info:
                info["death_timesteps"] = []
            if "step_one_deaths" not in info:
                info["step_one_deaths"] = []
        if forced_reset:
            self._forced_tail_reset_since += 1
            self._forced_tail_reset_total += 1
            self._pending_tail_injection = True
        if terminated or truncated:
            info["final_num_alive"] = info.get("num_alive", 0)
        else:
            info["final_num_alive"] = None
        return single_obs, reward, terminated, truncated, info

    def _record_prewarm_usage(self, pre_roll_stats: Optional[Dict[str, Any]]):
        self._prewarm_episode_total += 1
        self._prewarm_episode_since += 1
        flag = False
        if pre_roll_stats is not None:
            flag = bool(pre_roll_stats.get("prewarm_velocity_override", False))
        if flag:
            self._prewarm_override_total += 1
            self._prewarm_override_since += 1

    def consume_prewarm_stats(self) -> Dict[str, int]:
        payload = {
            "episodes": int(self._prewarm_episode_since),
            "overrides": int(self._prewarm_override_since),
            "total_episodes": int(self._prewarm_episode_total),
            "total_overrides": int(self._prewarm_override_total),
            "forced_resets": int(self._forced_tail_reset_since),
            "forced_resets_total": int(self._forced_tail_reset_total),
            "injections": int(self._tail_injection_since),
            "injections_total": int(self._tail_injection_total),
        }
        self._prewarm_episode_since = 0
        self._prewarm_override_since = 0
        self._forced_tail_reset_since = 0
        self._tail_injection_since = 0
        return payload

    def _inject_next_tail_seed(self, force: bool = False) -> bool:
        if not self._tail_seed_library:
            return False
        enqueue_fn = getattr(self.base_env, "enqueue_prewarm_velocity", None)
        queue_len_fn = getattr(self.base_env, "prewarm_queue_length", None)
        if enqueue_fn is None:
            return False
        current_len = 0
        if callable(queue_len_fn):
            try:
                current_len = int(queue_len_fn())
            except Exception:
                current_len = 0
        if not force and current_len > 0:
            return False
        index = self._tail_seed_pointer % len(self._tail_seed_library)
        vector = self._tail_seed_library[index]
        self._tail_seed_pointer += 1
        enqueue_fn(np.array(vector, dtype=np.float32), front=True)
        self._tail_injection_since += 1
        self._tail_injection_total += 1
        return True

    def _prepare_tail_seed(self) -> None:
        if not self._tail_seed_library:
            self._pending_tail_injection = False
            return
        self._inject_next_tail_seed(force=self._pending_tail_injection)
        self._pending_tail_injection = False

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()

    def set_density_penalty(self, coef: float, target: Optional[float] = None):
        """Expose density penalty control so VecEnv can adjust during training."""
        self.base_env.set_density_penalty(coef, target)

    def set_escape_boost_speed(self, speed: float):
        """Propagate escape boost speed adjustments to the base env."""
        self.base_env.set_escape_boost_speed(speed)


class DensityPenaltyScheduler:
    """Linearly interpolate density penalty across curriculum phases."""

    def __init__(
        self,
        base_value: float,
        phase_targets: Sequence[float],
        phase_iterations: Sequence[int],
        ramp_phases: Optional[Set[int]] = None,
        phase_plateaus: Optional[Sequence[int]] = None,
        lock_value: Optional[float] = None,
        lock_until_iteration: Optional[int] = None,
    ):
        if len(phase_targets) != len(phase_iterations):
            raise ValueError("density_penalty_phase_targets must match curriculum length")
        self.base_value = float(base_value)
        self.phase_targets = [float(v) for v in phase_targets]
        self.phase_iterations = list(phase_iterations)
        self.ramp_phases = set(ramp_phases or set())
        if phase_plateaus is not None and len(phase_plateaus) != len(phase_targets):
            raise ValueError("density_penalty_phase_plateaus must match curriculum length")
        self.phase_plateaus = [max(int(v), 0) for v in phase_plateaus] if phase_plateaus else [0] * len(self.phase_targets)
        self.phase_ranges: List[Tuple[int, int]] = []
        start = 1
        for count in self.phase_iterations:
            end = start + max(count, 1) - 1
            self.phase_ranges.append((start, end))
            start = end + 1
        self._initial_value = self._compute_initial_value()
        self._last_value: Optional[float] = self._initial_value
        self._lock_value = float(lock_value) if lock_value is not None else None
        if lock_until_iteration is not None and lock_until_iteration > 0:
            self._lock_until_iteration = int(lock_until_iteration)
        else:
            self._lock_until_iteration = None
        self._max_active_phase = len(self.phase_targets) - 1 if self.phase_targets else 0

    @property
    def current_value(self) -> Optional[float]:
        return self._last_value

    @property
    def final_value(self) -> float:
        return self.phase_targets[-1] if self.phase_targets else self.base_value

    def initial_value(self) -> float:
        return self._initial_value

    def _compute_initial_value(self) -> float:
        if not self.phase_ranges or not self.phase_targets:
            return self.base_value
        return self.base_value if 1 in self.ramp_phases else self.phase_targets[0]

    def _phase_info(self, iteration: int) -> Tuple[int, int, int]:
        for idx, (start, end) in enumerate(self.phase_ranges):
            if start <= iteration <= end:
                return idx, start, end
        # Fallback: after final iteration, clamp to last phase
        last_idx = len(self.phase_ranges) - 1
        return last_idx, *self.phase_ranges[last_idx]

    def value_for_iteration(self, iteration: int) -> Optional[float]:
        phase_idx_actual, _, _ = self._phase_info(iteration)
        phase_idx = min(max(phase_idx_actual, 0), self._max_active_phase)
        phase_start, phase_end = self.phase_ranges[phase_idx]
        clamped_iteration = int(np.clip(iteration, phase_start, phase_end))
        target_value = self.phase_targets[phase_idx]
        start_value = self.base_value if phase_idx == 0 else self.phase_targets[phase_idx - 1]
        if (phase_idx + 1) in self.ramp_phases and phase_end > phase_start:
            phase_length = max(phase_end - phase_start + 1, 1)
            plateau_iters = min(self.phase_plateaus[phase_idx], phase_length)
            plateau_end = phase_start + plateau_iters - 1
            ramp_start = plateau_end + 1
            if plateau_iters >= phase_length or iteration <= plateau_end:
                value = start_value
            else:
                span = phase_end - ramp_start
                if span <= 0:
                    value = target_value
                else:
                    progress = (clamped_iteration - ramp_start) / span
                    progress = float(np.clip(progress, 0.0, 1.0))
                    value = start_value + (target_value - start_value) * progress
        else:
            value = target_value

        if self._lock_value is not None and self._lock_until_iteration is not None:
            if iteration <= self._lock_until_iteration:
                value = min(value, self._lock_value)

        if self._last_value is None or abs(value - self._last_value) > 1e-6:
            self._last_value = value
            return value
        return None

    def set_max_active_phase(self, phase_idx: int) -> None:
        if not self.phase_targets:
            return
        max_idx = len(self.phase_targets) - 1
        new_idx = int(np.clip(phase_idx, 0, max_idx))
        if new_idx == self._max_active_phase:
            return
        self._max_active_phase = new_idx
        self._last_value = None

    @property
    def max_active_phase(self) -> int:
        return self._max_active_phase


class PenaltyStageGate:
    """Gate high-penalty phases until stability metrics stay healthy."""

    def __init__(
        self,
        *,
        phase_allowance: Sequence[int],
        lock_until: int,
        required_successes: int,
        freeze_iterations: int,
        success_step_one: int,
        success_step_one_ratio: Optional[float],
        success_early_death: float,
        success_early_death_window: int,
        success_early_death_by_stage: Optional[Dict[str, float]] = None,
        success_early_death_window_by_stage: Optional[Dict[str, int]] = None,
        success_first_death_p10: Optional[float],
        success_first_death_p10_by_stage: Optional[Dict[str, float]] = None,
        success_freeze_iterations: int,
        failure_p10: float,
        failure_min_final: float,
        success_p10_median_window: int = 1,
        success_p10_median_window_by_stage: Optional[Dict[str, int]] = None,
        failure_tolerance: int = 1,
        failure_tolerance_by_stage: Optional[Dict[str, int]] = None,
    ):
        if not phase_allowance:
            raise ValueError("phase_allowance must contain at least one phase index")
        self.phase_allowance = [max(int(idx), 0) for idx in sorted(set(phase_allowance))]
        self.lock_until = max(int(lock_until), 0)
        self.required_successes = max(int(required_successes), 1)
        self.freeze_iterations = max(int(freeze_iterations), 0)
        self.success_step_one = max(int(success_step_one), 0)
        self.success_step_one_ratio = (
            float(success_step_one_ratio) if success_step_one_ratio is not None and success_step_one_ratio > 0 else None
        )
        self.success_early_death = float(success_early_death)
        self.success_early_death_window = max(int(success_early_death_window), 1)
        self.success_early_death_by_stage = {
            key: float(value)
            for key, value in (success_early_death_by_stage or {}).items()
            if normalize_stage_type(key)
        }
        self.success_early_death_window_by_stage = {
            key: max(int(value), 1)
            for key, value in (success_early_death_window_by_stage or {}).items()
            if normalize_stage_type(key)
        }
        self.success_first_death_p10 = float(success_first_death_p10) if success_first_death_p10 is not None else None
        self.success_first_death_p10_by_stage = {
            key: float(value)
            for key, value in (success_first_death_p10_by_stage or {}).items()
            if normalize_stage_type(key)
        }
        self.success_freeze_iterations = max(int(success_freeze_iterations), 0)
        self.failure_p10 = float(failure_p10)
        self.failure_min_final = float(failure_min_final)
        self.current_stage = 0
        self.success_streak = 0
        self.freeze_until_iteration = 0
        self.last_success_metrics: Optional[Dict[str, Any]] = None
        self.success_p10_median_window = max(int(success_p10_median_window), 1)
        self.success_p10_median_window_by_stage = {
            key: max(int(value), 1)
            for key, value in (success_p10_median_window_by_stage or {}).items()
            if normalize_stage_type(key)
        }
        self._p10_history: Deque[float] = deque(maxlen=max(self.success_p10_median_window, 4))
        self._p10_history_by_stage: Dict[str, Deque[float]] = {}
        self._early_death_history: Deque[float] = deque(maxlen=max(self.success_early_death_window, 4))
        self._early_death_history_by_stage: Dict[str, Deque[float]] = {}
        self.failure_tolerance = max(int(failure_tolerance), 1)
        self.failure_tolerance_by_stage = {
            key: max(int(value), 1)
            for key, value in (failure_tolerance_by_stage or {}).items()
            if normalize_stage_type(key)
        }
        self.failure_streak = 0

    def _stage_threshold(self, stage_key: Optional[str]) -> float:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if normalized and normalized in self.success_early_death_by_stage:
            return self.success_early_death_by_stage[normalized]
        return self.success_early_death

    def _stage_window_size(self, stage_key: Optional[str]) -> int:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if normalized and normalized in self.success_early_death_window_by_stage:
            return self.success_early_death_window_by_stage[normalized]
        return self.success_early_death_window

    def _stage_failure_tolerance(self, stage_key: Optional[str]) -> int:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if normalized and normalized in self.failure_tolerance_by_stage:
            return self.failure_tolerance_by_stage[normalized]
        return self.failure_tolerance

    def _stage_history(self, stage_key: Optional[str]) -> Optional[Deque[float]]:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if not normalized:
            return None
        history = self._early_death_history_by_stage.get(normalized)
        if history is None:
            window_size = max(self._stage_window_size(stage_key), 1)
            history = deque(maxlen=max(window_size, 4))
            self._early_death_history_by_stage[normalized] = history
        return history

    def _stage_success_p10_threshold(self, stage_key: Optional[str]) -> Optional[float]:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if normalized and normalized in self.success_first_death_p10_by_stage:
            return self.success_first_death_p10_by_stage[normalized]
        return self.success_first_death_p10

    def _stage_p10_window_size(self, stage_key: Optional[str]) -> int:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if normalized and normalized in self.success_p10_median_window_by_stage:
            return self.success_p10_median_window_by_stage[normalized]
        return self.success_p10_median_window

    def _stage_p10_history(self, stage_key: Optional[str]) -> Optional[Deque[float]]:
        normalized = normalize_stage_type(stage_key) if stage_key is not None else None
        if not normalized:
            return None
        history = self._p10_history_by_stage.get(normalized)
        if history is None:
            window_size = max(self._stage_p10_window_size(stage_key), 1)
            history = deque(maxlen=max(window_size, 4))
            self._p10_history_by_stage[normalized] = history
        return history

    def current_phase_limit(self) -> int:
        return self.phase_allowance[self.current_stage]

    def max_stage(self) -> int:
        return len(self.phase_allowance) - 1

    def handle_multi_eval(self, iteration: int, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        summary = entry.get("summary") or {}
        death_stats = entry.get("death_stats") or {}
        if not summary:
            return None
        step_one = entry.get("step_one_death_count")
        step_one_ratio = entry.get("step_one_death_ratio")
        earlydeath = death_stats.get("early_death_fraction_100")
        first_death_p10 = death_stats.get("p10")
        min_final = summary.get("min_final_survival_rate")
        stage_type_value = entry.get("tail_stage_type") or entry.get("stage_type")
        stage_threshold = self._stage_threshold(stage_type_value)
        stage_key = normalize_stage_type(stage_type_value)
        stage_override_active = bool(stage_key and stage_key in self.success_early_death_by_stage)
        if step_one_ratio is None and step_one is not None:
            episodes = int(summary.get("episodes", 0) or 0)
            num_fish = int(entry.get("multi_eval_num_fish") or entry.get("num_fish") or 0)
            denom = episodes * num_fish
            if denom > 0:
                step_one_ratio = float(step_one) / float(denom)

        failure = False
        if first_death_p10 is not None and first_death_p10 < self.failure_p10:
            failure = True
        if min_final is not None and min_final < self.failure_min_final:
            failure = True

        median_window_value: Optional[float] = None
        stage_p10_median_value: Optional[float] = None
        stage_p10_history = self._stage_p10_history(stage_type_value)
        stage_p10_window = self._stage_p10_window_size(stage_type_value)
        if first_death_p10 is not None:
            value = float(first_death_p10)
            self._p10_history.append(value)
            if len(self._p10_history) >= self.success_p10_median_window:
                recent = list(self._p10_history)[-self.success_p10_median_window :]
                if recent:
                    median_window_value = float(np.median(recent))
            if stage_p10_history is not None:
                stage_p10_history.append(value)
                if len(stage_p10_history) >= stage_p10_window:
                    span = max(stage_p10_window, 1)
                    recent_stage = list(stage_p10_history)[-span:]
                    if recent_stage:
                        stage_p10_median_value = float(np.median(recent_stage))
        early_death_median: Optional[float] = None
        stage_early_death_median: Optional[float] = None
        stage_history = self._stage_history(stage_type_value)
        stage_window_len = self._stage_window_size(stage_type_value)
        if earlydeath is not None:
            early_value = float(earlydeath)
            self._early_death_history.append(early_value)
            if len(self._early_death_history) >= self.success_early_death_window:
                recent_ed = list(self._early_death_history)[-self.success_early_death_window :]
                if recent_ed:
                    early_death_median = float(np.median(recent_ed))
            if stage_history is not None:
                stage_history.append(early_value)
                if len(stage_history) >= stage_window_len:
                    span = max(stage_window_len, 1)
                    stage_recent = list(stage_history)[-span:]
                    if stage_recent:
                        stage_early_death_median = float(np.median(stage_recent))

        success = False
        step_one_ok = True
        if self.success_step_one >= 0:
            step_one_ok = step_one is not None and step_one <= self.success_step_one
        if step_one_ok and self.success_step_one_ratio is not None:
            step_one_ok = step_one_ratio is not None and step_one_ratio <= self.success_step_one_ratio
        earlydeath_ok = False
        if step_one_ok and earlydeath is not None:
            earlydeath_ok = earlydeath <= stage_threshold
            if earlydeath_ok:
                median_candidate: Optional[float] = None
                if stage_history is not None and stage_window_len > 1:
                    median_candidate = stage_early_death_median
                elif self.success_early_death_window > 1:
                    median_candidate = early_death_median
                if median_candidate is None or median_candidate > stage_threshold:
                    earlydeath_ok = False
        if step_one_ok and earlydeath_ok:
            p10_ok = True
            target_p10 = self._stage_success_p10_threshold(stage_type_value)
            if target_p10 is not None:
                median_candidate = stage_p10_median_value if stage_p10_median_value is not None else median_window_value
                if median_candidate is None or median_candidate < target_p10:
                    p10_ok = False
            if p10_ok:
                success = True

        event = "noop"
        changed = False
        frozen = iteration < self.freeze_until_iteration

        if failure:
            self.failure_streak += 1
        else:
            self.failure_streak = 0

        stage_failure_tolerance = self._stage_failure_tolerance(stage_type_value)
        if failure and self.current_stage > 0 and self.failure_streak >= stage_failure_tolerance:
            prev_stage = self.current_stage
            self.current_stage = max(self.current_stage - 1, 0)
            self.success_streak = 0
            self.freeze_until_iteration = iteration + self.freeze_iterations
            event = "rollback"
            changed = prev_stage != self.current_stage
            self.failure_streak = 0
        elif failure:
            self.success_streak = 0
            self.freeze_until_iteration = iteration + self.freeze_iterations
            event = "failure_hold"
        elif iteration <= self.lock_until:
            self.success_streak = 0
            event = "locked"
        elif frozen:
            self.success_streak = 0
            event = "frozen"
        elif success:
            self.success_streak += 1
            success_record = {
                "iteration": iteration,
                "step_one": step_one,
                "step_one_ratio": step_one_ratio,
                "early_death_fraction_100": earlydeath,
                "first_death_p10": first_death_p10,
                "min_final_survival_rate": min_final,
            }
            self.last_success_metrics = success_record
            if self.success_freeze_iterations > 0:
                self.freeze_until_iteration = max(
                    self.freeze_until_iteration,
                    iteration + self.success_freeze_iterations,
                )
            if self.success_streak >= self.required_successes and self.current_stage < self.max_stage():
                prev_stage = self.current_stage
                self.current_stage += 1
                self.success_streak = 0
                self.freeze_until_iteration = iteration + self.freeze_iterations
                event = "advance"
                changed = prev_stage != self.current_stage
            else:
                event = "success_progress"
        else:
            self.success_streak = 0

        payload = {
            "event": event,
            "stage": self.current_stage,
            "phase_limit": self.current_phase_limit(),
            "success_streak": self.success_streak,
            "freeze_until": self.freeze_until_iteration,
            "failure_detected": failure,
            "failure_streak": self.failure_streak,
            "failure_tolerance": stage_failure_tolerance,
        }
        if median_window_value is not None:
            payload["rolling_p10_median"] = median_window_value
        if stage_p10_median_value is not None:
            payload["rolling_p10_median_stage"] = stage_p10_median_value
        if early_death_median is not None:
            payload["rolling_early_death_median"] = early_death_median
        if stage_early_death_median is not None:
            payload["rolling_early_death_median_stage"] = stage_early_death_median
        if stage_type_value is not None:
            payload["tail_stage_type"] = stage_type_value
        payload["early_death_threshold"] = stage_threshold
        payload["early_death_window"] = stage_window_len
        if step_one_ratio is not None:
            payload["step_one_death_ratio"] = step_one_ratio
        if self.last_success_metrics:
            payload["last_success"] = dict(self.last_success_metrics)
        if changed:
            payload["changed"] = True
        return payload


class EntropyCoefScheduler:
    """Piecewise-constant entropy coefficient scheduler keyed by iteration."""

    def __init__(self, base_value: float, windows: Sequence[Tuple[int, Optional[int], float]]):
        self.base_value = float(base_value)
        # sort by start iteration to ensure deterministic override order
        self.windows = sorted(list(windows), key=lambda item: item[0])
        self._current_value: Optional[float] = None

    @property
    def current_value(self) -> float:
        if self._current_value is None:
            return self.base_value
        return self._current_value

    def value_for_iteration(self, iteration: int) -> Optional[float]:
        value = self.base_value
        for start, end, window_value in self.windows:
            if iteration >= start and (end is None or iteration <= end):
                value = float(window_value)
        if self._current_value is None or abs(value - self._current_value) > 1e-9:
            self._current_value = value
            return value
        return None


class IterationStatsCallback(BaseCallback):
    """Collect stats per rollout iteration, manage schedulers, and run probes."""

    def __init__(
        self,
        *,
        run_name: str,
        stats_path: Path,
        log_path: Path,
        checkpoint_dir: Path,
        save_every: int = 5,
        density_penalty_scheduler: Optional[DensityPenaltyScheduler] = None,
        density_target: Optional[float] = None,
        initial_density_penalty: float = 0.0,
        entropy_scheduler: Optional[EntropyCoefScheduler] = None,
        initial_entropy_coef: float = 0.0,
        signals: Optional[TrainingSignals] = None,
        initial_escape_boost_speed: float = 0.0,
        schedule_trace_path: Optional[Path] = None,
        escape_boost_penalty_caps: Optional[Sequence[Tuple[float, float]]] = None,
        multi_eval_every: int = 0,
        multi_eval_kwargs: Optional[Dict[str, Any]] = None,
        multi_eval_history_path: Optional[Path] = None,
        multi_eval_plot_path: Optional[Path] = None,
        multi_eval_hist_dir: Optional[Path] = None,
        multi_eval_tb_writer: Optional[SummaryWriter] = None,
        adaptive_boost_controller: Optional[AdaptiveBoostController] = None,
        best_checkpoint_gate: Optional[Dict[str, float]] = None,
        penalty_gate: Optional[PenaltyStageGate] = None,
        penalty_stage_log_path: Optional[Path] = None,
        pre_roll_stats_path: Optional[Path] = None,
        step_one_cluster_path: Optional[Path] = None,
        tail_replay_count: int = 0,
        tail_replay_video_fps: int = 20,
        tail_replay_max_steps: int = MAX_STEPS,
        media_dir: Optional[Path] = None,
        multi_eval_seed_base: Optional[int] = None,
        tail_stage_tracker: Optional[TailStageTracker] = None,
        tail_seed_cycler: Optional["TailSeedCycler"] = None,
        tail_stage_warn_ratio: float = 0.5,
        tail_stage_warn_patience: int = 2,
    ):
        super().__init__(verbose=1)
        self.run_name = run_name
        self.stats_path = stats_path
        self.log_path = log_path
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.iteration = 0
        self.schedule_trace_path = schedule_trace_path
        self.stats = {
            "iterations": [],
            "survival_rate": [],
            "avg_reward": [],
            "episode_length": [],
            "avg_num_alive": [],
            "final_num_alive": [],
            "first_death_step": [],
            "first_death_p10": [],
            "first_death_sample_size": [],
            "density_penalty_coef": [],
            "entropy_coef": [],
            "escape_boost_speed": [],
            "step_one_deaths": [],
            "step_one_death_ratio": [],
            "prewarm_override_ratio": [],
            "prewarm_override_hits": [],
            "prewarm_override_episodes": [],
            "step_one_top_share": [],
            "step_one_ne_high_speed_share": [],
            "tail_stage_index": [],
            "tail_stage_progress": [],
            "tail_stage_label": [],
            "tail_stage_type": [],
            "tail_queue_remaining": [],
            "tail_queue_remaining_ratio": [],
            "tail_stage_remaining_by_type": [],
            "tail_stage_reinjections_by_type": [],
            "tail_ne_reinjections": [],
            "tail_stage_warn_active": [],
            "tail_stage_warn_since": [],
            "tail_stage_warn_duration": [],
            "tail_forced_resets": [],
            "tail_forced_resets_total": [],
            "tail_prewarm_injections": [],
            "tail_prewarm_injections_total": [],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        self.density_penalty_scheduler = density_penalty_scheduler
        self.density_target = density_target
        self.current_density_penalty = float(initial_density_penalty)
        self.entropy_scheduler = entropy_scheduler
        self.current_entropy_coef = float(initial_entropy_coef)
        self.signals = signals
        self._base_escape_boost_speed = float(initial_escape_boost_speed)
        self.current_escape_boost_speed = float(initial_escape_boost_speed)
        self.escape_boost_penalty_caps = sorted(list(escape_boost_penalty_caps or []), key=lambda item: item[0])
        self.multi_eval_every = max(int(multi_eval_every or 0), 0)
        self.multi_eval_kwargs = dict(multi_eval_kwargs or {})
        self.multi_eval_history_path = multi_eval_history_path
        self.multi_eval_plot_path = multi_eval_plot_path
        self.multi_eval_history: List[Dict[str, Any]] = []
        self.multi_eval_hist_dir = multi_eval_hist_dir
        self.multi_eval_tb_writer = multi_eval_tb_writer
        self.adaptive_boost_controller = adaptive_boost_controller
        self.best_checkpoint_gate = {
            key: float(value)
            for key, value in (best_checkpoint_gate or {}).items()
            if value is not None
        }
        self.penalty_gate = penalty_gate
        self.penalty_stage_log_path = penalty_stage_log_path
        self.pre_roll_stats_path = pre_roll_stats_path
        self.step_one_cluster_path = step_one_cluster_path
        self.step_one_records_all: List[Dict[str, Any]] = []
        self.tail_replay_count = max(int(tail_replay_count or 0), 0)
        self.tail_replay_video_fps = int(tail_replay_video_fps)
        self.tail_replay_max_steps = int(tail_replay_max_steps)
        self.media_dir = media_dir
        self.multi_eval_seed_rng = np.random.default_rng(multi_eval_seed_base)
        self.tail_stage_tracker = tail_stage_tracker
        self.tail_seed_cycler = tail_seed_cycler
        self._tail_consumed_total = 0
        self._tail_forced_resets_total = 0
        self._tail_injection_total = 0
        self.tail_stage_warn_ratio = float(tail_stage_warn_ratio)
        self.tail_stage_warn_patience = max(int(tail_stage_warn_patience or 1), 1)
        self._tail_stage_warn_hits = 0
        self._tail_stage_warn_active = False
        self._tail_stage_warn_since: Optional[int] = None
        self.tail_stage_warning_events: List[Dict[str, Any]] = []
        self._tail_stage_warning_event_cursor = 0
        self._tail_stage_warning_active_index: Optional[int] = None

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        if self._log_file and not self._log_file.closed:
            self._log_file.write(line)
            self._log_file.flush()
        print(line, end="")

    def _generate_multi_eval_seeds(self, episodes: int) -> List[int]:
        if episodes <= 0:
            return []
        seeds = []
        for _ in range(episodes):
            value = int(self.multi_eval_seed_rng.integers(0, 2**31 - 1))
            seeds.append(value)
        return seeds

    def _extract_metric(self, episodes: List[Dict], key: str, default: float = 0.0):
        values = [ep[key] for ep in episodes if key in ep]
        if not values:
            return default
        return float(np.mean(values))

    def _gather_first_death_samples(self, episodes: List[Dict]) -> List[float]:
        samples: List[float] = []
        for ep in episodes:
            candidate: Optional[float] = None
            death_steps = ep.get("death_timesteps")
            if death_steps is not None:
                if isinstance(death_steps, np.ndarray):
                    iterable = death_steps.tolist()
                else:
                    iterable = death_steps
                valid = [float(step) for step in iterable if step is not None and step >= 0]
                if valid:
                    candidate = float(min(valid))
            if candidate is None:
                value = ep.get("first_death_step")
                if value is not None and value >= 0:
                    candidate = float(value)
            if candidate is not None:
                samples.append(candidate)
        return samples

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        next_iteration = self.iteration + 1
        base_changed = self._sync_base_escape_boost()
        penalty_changed = self._maybe_update_density_penalty(next_iteration)
        self._maybe_update_entropy(next_iteration)
        self._apply_escape_boost_speed(force=base_changed or penalty_changed)

    def _sync_base_escape_boost(self) -> bool:
        if not self.signals:
            return False
        new_base = float(self.signals.escape_boost_speed)
        if abs(new_base - self._base_escape_boost_speed) < 1e-6:
            return False
        self._base_escape_boost_speed = new_base
        self._log(
            f"escape_boost_base iteration={self.iteration + 1:03d} base={self._base_escape_boost_speed:.3f}"
        )
        return True

    def _maybe_update_density_penalty(self, iteration_idx: int) -> bool:
        if not self.density_penalty_scheduler:
            self.logger.record("custom/density_penalty_coef", self.current_density_penalty)
            return False
        updated = False
        new_penalty = self.density_penalty_scheduler.value_for_iteration(iteration_idx)
        if new_penalty is not None:
            env = self.model.get_env() if self.model is not None else None
            if env is not None:
                if self.density_target is not None:
                    env.env_method("set_density_penalty", new_penalty, self.density_target)
                else:
                    env.env_method("set_density_penalty", new_penalty)
            self.current_density_penalty = float(new_penalty)
            self._log(
                f"density_penalty_update iteration={iteration_idx:03d} value={self.current_density_penalty:.4f}"
            )
            updated = True
        elif self.density_penalty_scheduler.current_value is not None:
            current = float(self.density_penalty_scheduler.current_value)
            if abs(current - self.current_density_penalty) > 1e-6:
                self.current_density_penalty = current
                updated = True
        self.logger.record("custom/density_penalty_coef", self.current_density_penalty)
        return updated

    def _maybe_update_entropy(self, iteration_idx: int) -> bool:
        if not self.entropy_scheduler:
            return False
        new_value = self.entropy_scheduler.value_for_iteration(iteration_idx)
        if new_value is None:
            return False
        self.current_entropy_coef = float(new_value)
        if self.model is not None:
            self.model.ent_coef = float(new_value)
        self._log(
            f"entropy_coef_update iteration={iteration_idx:03d} value={self.current_entropy_coef:.4f}"
        )
        return True

    def _resolve_escape_boost_speed(self, base_speed: float, penalty_value: float) -> float:
        return apply_penalty_caps(base_speed, penalty_value, self.escape_boost_penalty_caps)

    def _apply_escape_boost_speed(self, force: bool = False):
        if self.model is None:
            return
        desired = self._resolve_escape_boost_speed(self._base_escape_boost_speed, self.current_density_penalty)
        if not force and abs(desired - self.current_escape_boost_speed) < 1e-6:
            return
        self.current_escape_boost_speed = desired
        self.model.get_env().env_method("set_escape_boost_speed", desired)
        self._log(
            f"escape_boost_apply iteration={self.iteration + 1:03d} speed={self.current_escape_boost_speed:.3f}"
        )

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
        first_death_samples = self._gather_first_death_samples(episodes)
        if first_death_samples:
            first_death_step = float(np.mean(first_death_samples))
            first_death_p10 = float(np.percentile(first_death_samples, 10))
        else:
            first_death_step = -1.0
            first_death_p10 = None
        first_death_sample_size = len(first_death_samples)
        step_one_deaths = extract_step_one_deaths(episodes)

        self.stats["iterations"].append(self.iteration)
        self.stats["survival_rate"].append(survival_rate)
        self.stats["avg_reward"].append(avg_reward)
        self.stats["episode_length"].append(avg_length)
        self.stats["avg_num_alive"].append(avg_num_alive)
        self.stats["final_num_alive"].append(final_num_alive)
        self.stats["first_death_step"].append(first_death_step)
        self.stats["first_death_p10"].append(first_death_p10)
        self.stats["first_death_sample_size"].append(first_death_sample_size)
        self.stats["density_penalty_coef"].append(self.current_density_penalty)
        self.stats["entropy_coef"].append(self.current_entropy_coef)
        self.stats["escape_boost_speed"].append(self.current_escape_boost_speed)
        self.stats["step_one_deaths"].append(step_one_deaths)
        self.stats["step_one_death_ratio"].append(None)
        self.stats["step_one_top_share"].append(None)
        self.stats["step_one_ne_high_speed_share"].append(None)
        self.stats["tail_stage_index"].append(None)
        self.stats["tail_stage_progress"].append(None)
        self.stats["tail_stage_label"].append(None)
        self.stats["tail_stage_type"].append(None)
        self.stats["tail_queue_remaining"].append(None)
        self.stats["tail_queue_remaining_ratio"].append(None)
        self.stats["tail_stage_remaining_by_type"].append(None)
        self.stats["tail_stage_reinjections_by_type"].append(None)
        self.stats["tail_ne_reinjections"].append(None)
        self.stats["tail_stage_warn_active"].append(0)
        self.stats["tail_stage_warn_since"].append(None)
        self.stats["tail_stage_warn_duration"].append(0)
        self.stats["tail_forced_resets"].append(0)
        self.stats["tail_forced_resets_total"].append(None)
        self.stats["tail_prewarm_injections"].append(0)
        self.stats["tail_prewarm_injections_total"].append(None)
        if step_one_deaths:
            training_clusters = summarize_step_one_clusters(step_one_deaths)
            if training_clusters:
                clusters = training_clusters.get("clusters") or []
                top_share = float(clusters[0].get("share")) if clusters else None
                ne_share = None
                if clusters:
                    ne_share = compute_cluster_angle_speed_share(
                        clusters,
                        angle_min=90.0,
                        angle_max=150.0,
                        min_speed=1.8,
                    )
                self._record_cluster_shares(top_share, ne_share)
        prewarm_usage = self._collect_prewarm_usage()
        if prewarm_usage is None:
            self.stats["prewarm_override_ratio"].append(None)
            self.stats["prewarm_override_hits"].append(0)
            self.stats["prewarm_override_episodes"].append(0)
            self.stats["tail_forced_resets"][-1] = 0
            self.stats["tail_forced_resets_total"][-1] = None
            self.stats["tail_prewarm_injections"][-1] = 0
            self.stats["tail_prewarm_injections_total"][-1] = None
            self.logger.record("custom/tail_forced_resets", 0)
            self.logger.record("custom/tail_prewarm_injections", 0)
        else:
            ratio = prewarm_usage.get("ratio")
            self.stats["prewarm_override_ratio"].append(ratio)
            self.stats["prewarm_override_hits"].append(int(prewarm_usage.get("overrides", 0)))
            self.stats["prewarm_override_episodes"].append(int(prewarm_usage.get("episodes", 0)))
            if ratio is not None:
                self.logger.record("custom/prewarm_override_ratio", ratio)
            self.logger.record("custom/prewarm_override_hits", prewarm_usage.get("overrides", 0))
            self.logger.record("custom/prewarm_override_episodes", prewarm_usage.get("episodes", 0))
            forced_resets = int(prewarm_usage.get("forced_resets", 0) or 0)
            self.stats["tail_forced_resets"][-1] = forced_resets
            self.logger.record("custom/tail_forced_resets", forced_resets)
            forced_total = prewarm_usage.get("forced_resets_total")
            if forced_total is not None:
                forced_total = int(forced_total)
                self.stats["tail_forced_resets_total"][-1] = forced_total
                self.logger.record("custom/tail_forced_resets_total", forced_total)
            else:
                self.stats["tail_forced_resets_total"][-1] = None
            injections = int(prewarm_usage.get("injections", 0) or 0)
            self.stats["tail_prewarm_injections"][-1] = injections
            self.logger.record("custom/tail_prewarm_injections", injections)
            injection_total = prewarm_usage.get("injections_total")
            if injection_total is not None:
                injection_total = int(injection_total)
                self.stats["tail_prewarm_injections_total"][-1] = injection_total
                self.logger.record("custom/tail_prewarm_injections_total", injection_total)
            else:
                self.stats["tail_prewarm_injections_total"][-1] = None
        tail_stage_idx = None
        tail_stage_progress = None
        tail_queue_remaining = None
        tail_queue_ratio = None
        tail_stage_label = None
        tail_stage_type = None
        tail_stage_remaining_by_type = None
        tail_stage_reinjections_by_type = None
        tail_ne_reinjections = None
        if prewarm_usage:
            tail_stage = prewarm_usage.get("tail_stage") or {}
            tail_stage_idx = tail_stage.get("index")
            tail_stage_progress = tail_stage.get("progress")
            tail_queue_remaining = tail_stage.get("remaining")
            tail_queue_ratio = tail_stage.get("remaining_ratio")
            tail_stage_label = tail_stage.get("label")
            tail_stage_type = tail_stage.get("stage_type") or tail_stage.get("type")
            tail_stage_remaining_by_type = tail_stage.get("remaining_by_type")
            tail_stage_reinjections_by_type = tail_stage.get("reinjections_by_type")
            tail_ne_reinjections = tail_stage.get("ne_reinjections")
        self.stats["tail_stage_index"][-1] = tail_stage_idx
        self.stats["tail_stage_progress"][-1] = tail_stage_progress
        self.stats["tail_stage_label"][-1] = tail_stage_label
        self.stats["tail_stage_type"][-1] = tail_stage_type
        self.stats["tail_queue_remaining"][-1] = tail_queue_remaining
        self.stats["tail_queue_remaining_ratio"][-1] = tail_queue_ratio
        self.stats["tail_stage_remaining_by_type"][-1] = tail_stage_remaining_by_type
        self.stats["tail_stage_reinjections_by_type"][-1] = tail_stage_reinjections_by_type
        self.stats["tail_ne_reinjections"][-1] = tail_ne_reinjections
        if tail_stage_idx is not None:
            self.logger.record("custom/tail_stage_index", tail_stage_idx)
        if tail_stage_progress is not None:
            self.logger.record("custom/tail_stage_progress", tail_stage_progress)
        if tail_queue_remaining is not None:
            self.logger.record("custom/tail_queue_remaining", tail_queue_remaining)
        if tail_queue_ratio is not None:
            self.logger.record("custom/tail_queue_remaining_ratio", tail_queue_ratio)
        if tail_ne_reinjections is not None:
            self.logger.record("custom/tail_ne_reinjections", tail_ne_reinjections)
        warn_active, warn_since, warn_duration = self._update_tail_stage_warning(
            tail_queue_ratio,
            tail_stage_label,
            tail_stage_type,
            tail_queue_remaining,
        )
        self.logger.record("custom/tail_stage_warn_active", int(warn_active))
        if warn_since is not None:
            self.logger.record("custom/tail_stage_warn_since", int(warn_since))
        if warn_duration is not None:
            self.logger.record("custom/tail_stage_warn_duration", warn_duration)

        self.logger.record("custom/avg_survival_rate", survival_rate)
        self.logger.record("custom/final_num_alive", final_num_alive)
        self.logger.record("custom/avg_num_alive", avg_num_alive)
        self.logger.record("custom/first_death_step", first_death_step)
        if first_death_p10 is not None:
            self.logger.record("custom/first_death_p10", first_death_p10)
        self.logger.record("custom/first_death_sample_size", first_death_sample_size)
        self.logger.record("custom/entropy_coef", self.current_entropy_coef)
        self.logger.record("custom/escape_boost_speed", self.current_escape_boost_speed)
        self.logger.record("custom/step_one_death_count", len(step_one_deaths))

        self._log(
            f"iter={self.iteration:03d} sr={survival_rate:6.2%} "
            f"final_alive={final_num_alive:5.2f} reward={avg_reward:7.2f} steps={avg_length:6.1f}"
        )

        if self.iteration % self.save_every == 0:
            self._save_checkpoint(self.iteration)
            self._dump_stats()

        if first_death_p10 is not None:
            self._maybe_apply_adaptive_boost(first_death_p10)

        if self.multi_eval_every and self.multi_eval_kwargs:
            self._maybe_run_multi_eval()

        return True

    def _maybe_apply_adaptive_boost(self, first_death_p10: float):
        if not self.adaptive_boost_controller or not self.signals:
            return
        result = self.adaptive_boost_controller.update(self.iteration, first_death_p10)
        if result is None:
            return
        target_speed, window_metric = result
        phase_base = getattr(self.signals, "phase_base_escape_boost_speed", self._base_escape_boost_speed)
        phase_floor = getattr(self.signals, "phase_floor_escape_boost_speed", None)
        stage_floor = None
        if hasattr(self.adaptive_boost_controller, "stage_floor"):
            stage_floor = self.adaptive_boost_controller.stage_floor()
        floor_value = None
        if phase_floor is not None:
            floor_value = float(phase_floor)
        if stage_floor is not None:
            floor_value = max(floor_value if floor_value is not None else stage_floor, stage_floor)
        if floor_value is not None:
            phase_base = max(phase_base, floor_value)
        desired_speed = min(phase_base, target_speed)
        if floor_value is not None:
            desired_speed = max(desired_speed, floor_value)
        if abs(desired_speed - self.signals.escape_boost_speed) < 1e-6:
            return
        self.signals.escape_boost_speed = desired_speed
        self._base_escape_boost_speed = desired_speed
        self._log(
            f"adaptive_boost iteration={self.iteration:03d} window_p10={window_metric:.2f} target={desired_speed:.3f}"
        )
        self._apply_escape_boost_speed(force=True)

    def _save_checkpoint(self, iteration: int, *, force: bool = False) -> Optional[Path]:
        if self.model is None:
            return None
        path = self.checkpoint_dir / f"model_iter_{iteration}"
        zip_path = path.with_suffix(".zip")
        if zip_path.exists() and not force:
            return zip_path
        self.model.save(path)
        self._log(f"checkpoint_saved iteration={iteration}")
        return zip_path

    def _dump_stats(self):
        with open(self.stats_path, "wb") as f:
            pickle.dump(self.stats, f)
        self._dump_schedule_trace()

    def _dump_schedule_trace(self):
        if not self.schedule_trace_path:
            return
        trace = []
        total = len(self.stats["iterations"])
        step_one_seq = self.stats.get("step_one_deaths", [])
        for idx in range(total):
            entry = {
                "iteration": int(self.stats["iterations"][idx]),
                "density_penalty_coef": self._safe_trace_value(self.stats["density_penalty_coef"], idx),
                "entropy_coef": self._safe_trace_value(self.stats["entropy_coef"], idx),
                "escape_boost_speed": self._safe_trace_value(self.stats["escape_boost_speed"], idx),
                "survival_rate": self._safe_trace_value(self.stats["survival_rate"], idx),
                "first_death_step": self._safe_trace_value(self.stats["first_death_step"], idx),
                "first_death_p10": self._safe_trace_value(self.stats["first_death_p10"], idx),
                "first_death_sample_size": self._safe_trace_value(self.stats["first_death_sample_size"], idx),
                "prewarm_override_ratio": self._safe_trace_value(self.stats["prewarm_override_ratio"], idx),
                "prewarm_override_hits": self._safe_trace_value(self.stats["prewarm_override_hits"], idx),
                "prewarm_override_episodes": self._safe_trace_value(self.stats["prewarm_override_episodes"], idx),
                "step_one_death_ratio": self._safe_trace_value(self.stats["step_one_death_ratio"], idx),
                "step_one_top_share": self._safe_trace_value(self.stats["step_one_top_share"], idx),
                "step_one_ne_high_speed_share": self._safe_trace_value(
                    self.stats["step_one_ne_high_speed_share"], idx
                ),
                "tail_stage_index": self._safe_trace_value(self.stats["tail_stage_index"], idx),
                "tail_stage_progress": self._safe_trace_value(self.stats["tail_stage_progress"], idx),
                "tail_stage_label": self._safe_trace_text(self.stats["tail_stage_label"], idx),
                "tail_stage_type": self._safe_trace_text(self.stats.get("tail_stage_type", []), idx),
                "tail_queue_remaining": self._safe_trace_value(self.stats["tail_queue_remaining"], idx),
                "tail_queue_remaining_ratio": self._safe_trace_value(
                    self.stats["tail_queue_remaining_ratio"], idx
                ),
                "tail_stage_remaining_by_type": (
                    self.stats["tail_stage_remaining_by_type"][idx]
                    if idx < len(self.stats["tail_stage_remaining_by_type"])
                    else None
                ),
                "tail_stage_reinjections_by_type": (
                    self.stats["tail_stage_reinjections_by_type"][idx]
                    if idx < len(self.stats["tail_stage_reinjections_by_type"])
                    else None
                ),
                "tail_ne_reinjections": self._safe_trace_value(
                    self.stats["tail_ne_reinjections"], idx
                ),
                "tail_stage_warn_active": self._safe_trace_value(
                    self.stats["tail_stage_warn_active"], idx
                ),
                "tail_stage_warn_since": self._safe_trace_value(
                    self.stats["tail_stage_warn_since"], idx
                ),
                "tail_stage_warn_duration": self._safe_trace_value(
                    self.stats["tail_stage_warn_duration"], idx
                ),
                "tail_forced_resets": self._safe_trace_value(self.stats["tail_forced_resets"], idx),
                "tail_forced_resets_total": self._safe_trace_value(
                    self.stats["tail_forced_resets_total"], idx
                ),
                "tail_prewarm_injections": self._safe_trace_value(
                    self.stats["tail_prewarm_injections"], idx
                ),
                "tail_prewarm_injections_total": self._safe_trace_value(
                    self.stats["tail_prewarm_injections_total"], idx
                ),
            }
            records = []
            if idx < len(step_one_seq):
                records = step_one_seq[idx] or []
            entry["step_one_death_count"] = len(records)
            if records:
                entry["step_one_deaths"] = records
            trace.append(entry)
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "trace": trace,
        }
        if self.tail_stage_warning_events:
            payload["tail_stage_warnings"] = self.tail_stage_warning_events
        self.schedule_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.schedule_trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _safe_trace_value(seq: List[float], idx: int) -> Optional[float]:
        if idx >= len(seq):
            return None
        value = seq[idx]
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _safe_trace_text(seq: List[Any], idx: int) -> Optional[str]:
        if idx >= len(seq):
            return None
        value = seq[idx]
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text

    def _collect_prewarm_usage(self) -> Optional[Dict[str, float]]:
        if self.model is None:
            return None
        env = self.model.get_env()
        if env is None:
            return None
        try:
            stats_list = env.env_method("consume_prewarm_stats")
        except AttributeError:
            return None
        except Exception:
            return None
        total_episodes = 0
        total_overrides = 0
        cumulative_episodes = 0
        forced_since = 0
        per_env_overrides: List[int] = []
        iteration_injections = 0
        for entry in stats_list or []:
            if not entry:
                per_env_overrides.append(0)
                continue
            episodes = int(entry.get("episodes", 0) or 0)
            overrides = int(entry.get("overrides", 0) or 0)
            total_episodes += episodes
            total_overrides += overrides
            cumulative_episodes += int(entry.get("total_episodes", 0) or 0)
            forced_since += int(entry.get("forced_resets", 0) or 0)
            iteration_injections += int(entry.get("injections", 0) or 0)
            per_env_overrides.append(overrides)
        if per_env_overrides:
            if self.tail_seed_cycler:
                self.tail_seed_cycler.record_overrides(per_env_overrides)
                self._tail_consumed_total = self.tail_seed_cycler.total_consumed
            else:
                self._tail_consumed_total += sum(per_env_overrides)
        ratio = None
        if total_episodes > 0:
            ratio = float(total_overrides / max(total_episodes, 1))
        tail_stage = None
        if self.tail_stage_tracker:
            self.tail_stage_tracker.update_total_consumed(self._tail_consumed_total)
            tail_stage = self.tail_stage_tracker.status()
        self._tail_forced_resets_total += forced_since
        self._tail_injection_total += iteration_injections
        return {
            "episodes": total_episodes,
            "overrides": total_overrides,
            "ratio": ratio,
            "total_episodes": cumulative_episodes,
            "total_overrides": self._tail_consumed_total,
            "forced_resets": forced_since,
            "forced_resets_total": self._tail_forced_resets_total,
            "injections": iteration_injections,
            "injections_total": self._tail_injection_total,
            "tail_stage": tail_stage,
        }

    def _resolve_tail_stage_warning_event(self, iteration: int, tail_queue_ratio: Optional[float]):
        if self._tail_stage_warning_active_index is None:
            return
        if self._tail_stage_warning_active_index < 0 or self._tail_stage_warning_active_index >= len(self.tail_stage_warning_events):
            self._tail_stage_warning_active_index = None
            return
        event = self.tail_stage_warning_events[self._tail_stage_warning_active_index]
        event["resolved_iteration"] = iteration
        start_iter = event.get("iteration")
        try:
            start_val = int(start_iter)
        except (TypeError, ValueError):
            start_val = None
        if start_val is not None:
            event["duration_iterations"] = int(iteration - start_val + 1)
        if tail_queue_ratio is not None:
            event["clear_ratio"] = float(tail_queue_ratio)
        self._tail_stage_warning_active_index = None

    def _update_tail_stage_warning(
        self,
        tail_queue_ratio: Optional[float],
        tail_stage_label: Optional[str],
        tail_stage_type: Optional[str],
        tail_queue_remaining: Optional[int],
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        threshold = self.tail_stage_warn_ratio
        previous_active = self._tail_stage_warn_active
        ratio_valid = tail_queue_ratio is not None and threshold > 0
        if ratio_valid:
            if tail_queue_ratio <= threshold:
                self._tail_stage_warn_hits += 1
                if not self._tail_stage_warn_active and self._tail_stage_warn_hits >= self.tail_stage_warn_patience:
                    self._tail_stage_warn_active = True
                    self._tail_stage_warn_since = self.iteration
                    event = {
                        "iteration": self.iteration,
                        "tail_stage_label": tail_stage_label,
                        "tail_stage_type": tail_stage_type,
                        "remaining_ratio": float(tail_queue_ratio),
                        "remaining": int(tail_queue_remaining) if tail_queue_remaining is not None else None,
                        "threshold": float(threshold),
                    }
                    self.tail_stage_warning_events.append(event)
                    self._tail_stage_warning_active_index = len(self.tail_stage_warning_events) - 1
                    label = tail_stage_label or "unknown"
                    self._log(
                        f"[tail-stage] warning triggered iteration={self.iteration:03d} "
                        f"label={label} ratio={tail_queue_ratio:.3f} <= {threshold:.3f}"
                    )
            else:
                if self._tail_stage_warn_active:
                    self._resolve_tail_stage_warning_event(self.iteration, tail_queue_ratio)
                self._tail_stage_warn_hits = 0
                self._tail_stage_warn_active = False
                self._tail_stage_warn_since = None
        else:
            if self._tail_stage_warn_active:
                self._resolve_tail_stage_warning_event(self.iteration, tail_queue_ratio)
            self._tail_stage_warn_hits = 0
            self._tail_stage_warn_active = False
            self._tail_stage_warn_since = None

        if previous_active and not self._tail_stage_warn_active:
            ratio_display = "n/a" if tail_queue_ratio is None else f"{tail_queue_ratio:.3f}"
            self._log(
                f"[tail-stage] warning cleared iteration={self.iteration:03d} ratio={ratio_display}"
            )

        warn_active = self._tail_stage_warn_active
        warn_since = self._tail_stage_warn_since
        self.stats["tail_stage_warn_active"][-1] = 1 if warn_active else 0
        self.stats["tail_stage_warn_since"][-1] = warn_since
        warn_duration: Optional[int]
        if warn_active and warn_since is not None:
            warn_duration = int(self.iteration - warn_since + 1)
        elif warn_active:
            warn_duration = 1
        else:
            warn_duration = 0
        self.stats["tail_stage_warn_duration"][-1] = warn_duration
        return warn_active, warn_since, warn_duration

    def _consume_tail_stage_warning_events(self) -> List[Dict[str, Any]]:
        if self._tail_stage_warning_event_cursor >= len(self.tail_stage_warning_events):
            return []
        events = self.tail_stage_warning_events[self._tail_stage_warning_event_cursor :]
        self._tail_stage_warning_event_cursor = len(self.tail_stage_warning_events)
        return [dict(event) for event in events]

    def _record_cluster_shares(
        self,
        top_share: Optional[float],
        ne_high_speed_share: Optional[float],
    ) -> None:
        if not self.stats.get("step_one_top_share"):
            return
        if top_share is not None:
            self.stats["step_one_top_share"][-1] = float(top_share)
            self.logger.record("custom/step_one_top_share", float(top_share))
        if ne_high_speed_share is not None:
            self.stats["step_one_ne_high_speed_share"][-1] = float(ne_high_speed_share)
            self.logger.record("custom/step_one_ne_high_speed_share", float(ne_high_speed_share))
        stage_type = None
        if self.stats.get("tail_stage_type"):
            stage_type = self.stats["tail_stage_type"][-1]
        slug = stage_type_slug(stage_type)
        if slug:
            if top_share is not None:
                self.logger.record(f"custom/step_one_top_share_{slug}", float(top_share))
            if ne_high_speed_share is not None:
                self.logger.record(f"custom/step_one_ne_high_speed_share_{slug}", float(ne_high_speed_share))

    def _record_step_one_ratio(self, ratio: Optional[float]) -> None:
        if not self.stats["step_one_death_ratio"]:
            return
        if ratio is None:
            return
        self.stats["step_one_death_ratio"][-1] = float(ratio)
        self.logger.record("custom/step_one_death_ratio", float(ratio))

    def _on_training_end(self) -> None:
        # 多阶段训练时保持日志句柄，最终由 finalize() 关闭
        pass

    def finalize(self):
        if self._tail_stage_warn_active:
            self._resolve_tail_stage_warning_event(self.iteration, None)
        self._dump_stats()
        if not self._log_file.closed:
            self._log_file.close()

    def _maybe_run_multi_eval(self):
        if self.iteration == 0:
            return
        if self.iteration % self.multi_eval_every != 0:
            return
        if self.model is None:
            return
        eval_kwargs = dict(self.multi_eval_kwargs)
        eval_kwargs["escape_boost_speed"] = self.current_escape_boost_speed
        eval_kwargs["density_penalty_coef"] = self.current_density_penalty
        episode_count = int(eval_kwargs.get("episodes", 0) or 0)
        num_fish = int(eval_kwargs.get("num_fish", 0) or 0)
        seeds = self._generate_multi_eval_seeds(episode_count)
        results = evaluate_multi_fish(self.model, seeds=seeds, **eval_kwargs)
        summary = summarize_multi_eval(results, eval_kwargs.get("num_fish", 0))
        death_stats = None
        death_hist_path: Optional[Path] = None
        death_steps: List[int] = []
        if results:
            death_steps = collect_death_timesteps(results, MAX_STEPS)
            death_stats = summarize_death_timesteps(death_steps, MAX_STEPS)
            if self.multi_eval_hist_dir and death_steps:
                self.multi_eval_hist_dir.mkdir(parents=True, exist_ok=True)
                death_hist_path = self.multi_eval_hist_dir / f"{self.run_name}_iter{self.iteration:03d}_death_hist.png"
                plot_death_histogram(death_steps, self.run_name, death_hist_path, MAX_STEPS)
        step_one_death_records = extract_step_one_deaths(results)
        denom = episode_count * num_fish
        step_one_ratio = None
        if denom > 0:
            step_one_ratio = float(len(step_one_death_records)) / float(denom)
        entry: Dict[str, Any] = {
            "iteration": int(self.iteration),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": summary,
            "death_stats": death_stats,
            "density_penalty_coef": float(self.current_density_penalty),
            "escape_boost_speed": float(self.current_escape_boost_speed),
            "step_one_death_count": len(step_one_death_records),
             "step_one_death_ratio": step_one_ratio,
            "episode_seeds": seeds,
            "multi_eval_num_fish": num_fish,
            "multi_eval_episodes": episode_count,
        }
        if self.stats.get("tail_stage_label"):
            entry["tail_stage_label"] = self.stats["tail_stage_label"][-1]
        if self.stats.get("tail_stage_type"):
            entry["tail_stage_type"] = self.stats["tail_stage_type"][-1]
        if self.stats.get("tail_stage_remaining_by_type"):
            entry["tail_stage_remaining_by_type"] = self.stats["tail_stage_remaining_by_type"][-1]
        if self.stats.get("tail_stage_warn_active"):
            entry["tail_stage_warn_active"] = bool(self.stats["tail_stage_warn_active"][-1])
        if self.stats.get("tail_stage_warn_since"):
            entry["tail_stage_warn_since"] = self.stats["tail_stage_warn_since"][-1]
        warning_events = self._consume_tail_stage_warning_events()
        if warning_events:
            entry["tail_stage_warning_events"] = warning_events
        self._record_step_one_ratio(step_one_ratio)
        if death_hist_path is not None:
            entry["death_hist_path"] = str(death_hist_path)
        if step_one_death_records:
            entry["step_one_deaths"] = step_one_death_records
            self._record_tail_events(step_one_death_records)
        cluster_summary = summarize_step_one_clusters(step_one_death_records)
        if cluster_summary:
            entry["step_one_clusters"] = cluster_summary
            clusters = cluster_summary.get("clusters") or []
            top_share = float(clusters[0].get("share")) if clusters else None
            ne_share = None
            if clusters:
                ne_share = compute_cluster_angle_speed_share(
                    clusters,
                    angle_min=90.0,
                    angle_max=150.0,
                    min_speed=1.8,
                )
            if ne_share is not None:
                entry["step_one_ne_high_speed_share"] = float(ne_share)
            self._record_cluster_shares(top_share, ne_share)
            if self.step_one_cluster_path:
                append_jsonl(
                    self.step_one_cluster_path,
                    {
                        "iteration": int(self.iteration),
                        "timestamp": entry["timestamp"],
                        "clusters": cluster_summary,
                        "tail_stage_label": entry.get("tail_stage_label"),
                        "tail_stage_type": entry.get("tail_stage_type"),
                    },
                )
        pre_roll_samples = collect_pre_roll_samples(results)
        if pre_roll_samples:
            pre_roll_summary = summarize_pre_roll_samples(pre_roll_samples)
            entry["pre_roll_stats_summary"] = pre_roll_summary
            if self.pre_roll_stats_path:
                append_jsonl(
                    self.pre_roll_stats_path,
                    {
                        "iteration": int(self.iteration),
                        "timestamp": entry["timestamp"],
                        "samples": pre_roll_samples,
                        "summary": pre_roll_summary,
                    },
                )
        if self._best_checkpoint_gate_satisfied(entry):
            checkpoint_zip = self._save_checkpoint(self.iteration)
            if checkpoint_zip:
                entry["best_model_path"] = str(checkpoint_zip)
                entry["best_checkpoint_saved"] = True
                early_val = None
                if death_stats:
                    early_val = death_stats.get("early_death_fraction_100")
                early_str = "n/a" if early_val is None else f"{float(early_val):.3f}"
                self._log(
                    f"best_checkpoint_auto_save iteration={self.iteration:03d} step_one={len(step_one_death_records)} "
                    f"early_death={early_str} path={checkpoint_zip.name}"
                )
        if self.penalty_gate and self.density_penalty_scheduler:
            gate_event = self.penalty_gate.handle_multi_eval(self.iteration, entry)
            if gate_event:
                entry["penalty_gate"] = gate_event
                if self.adaptive_boost_controller is not None:
                    self.adaptive_boost_controller.set_stage(gate_event.get("stage"))
                phase_limit = gate_event.get("phase_limit")
                if phase_limit is not None:
                    self.density_penalty_scheduler.set_max_active_phase(int(phase_limit))
                event_name = gate_event.get("event")
                if event_name and event_name not in {"noop", "locked"}:
                    self._log(
                        "penalty_gate event="
                        f"{event_name} iteration={self.iteration:03d} stage={gate_event.get('stage')} "
                        f"phase_limit={phase_limit} freeze_until={gate_event.get('freeze_until')}"
                    )
                if self.penalty_stage_log_path:
                    append_jsonl(
                        self.penalty_stage_log_path,
                        {
                            "iteration": int(self.iteration),
                            "timestamp": entry["timestamp"],
                            "event": gate_event,
                            "summary": summary,
                            "death_stats": death_stats,
                        },
                    )
        self.multi_eval_history.append(entry)
        if self.multi_eval_history_path:
            self.multi_eval_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.multi_eval_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        if self.multi_eval_plot_path:
            plot_multi_eval_history(self.multi_eval_history, self.multi_eval_plot_path)
        if self.multi_eval_tb_writer and summary:
            global_step = self.iteration
            for key, value in summary.items():
                if value is None:
                    continue
                self.multi_eval_tb_writer.add_scalar(f"multi_eval/{key}", float(value), global_step)
            if death_stats:
                for key, value in death_stats.items():
                    if isinstance(value, (int, float)):
                        self.multi_eval_tb_writer.add_scalar(f"multi_eval_death/{key}", float(value), global_step)
            ne_share = entry.get("step_one_ne_high_speed_share")
            if ne_share is not None:
                self.multi_eval_tb_writer.add_scalar(
                    "multi_eval/step_one_ne_high_speed_share",
                    float(ne_share),
                    global_step,
                )
            self.multi_eval_tb_writer.flush()

        tail_replays = self._maybe_record_tail_replays(results, seeds)
        if tail_replays:
            entry["tail_replays"] = tail_replays

    def _record_tail_events(self, records: Sequence[Dict[str, Any]]) -> None:
        stage_label = self.stats["tail_stage_label"][-1] if self.stats.get("tail_stage_label") else None
        stage_type = self.stats["tail_stage_type"][-1] if self.stats.get("tail_stage_type") else None
        for record in records:
            enriched = dict(record)
            enriched["iteration"] = int(self.iteration)
            if stage_label is not None:
                enriched["tail_stage_label"] = stage_label
            if stage_type is not None:
                enriched["tail_stage_type"] = stage_type
            self.step_one_records_all.append(enriched)

    def _maybe_record_tail_replays(
        self,
        results: Optional[Sequence[Dict[str, Any]]],
        seeds: Optional[Sequence[int]],
    ) -> List[Dict[str, Any]]:
        if (
            self.tail_replay_count <= 0
            or not results
            or self.media_dir is None
            or self.model is None
            or not self.multi_eval_kwargs
        ):
            return []
        video_kwargs = dict(self.multi_eval_kwargs)
        video_kwargs.pop("episodes", None)
        num_fish = int(video_kwargs.pop("num_fish", 0) or 0)
        if num_fish <= 0:
            return []
        ranked: List[Tuple[Dict[str, Any], Optional[int]]] = []
        for idx, episode in enumerate(results):
            seed_value = None
            if seeds is not None and idx < len(seeds):
                seed_value = int(seeds[idx])
            elif episode.get("seed") is not None:
                seed_value = int(episode["seed"])
            ranked.append((episode, seed_value))
        ranked.sort(
            key=lambda pair: (
                float(pair[0].get("final_num_alive", float("inf"))),
                float(pair[0].get("min_survival_rate", float("inf"))),
            )
        )
        recorded: List[Dict[str, Any]] = []
        limit = min(self.tail_replay_count, len(ranked))
        for rank_idx in range(limit):
            episode, seed_value = ranked[rank_idx]
            video_name = (
                f"{self.run_name}_iter{self.iteration:03d}_tail_rank{rank_idx}_ep{episode.get('episode', rank_idx)}.mp4"
            )
            video_path = self.media_dir / video_name
            record_multi_fish_video(
                self.model,
                num_fish=num_fish,
                video_path=video_path,
                max_steps=self.tail_replay_max_steps,
                fps=self.tail_replay_video_fps,
                include_neighbor_features=bool(video_kwargs.get("include_neighbor_features", False)),
                neighbor_radius=video_kwargs.get("neighbor_radius", 2.0),
                neighbor_average_count=video_kwargs.get("neighbor_average_count", 6),
                initial_escape_boost=bool(video_kwargs.get("initial_escape_boost", False)),
                escape_boost_speed=video_kwargs.get("escape_boost_speed", 0.6),
                escape_jitter_std=video_kwargs.get("escape_jitter_std", np.pi / 12),
                divergence_reward_coef=video_kwargs.get("divergence_reward_coef", 0.0),
                density_penalty_coef=video_kwargs.get("density_penalty_coef", 0.0),
                density_target=video_kwargs.get("density_target", 0.4),
                predator_spawn_jitter_radius=video_kwargs.get("predator_spawn_jitter_radius", 0.0),
                predator_pre_roll_steps=video_kwargs.get("predator_pre_roll_steps", 0),
                predator_pre_roll_angle_jitter=video_kwargs.get("predator_pre_roll_angle_jitter", 0.0),
                predator_pre_roll_speed_jitter=video_kwargs.get("predator_pre_roll_speed_jitter", 0.0),
                predator_heading_bias=video_kwargs.get("predator_heading_bias"),
                seed=seed_value,
                predator_pre_roll_speed_bias=video_kwargs.get("predator_pre_roll_speed_bias"),
            )
            recorded.append(
                {
                    "video_path": str(video_path),
                    "seed": seed_value,
                    "episode": int(episode.get("episode", rank_idx)),
                    "final_num_alive": float(episode.get("final_num_alive", 0.0)),
                }
            )
        return recorded

    def _best_checkpoint_gate_satisfied(self, entry: Dict[str, Any]) -> bool:
        if not self.best_checkpoint_gate:
            return False
        step_one_limit = self.best_checkpoint_gate.get("max_step_one")
        if step_one_limit is not None:
            step_one_count = entry.get("step_one_death_count")
            if step_one_count is None:
                return False
            try:
                if float(step_one_count) > step_one_limit:
                    return False
            except (TypeError, ValueError):
                return False
        early_limit = self.best_checkpoint_gate.get("max_early_death")
        if early_limit is not None:
            death_stats = entry.get("death_stats") or {}
            early_value = death_stats.get("early_death_fraction_100")
            if early_value is None:
                return False
            try:
                if float(early_value) > early_limit:
                    return False
            except (TypeError, ValueError):
                return False
        return True


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
    predator_spawn_jitter_radius: float,
    predator_pre_roll_steps: int,
    predator_pre_roll_angle_jitter: float,
    predator_pre_roll_speed_jitter: float,
    predator_heading_bias: Optional[Sequence[Dict[str, float]]],
    predator_pre_roll_speed_bias: Optional[Sequence[Dict[str, float]]] = None,
    prewarm_predator_velocity_seqs: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    tail_force_reset_steps: int = 0,
):
    def _factory(rank: int):
        def _init():
            seed = None if base_seed is None else base_seed + rank
            prewarm_payload: Optional[Sequence[Sequence[float]]] = None
            initial_offset = 0
            if prewarm_predator_velocity_seqs is not None and rank < len(prewarm_predator_velocity_seqs):
                entry = prewarm_predator_velocity_seqs[rank]
                seq = entry
                if isinstance(entry, dict):
                    seq = entry.get("velocities") or entry.get("sequence")
                    initial_offset = int(entry.get("offset", 0) or 0)
                if seq:
                    prepared = []
                    for item in seq:
                        if item is None or len(item) < 2:
                            continue
                        prepared.append((float(item[0]), float(item[1])))
                    if prepared:
                        prewarm_payload = prepared
                        initial_offset = max(initial_offset, 0)
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
                predator_spawn_jitter_radius=predator_spawn_jitter_radius,
                predator_pre_roll_steps=predator_pre_roll_steps,
                predator_pre_roll_angle_jitter=predator_pre_roll_angle_jitter,
                predator_pre_roll_speed_jitter=predator_pre_roll_speed_jitter,
                predator_heading_bias=predator_heading_bias,
                predator_pre_roll_speed_bias=predator_pre_roll_speed_bias,
                prewarm_predator_velocities=prewarm_payload,
                tail_force_reset_steps=tail_force_reset_steps,
                tail_seed_offset=initial_offset,
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
    exp_tag = EXPERIMENT_DIR.name
    plt.title(f"Fish RL {exp_tag} - {run_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_first_death_curve(stats: Dict[str, List[float]], run_name: str, plot_path: Path):
    if not stats["iterations"]:
        return
    iterations = stats["iterations"]
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, stats["first_death_step"], color="tab:red", label="first_death_step")
    first_death_p10 = stats.get("first_death_p10") or []
    if first_death_p10:
        p10_series = [np.nan if value is None else value for value in first_death_p10]
        plt.plot(iterations[: len(p10_series)], p10_series, color="tab:orange", linestyle="--", label="first_death_p10")
    plt.xlabel("Iteration")
    plt.ylabel("First death step")
    plt.title(f"First-death progression - {run_name}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_penalty_alignment(stats: Dict[str, List[float]], run_name: str, plot_path: Path):
    iterations = stats.get("iterations")
    penalties = stats.get("density_penalty_coef")
    first_death = stats.get("first_death_step")
    if not iterations or not penalties or not first_death:
        return
    if len(penalties) != len(iterations) or len(first_death) != len(iterations):
        return
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax1.plot(iterations, penalties, color="tab:purple", label="density_penalty")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Density penalty coef", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax2 = ax1.twinx()
    ax2.plot(iterations, first_death, color="tab:red", label="first_death")
    ax2.set_ylabel("First death step", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title(f"Penalty vs first-death - {run_name}")
    ax1.grid(True, linestyle="--", alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_penalty_entropy(stats: Dict[str, List[float]], run_name: str, plot_path: Path):
    iterations = stats.get("iterations")
    penalties = stats.get("density_penalty_coef")
    entropy = stats.get("entropy_coef")
    if not iterations or not penalties or not entropy:
        return
    if len(iterations) != len(penalties) or len(iterations) != len(entropy):
        return
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax1.plot(iterations, penalties, label="density_penalty", color="tab:purple")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Density penalty", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax2 = ax1.twinx()
    ax2.plot(iterations, entropy, label="entropy_coef", color="tab:green")
    ax2.set_ylabel("Entropy coef", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    plt.title(f"Penalty vs entropy - {run_name}")
    ax1.grid(True, linestyle="--", alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_tail_stage_warnings(
    stats: Dict[str, List[Any]],
    run_name: str,
    plot_path: Path,
    threshold: Optional[float] = None,
):
    iterations = stats.get("iterations") or []
    remaining_ratio = stats.get("tail_queue_remaining_ratio") or []
    warn_active = stats.get("tail_stage_warn_active") or []
    warn_duration = stats.get("tail_stage_warn_duration") or []
    if not iterations or not remaining_ratio:
        return
    count = min(len(iterations), len(remaining_ratio))
    xs = iterations[:count]
    ratios = remaining_ratio[:count]
    warn_active = warn_active[:count] if warn_active else [0] * count
    warn_duration = warn_duration[:count] if warn_duration else [0] * count
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(xs, ratios, color="tab:blue", label="tail queue remaining ratio")
    if threshold is not None and threshold > 0:
        ax1.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label=f"warn threshold={threshold:.2f}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Tail queue remaining ratio")
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(xs, warn_duration, color="tab:orange", alpha=0.25, label="warning duration")
    ax2.set_ylabel("Active warning duration (iters)")

    active_spans: List[Tuple[int, int]] = []
    current_start: Optional[int] = None
    for idx, flag in enumerate(warn_active):
        if flag and current_start is None:
            current_start = xs[idx]
        elif not flag and current_start is not None:
            active_spans.append((current_start, xs[idx]))
            current_start = None
    if current_start is not None:
        active_spans.append((current_start, xs[-1]))
    for start, end in active_spans:
        ax1.axvspan(start, end, color="tab:red", alpha=0.08)

    lines_labels = [ax.get_legend_handles_labels() for ax in (ax1, ax2)]
    lines, labels = [sum(items, []) for items in zip(*lines_labels)]
    if lines and labels:
        ax1.legend(lines, labels, loc="upper right")
    plt.title(f"Tail stage warning monitor - {run_name}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)


def plot_step_one_heading_polar(
    records: Sequence[Dict[str, Any]],
    run_name: str,
    plot_path: Path,
    bins: int = 12,
):
    if not records:
        return
    angles: List[float] = []
    weights: List[float] = []
    for record in records:
        velocity = record.get("predator_velocity") or []
        if len(velocity) < 2:
            continue
        vx = float(velocity[0])
        vy = float(velocity[1])
        speed = math.sqrt(vx * vx + vy * vy)
        if speed <= 1e-9:
            continue
        angle_rad = math.atan2(vy, vx) % (2 * math.pi)
        angles.append(angle_rad)
        weights.append(speed)
    if not angles:
        return
    bin_count = max(int(bins), 4)
    bin_edges = np.linspace(0.0, 2.0 * math.pi, bin_count + 1)
    hist, _ = np.histogram(angles, bins=bin_edges, weights=weights if weights else None)
    centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    widths = np.diff(bin_edges)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    ax.bar(centers, hist, width=widths, bottom=0.0, align="center", alpha=0.7, color="tab:orange")
    ax.set_title(f"Step-one heading polar - {run_name}")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def summarize_worst_step_one_clusters(
    records: Sequence[Dict[str, Any]],
    angle_bin_deg: float = 30.0,
    speed_bin: float = 0.4,
    top_k: int = 6,
) -> Optional[Dict[str, Any]]:
    cluster_summary = summarize_step_one_clusters(records, angle_bin_deg=angle_bin_deg, speed_bin=speed_bin)
    if not cluster_summary:
        return None
    clusters = cluster_summary.get("clusters") or []
    if not clusters:
        return None
    selected = clusters[: max(int(top_k), 1)]

    def _bin_indices(record: Dict[str, Any]) -> Optional[Tuple[int, int, float, float]]:
        velocity = record.get("predator_velocity") or []
        if len(velocity) < 2:
            return None
        vx = float(velocity[0])
        vy = float(velocity[1])
        speed = math.sqrt(vx * vx + vy * vy)
        if speed <= 1e-9:
            return None
        angle = (math.degrees(math.atan2(vy, vx)) % 360.0)
        angle_bin = int(angle // angle_bin_deg) if angle_bin_deg > 0 else 0
        speed_bin_idx = int(speed // speed_bin) if speed_bin > 0 else 0
        return angle_bin, speed_bin_idx, angle, speed

    cache: List[Tuple[int, int, float, float, Dict[str, Any]]] = []
    for record in records:
        result = _bin_indices(record)
        if result is None:
            continue
        cache.append((*result, record))

    enriched_clusters: List[Dict[str, Any]] = []
    for cluster in selected:
        start_angle = float(cluster.get("angle_deg_start", 0.0))
        angle_idx = int(start_angle // angle_bin_deg) if angle_bin_deg > 0 else 0
        speed_start = float(cluster.get("speed_bin_start", 0.0))
        speed_idx = int(speed_start // speed_bin) if speed_bin > 0 else 0
        sample_payload: Optional[Dict[str, Any]] = None
        for cached_angle, cached_speed, angle_deg, speed_value, record in cache:
            if cached_angle == angle_idx and cached_speed == speed_idx:
                sample_payload = {
                    "iteration": int(record.get("iteration", -1)),
                    "angle_deg": angle_deg,
                    "speed": speed_value,
                    "predator_velocity": record.get("predator_velocity"),
                    "fish_idx": record.get("fish_idx"),
                }
                break
        enriched = dict(cluster)
        if sample_payload:
            enriched["sample"] = sample_payload
        enriched_clusters.append(enriched)

    return {
        "total_events": int(cluster_summary.get("total", 0)),
        "angle_bin_deg": float(angle_bin_deg),
        "speed_bin": float(speed_bin),
        "clusters": enriched_clusters,
    }


def summarize_step_one_by_stage(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    total = len(records)
    stage_buckets: Dict[str, Dict[str, Any]] = {}
    for record in records:
        stage_type = record.get("tail_stage_type") or record.get("stage_type") or "unknown"
        stage_key = str(stage_type).strip() or "unknown"
        bucket = stage_buckets.setdefault(stage_key, {"count": 0, "iterations": [], "speeds": []})
        bucket["count"] += 1
        iteration = record.get("iteration")
        if iteration is not None:
            try:
                bucket["iterations"].append(int(iteration))
            except (TypeError, ValueError):
                pass
        velocity = record.get("predator_velocity") or []
        if len(velocity) >= 2:
            try:
                vx = float(velocity[0])
                vy = float(velocity[1])
                speed = math.sqrt(vx * vx + vy * vy)
                bucket["speeds"].append(speed)
            except (TypeError, ValueError):
                continue
    summary = {
        "total_events": total,
        "stages": [],
    }
    for stage_key in sorted(stage_buckets.keys()):
        data = stage_buckets[stage_key]
        entry = {
            "stage_type": stage_key,
            "count": int(data["count"]),
            "share": float(data["count"] / total),
        }
        iterations = data.get("iterations") or []
        if iterations:
            entry["iteration_min"] = int(min(iterations))
            entry["iteration_max"] = int(max(iterations))
        speeds = data.get("speeds") or []
        if speeds:
            entry["predator_speed"] = _describe_scalar_series(speeds)
        summary["stages"].append(entry)
    return summary


def plot_step_one_stage_distribution(stage_summary: Dict[str, Any], run_name: str, plot_path: Path):
    stages = (stage_summary or {}).get("stages") or []
    if not stages:
        return
    sorted_stages = sorted(stages, key=lambda item: item.get("share", 0.0), reverse=True)
    labels = [entry.get("stage_type", "unknown") for entry in sorted_stages]
    shares = [float(entry.get("share", 0.0)) * 100.0 for entry in sorted_stages]
    counts = [int(entry.get("count", 0)) for entry in sorted_stages]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4 + 0.4 * len(labels)))
    bars = ax.barh(positions, shares, color="tab:cyan", alpha=0.8)
    ax.set_xlabel("Step-one share (%)")
    ax.set_xlim(0, max(100.0, max(shares) + 5))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"Step-one stage distribution - {run_name}")
    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{shares[idx]:.1f}% ({counts[idx]})", va="center")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def export_tail_diagnostics(
    records: Sequence[Dict[str, Any]],
    run_name: str,
    plot_path: Path,
    summary_path: Path,
    stage_summary_path: Optional[Path] = None,
    stage_plot_path: Optional[Path] = None,
    angle_bin_deg: float = 30.0,
    speed_bin: float = 0.4,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    plot_step_one_heading_polar(records, run_name, plot_path)
    summary = summarize_worst_step_one_clusters(records, angle_bin_deg=angle_bin_deg, speed_bin=speed_bin)
    stage_summary = summarize_step_one_by_stage(records)
    result = {
        "plot_path": str(plot_path),
        "summary_path": None,
        "stage_summary_path": None,
        "stage_plot_path": None,
    }
    if summary:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "run_name": run_name,
            "records": int(len(records)),
            "summary": summary,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        result["summary_path"] = str(summary_path)
    if stage_summary and stage_summary_path:
        stage_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "run_name": run_name,
            "records": int(len(records)),
            "stage_summary": stage_summary,
        }
        stage_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stage_summary_path, "w", encoding="utf-8") as f:
            json.dump(stage_payload, f, indent=2)
        result["stage_summary_path"] = str(stage_summary_path)
    if stage_summary and stage_plot_path:
        plot_step_one_stage_distribution(stage_summary, run_name, stage_plot_path)
        result["stage_plot_path"] = str(stage_plot_path)
    return result


def plot_multi_eval_history(history: Sequence[Dict[str, Any]], plot_path: Path):
    if not history:
        return
    iterations: List[int] = []
    avg_final: List[float] = []
    min_final: List[float] = []
    first_death_p10: List[Optional[float]] = []
    early_death_fraction_100: List[Optional[float]] = []
    for entry in history:
        summary = entry.get("summary") or {}
        if not summary:
            continue
        iterations.append(int(entry.get("iteration", 0)))
        avg_final.append(float(summary.get("avg_final_survival_rate", 0.0)))
        min_final.append(float(summary.get("min_final_survival_rate", 0.0)))
        death_stats = entry.get("death_stats") or {}
        first_death_p10.append(death_stats.get("p10"))
        early_death_fraction_100.append(death_stats.get("early_death_fraction_100"))
    if not iterations:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(iterations, avg_final, marker="o", label="avg_final_survival_rate")
    ax1.plot(iterations, min_final, marker="s", label="min_final_survival_rate")
    early_curve = [np.nan if value is None else value for value in early_death_fraction_100]
    ax1.plot(iterations, early_curve, marker="^", linestyle=":", label="early_death_frac_100")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Survival rate")
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    death_curve = [np.nan if value is None else value for value in first_death_p10]
    ax2.plot(iterations, death_curve, color="tab:red", linestyle="--", marker="^", label="first_death_p10")
    ax2.set_ylabel("First death p10 (steps)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Multi-fish eval timeline")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


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
        ax1.set_title(f"dev_v32 progression ({run_name})")
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


def parse_phase_value_map(spec: str, total_phases: int) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    if not spec:
        return mapping
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("phase value mapping 必须是 'phase:value' 形式")
        phase_str, value_str = chunk.split(":", maxsplit=1)
        try:
            phase_idx = int(phase_str)
        except ValueError as exc:
            raise ValueError(f"非法 phase 索引: '{phase_str}'") from exc
        if phase_idx < 1 or phase_idx > total_phases:
            raise ValueError(f"phase 索引 {phase_idx} 超出范围 1..{total_phases}")
        try:
            value = float(value_str)
        except ValueError as exc:
            raise ValueError(f"非法数值: '{value_str}'") from exc
        mapping[phase_idx] = value
    return mapping


def parse_penalty_caps(spec: str) -> List[Tuple[float, float]]:
    caps: List[Tuple[float, float]] = []
    if not spec:
        return caps
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("Penalty caps must be 'threshold:value' format")
        threshold_str, cap_str = chunk.split(":", maxsplit=1)
        try:
            threshold = float(threshold_str)
            cap_value = float(cap_str)
        except ValueError as exc:
            raise ValueError(f"Invalid penalty cap chunk '{chunk}'") from exc
        caps.append((threshold, cap_value))
    caps.sort(key=lambda item: item[0])
    return caps


def parse_stage_speed_floors(spec: str) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    if not spec:
        return mapping
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("adaptive_boost_stage_floor chunk 必须是 'stage:speed'")
        stage_str, value_str = chunk.split(":", maxsplit=1)
        try:
            stage = int(stage_str)
        except ValueError as exc:
            raise ValueError(f"无效 stage '{stage_str}'") from exc
        if stage < 0:
            raise ValueError("stage 索引需 >=0")
        try:
            value = float(value_str)
        except ValueError as exc:
            raise ValueError(f"无效 escape boost 值 '{value_str}'") from exc
        mapping[stage] = value
    return mapping


def parse_heading_bias(spec: str) -> List[Dict[str, float]]:
    bias: List[Dict[str, float]] = []
    if not spec:
        return bias
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("predator_heading_bias 项必须是 'start-end:weight'")
        range_part, weight_part = chunk.split(":", maxsplit=1)
        if "-" not in range_part:
            raise ValueError("predator_heading_bias 范围必须包含 '-'")
        start_str, end_str = range_part.split("-", maxsplit=1)
        try:
            start = float(start_str)
            end = float(end_str)
            weight = float(weight_part)
        except ValueError as exc:
            raise ValueError(f"非法 heading bias 定义 '{chunk}'") from exc
        if weight <= 0:
            continue
        start_norm = start % 360.0
        end_norm = end % 360.0
        if math.isclose(start_norm, end_norm, abs_tol=1e-6):
            continue
        if end_norm <= start_norm:
            bias.append({"start_deg": start_norm, "end_deg": 360.0, "weight": weight})
            bias.append({"start_deg": 0.0, "end_deg": end_norm, "weight": weight})
        else:
            bias.append({"start_deg": start_norm, "end_deg": end_norm, "weight": weight})
    return bias


def parse_speed_bias_bins(spec: str) -> List[Dict[str, float]]:
    if not spec:
        return []
    bins: List[Dict[str, float]] = []
    chunks = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    for chunk in chunks:
        if ":" not in chunk or "-" not in chunk:
            raise ValueError("predator_pre_roll_speed_bias 格式应为 'start-end:weight'")
        range_part, weight_part = chunk.split(":", maxsplit=1)
        start_str, end_str = range_part.split("-", maxsplit=1)
        try:
            start_val = float(start_str)
            end_val = float(end_str)
            weight_val = float(weight_part)
        except ValueError as exc:
            raise ValueError("predator_pre_roll_speed_bias 需要浮点数边界和权重") from exc
        if end_val <= start_val:
            raise ValueError("predator_pre_roll_speed_bias 区间必须满足 end>start")
        if weight_val <= 0:
            raise ValueError("predator_pre_roll_speed_bias 权重必须为正")
        bins.append({
            "min_speed": start_val,
            "max_speed": end_val,
            "weight": weight_val,
        })
    return bins


def load_tail_seed_velocities(path: Path, limit: int) -> List[Tuple[float, float]]:
    if limit <= 0 or not path:
        return []
    if not path.exists():
        print(f"[tail-prewarm] summary not found at {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[tail-prewarm] failed to load {path}: {exc}")
        return []
    base_vectors: List[Tuple[float, float]] = []
    custom_velocities = payload.get("velocities")
    if isinstance(custom_velocities, list):
        for entry in custom_velocities:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            try:
                vx = float(entry[0])
                vy = float(entry[1])
            except (TypeError, ValueError):
                continue
            base_vectors.append((vx, vy))
    clusters = (payload.get("summary") or {}).get("clusters") or []
    for cluster in clusters:
        sample = cluster.get("sample") or {}
        velocity = sample.get("predator_velocity")
        if not velocity or len(velocity) < 2:
            continue
        try:
            vx = float(velocity[0])
            vy = float(velocity[1])
        except (TypeError, ValueError):
            continue
        base_vectors.append((vx, vy))
    if not base_vectors:
        print(f"[tail-prewarm] no predator_velocity samples in {path}")
        return []
    repeated: List[Tuple[float, float]] = []
    while len(repeated) < limit:
        for vector in base_vectors:
            repeated.append(vector)
            if len(repeated) >= limit:
                break
    return repeated[:limit]


def parse_tail_seed_stage_spec(spec: str) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []
    if not spec:
        return stages
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("tail_seed_stage_spec 项必须是 'iterations:path'")
        iter_part, path_part = chunk.split(":", maxsplit=1)
        try:
            iterations = int(iter_part.strip())
        except ValueError as exc:
            raise ValueError(f"非法 tail_seed_stage_spec iterations '{iter_part}'") from exc
        if iterations <= 0:
            raise ValueError("tail_seed_stage_spec iterations 必须为正整数")
        meta: Dict[str, Any] = {}
        path_section = path_part.strip()
        if "|" in path_section:
            parts = [part.strip() for part in path_section.split("|") if part.strip()]
            if not parts:
                raise ValueError("tail_seed_stage_spec 中缺少合法路径")
            path_str = parts[0]
            for extra in parts[1:]:
                if "=" not in extra:
                    continue
                key, value = extra.split("=", maxsplit=1)
                meta[key.strip().lower()] = value.strip()
        else:
            path_str = path_section
        path = Path(path_str).expanduser()
        payload: Dict[str, Any] = {"iterations": iterations, "path": path}
        if meta:
            payload["metadata"] = meta
        stages.append(payload)
    return stages


def parse_penalty_gate_allowance(spec: str, total_phases: int) -> List[int]:
    if not spec:
        return []
    values: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            phase_idx = int(chunk)
        except ValueError as exc:
            raise ValueError("penalty_gate_phase_allowance must be integers") from exc
        zero_based = phase_idx - 1
        if zero_based < 0 or zero_based >= total_phases:
            raise ValueError(
                f"penalty gate phase index {phase_idx} is out of range for {total_phases} curriculum phases"
            )
        values.append(zero_based)
    values = sorted(set(values))
    return values


def apply_penalty_caps(
    base_speed: float,
    penalty_value: float,
    caps: Sequence[Tuple[float, float]],
) -> float:
    speed = float(base_speed)
    if penalty_value is None:
        return speed
    for threshold, cap in caps:
        if penalty_value >= threshold:
            speed = min(speed, cap)
    return speed


def parse_ent_coef_schedule(schedule: str) -> List[Tuple[int, Optional[int], float]]:
    windows: List[Tuple[int, Optional[int], float]] = []
    if not schedule:
        return windows
    for chunk in schedule.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("Each ent_coef schedule chunk must be 'start-end:value'")
        range_part, value_part = chunk.split(":", maxsplit=1)
        try:
            value = float(value_part)
        except ValueError as exc:
            raise ValueError(f"Invalid ent_coef value in chunk '{chunk}'") from exc
        range_part = range_part.strip()
        if not range_part:
            raise ValueError(f"Empty range in ent_coef schedule chunk '{chunk}'")
        if range_part.endswith("+"):
            start = int(range_part[:-1])
            end = None
        elif "-" in range_part:
            start_str, end_str = range_part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid ent_coef range '{range_part}' (end < start)")
        else:
            start = int(range_part)
            end = start
        if start <= 0:
            raise ValueError("ent_coef schedule iterations must be positive integers")
        windows.append((start, end, value))
    return windows


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
    predator_spawn_jitter_radius: float,
    predator_pre_roll_steps: int,
    predator_pre_roll_angle_jitter: float,
    predator_pre_roll_speed_jitter: float,
    predator_heading_bias: Optional[Sequence[Dict[str, float]]],
    predator_pre_roll_speed_bias: Optional[Sequence[Dict[str, float]]] = None,
    prewarm_predator_velocity_seqs: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    tail_force_reset_steps: int = 0,
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
        predator_spawn_jitter_radius,
        predator_pre_roll_steps,
        predator_pre_roll_angle_jitter,
        predator_pre_roll_speed_jitter,
        predator_heading_bias,
        predator_pre_roll_speed_bias,
        prewarm_predator_velocity_seqs,
        tail_force_reset_steps,
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
            "death_timesteps",
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
    predator_spawn_jitter_radius: float = 0.0,
    predator_pre_roll_steps: int = 0,
    predator_pre_roll_angle_jitter: float = 0.0,
    predator_pre_roll_speed_jitter: float = 0.0,
    predator_heading_bias: Optional[Sequence[Dict[str, float]]] = None,
    predator_pre_roll_speed_bias: Optional[Sequence[Dict[str, float]]] = None,
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
            predator_spawn_jitter_radius=predator_spawn_jitter_radius,
            predator_pre_roll_steps=predator_pre_roll_steps,
            predator_pre_roll_angle_jitter=predator_pre_roll_angle_jitter,
            predator_pre_roll_speed_jitter=predator_pre_roll_speed_jitter,
            predator_heading_bias=predator_heading_bias,
            predator_pre_roll_speed_bias=predator_pre_roll_speed_bias,
        )
        obs, info = env.reset()
        pre_roll_stats = (info or {}).get("pre_roll_stats") if info is not None else None
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
    predator_spawn_jitter_radius: float = 0.0,
    predator_pre_roll_steps: int = 0,
    predator_pre_roll_angle_jitter: float = 0.0,
    predator_pre_roll_speed_jitter: float = 0.0,
    predator_heading_bias: Optional[Sequence[Dict[str, float]]] = None,
    seeds: Optional[Sequence[int]] = None,
    predator_pre_roll_speed_bias: Optional[Sequence[Dict[str, float]]] = None,
):
    """Use a single-fish policy to control all fish in the base env."""

    results = []
    for ep in range(episodes):
        seed_value = None
        if seeds is not None and ep < len(seeds):
            seed_value = int(seeds[ep])
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
            predator_spawn_jitter_radius=predator_spawn_jitter_radius,
            predator_pre_roll_steps=predator_pre_roll_steps,
            predator_pre_roll_angle_jitter=predator_pre_roll_angle_jitter,
            predator_pre_roll_speed_jitter=predator_pre_roll_speed_jitter,
            predator_heading_bias=predator_heading_bias,
            predator_pre_roll_speed_bias=predator_pre_roll_speed_bias,
        )
        obs, info = env.reset(seed=seed_value)
        pre_roll_stats = (info or {}).get("pre_roll_stats") if info is not None else None
        done = False
        total_reward = 0.0
        steps = 0
        survival_trace = []
        final_alive = num_fish
        death_timesteps = None
        step_one_deaths = None
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
                step_one_deaths = info.get("step_one_deaths")
        env.close()
        results.append({
            "episode": int(ep),
            "avg_reward": float(total_reward / max(steps, 1)),
            "total_reward": float(total_reward),
            "steps": int(steps),
            "final_num_alive": final_alive,
            "min_survival_rate": float(min(survival_trace) if survival_trace else 0.0),
            "death_timesteps": death_timesteps,
            "step_one_deaths": step_one_deaths,
            "pre_roll_stats": pre_roll_stats,
            "seed": seed_value,
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
    predator_spawn_jitter_radius: float = 0.0,
    predator_pre_roll_steps: int = 0,
    predator_pre_roll_angle_jitter: float = 0.0,
    predator_pre_roll_speed_jitter: float = 0.0,
    predator_heading_bias: Optional[Sequence[Dict[str, float]]] = None,
    seed: Optional[int] = None,
    predator_pre_roll_speed_bias: Optional[Sequence[Dict[str, float]]] = None,
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
        predator_spawn_jitter_radius=predator_spawn_jitter_radius,
        predator_pre_roll_steps=predator_pre_roll_steps,
        predator_pre_roll_angle_jitter=predator_pre_roll_angle_jitter,
        predator_pre_roll_speed_jitter=predator_pre_roll_speed_jitter,
        predator_heading_bias=predator_heading_bias,
        predator_pre_roll_speed_bias=predator_pre_roll_speed_bias,
    )
    obs, _ = env.reset(seed=seed)
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


def _coerce_xy(value: Any) -> List[float]:
    try:
        iterable = list(value)
    except TypeError:
        return [0.0, 0.0]
    if not iterable:
        return [0.0, 0.0]
    x = float(iterable[0]) if len(iterable) > 0 else 0.0
    y = float(iterable[1]) if len(iterable) > 1 else 0.0
    return [x, y]


def extract_step_one_deaths(records: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    events: List[Dict[str, Any]] = []
    for entry in records:
        raw_events = entry.get("step_one_deaths")
        if not raw_events:
            continue
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            try:
                timestep = int(raw.get("timestep", 0))
            except (TypeError, ValueError):
                continue
            if timestep != 1:
                continue
            record = {
                "fish_idx": int(raw.get("fish_idx", -1)),
                "predator_index": int(raw.get("predator_index", 0)),
                "fish_position": _coerce_xy(raw.get("fish_position")),
                "predator_position": _coerce_xy(raw.get("predator_position")),
                "predator_velocity": _coerce_xy(raw.get("predator_velocity")),
            }
            events.append(record)
    return events


def select_best_multi_eval_entry(
    history: Optional[Sequence[Dict[str, Any]]],
    *,
    max_step_one: Optional[float] = None,
    max_early_death: Optional[float] = None,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not history:
        return None, None
    best_entry: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float]] = None
    for entry in history:
        summary = entry.get("summary") or {}
        death_stats = entry.get("death_stats") or {}
        step_one = entry.get("step_one_death_count")
        try:
            step_one_val = float(step_one)
        except (TypeError, ValueError):
            step_one_val = float("inf")
        early = death_stats.get("early_death_fraction_100")
        try:
            early_val = float(early)
        except (TypeError, ValueError):
            early_val = float("inf")
        if max_step_one is not None and step_one_val > max_step_one:
            continue
        if max_early_death is not None and early_val > max_early_death:
            continue
        avg_final = summary.get("avg_final_survival_rate")
        try:
            avg_final_val = float(avg_final)
        except (TypeError, ValueError):
            avg_final_val = 0.0
        score = (step_one_val, early_val, -avg_final_val)
        if best_key is None or score < best_key:
            best_key = score
            best_entry = entry
    iteration = None
    if best_entry is not None:
        iteration = best_entry.get("iteration")
        if iteration is not None:
            iteration = int(iteration)
    return iteration, best_entry


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


def summarize_death_timesteps(death_timesteps: List[int], max_steps: int) -> Optional[Dict[str, float]]:
    if not death_timesteps:
        return None
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
        "early_death_fraction_100": float(np.mean(arr < 100)),
        "early_death_fraction_150": float(np.mean(arr < 150)),
    }
    return summary


def save_death_stats(death_timesteps: List[int], output_path: Path, max_steps: int):
    summary = summarize_death_timesteps(death_timesteps, max_steps)
    if summary is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def summarize_multi_eval(
    eval_multi_results: Optional[Sequence[Dict]],
    num_fish: int,
) -> Optional[Dict[str, float]]:
    if not eval_multi_results:
        return None
    final_alive = np.array([max(res.get("final_num_alive", 0), 0) for res in eval_multi_results], dtype=float)
    min_survival_rates = np.array([max(res.get("min_survival_rate", 0.0), 0.0) for res in eval_multi_results], dtype=float)
    final_rates = final_alive / max(num_fish, 1)
    summary = {
        "episodes": int(len(eval_multi_results)),
        "avg_final_survival_rate": float(np.mean(final_rates)),
        "min_final_survival_rate": float(np.min(final_rates)),
        "avg_min_survival_rate": float(np.mean(min_survival_rates)),
        "min_of_min_survival_rate": float(np.min(min_survival_rates)),
        "median_min_survival_rate": float(np.median(min_survival_rates)),
    }
    return summary


def train(args):
    if args.num_envs < 64:
        raise ValueError("SOP requires num_envs >= 64; please increase --num_envs.")

    success_early_death_overrides = parse_stage_threshold_overrides(
        args.penalty_gate_success_early_death_by_stage
    )
    success_window_overrides = parse_stage_int_overrides(
        args.penalty_gate_success_early_death_window_by_stage,
        "penalty_gate_success_early_death_window_by_stage",
        min_value=1,
    )
    success_p10_overrides = parse_stage_threshold_overrides(
        args.penalty_gate_success_first_death_p10_by_stage
    )
    success_p10_window_overrides = parse_stage_int_overrides(
        args.penalty_gate_success_p10_median_window_by_stage,
        "penalty_gate_success_p10_median_window_by_stage",
        min_value=1,
    )
    failure_tolerance_overrides = parse_stage_int_overrides(
        args.penalty_gate_failure_tolerance_by_stage,
        "penalty_gate_failure_tolerance_by_stage",
        min_value=1,
    )

    parsed_heading_bias = parse_heading_bias(args.predator_heading_bias)
    predator_heading_bias = parsed_heading_bias if parsed_heading_bias else None
    parsed_speed_bias = parse_speed_bias_bins(args.predator_pre_roll_speed_bias)
    predator_speed_bias = parsed_speed_bias if parsed_speed_bias else None

    tail_seed_sequences: Optional[List[List[Tuple[float, float]]]] = None
    tail_stage_tracker: Optional[TailStageTracker] = None
    tail_stage_metadata: List[Dict[str, Any]] = []
    tail_stage_spec = parse_tail_seed_stage_spec(args.tail_seed_stage_spec)
    if tail_stage_spec:
        velocities: List[Tuple[float, float]] = []
        for stage_idx, stage_info in enumerate(tail_stage_spec, start=1):
            stage_iterations = int(stage_info.get("iterations", 0) or 0)
            stage_path_raw = stage_info.get("path")
            if stage_iterations <= 0 or stage_path_raw is None:
                print(f"[tail-prewarm] skip stage {stage_idx} due to missing iterations/path")
                continue
            stage_path = stage_path_raw
            if not stage_path.is_absolute():
                stage_path = (ROOT_DIR / stage_path).resolve()
            stage_limit = stage_iterations * args.num_envs
            stage_velocities = load_tail_seed_velocities(stage_path, stage_limit)
            metadata = dict(stage_info.get("metadata") or {})
            stage_label_override = metadata.get("label")
            stage_type = metadata.get("stage_type") or metadata.get("type")
            stage_label = stage_label_override or f"stage{stage_idx}:{stage_path.stem}"
            if stage_velocities:
                print(
                    f"[tail-prewarm] loaded {len(stage_velocities)} samples from stage {stage_idx} ({stage_path}) "
                    f"covering {stage_iterations} iterations"
                )
                velocities.extend(stage_velocities)
                tail_stage_metadata.append(
                    {
                        "label": stage_label,
                        "capacity": len(stage_velocities),
                        "stage_type": stage_type,
                    }
                )
            else:
                print(f"[tail-prewarm] stage {stage_idx} at {stage_path} yielded no samples")
        if velocities:
            sequences = [[] for _ in range(args.num_envs)]
            for idx, velocity in enumerate(velocities):
                sequences[idx % args.num_envs].append(velocity)
            tail_seed_sequences = sequences
    elif args.tail_seed_replay_path and args.tail_seed_prewarm_iterations > 0:
        tail_seed_path = Path(args.tail_seed_replay_path).expanduser()
        if not tail_seed_path.is_absolute():
            tail_seed_path = (ROOT_DIR / tail_seed_path).resolve()
        total_samples = args.tail_seed_prewarm_iterations * args.num_envs
        velocities = load_tail_seed_velocities(tail_seed_path, total_samples)
        if velocities:
            sequences: List[List[Tuple[float, float]]] = [[] for _ in range(args.num_envs)]
            for idx, velocity in enumerate(velocities):
                sequences[idx % args.num_envs].append(velocity)
            tail_seed_sequences = sequences
            tail_stage_metadata.append(
                {
                    "label": tail_seed_path.stem,
                    "capacity": len(velocities),
                    "stage_type": None,
                }
            )

    if not tail_stage_metadata and tail_seed_sequences:
        total_capacity = sum(len(seq) for seq in tail_seed_sequences)
        if total_capacity > 0:
            tail_stage_metadata.append({"label": "prewarm", "capacity": total_capacity, "stage_type": None})
    if tail_stage_metadata:
        tail_stage_tracker = TailStageTracker(tail_stage_metadata)
    tail_seed_cycler: Optional[TailSeedCycler] = None
    if tail_seed_sequences:
        cycler = TailSeedCycler(tail_seed_sequences)
        if cycler.has_data():
            tail_seed_cycler = cycler

    stage_speed_floors = parse_stage_speed_floors(args.adaptive_boost_stage_floor)

    curriculum = parse_curriculum(args.curriculum, args.total_iterations, args.num_fish)
    phase_iterations = [phase_iter for _, phase_iter in curriculum]
    escape_boost_phase_speeds = parse_phase_value_map(
        args.escape_boost_phase_speeds,
        len(curriculum)
    )
    escape_boost_phase_floors = parse_phase_value_map(
        args.escape_boost_phase_floors,
        len(curriculum)
    )
    escape_boost_penalty_caps = parse_penalty_caps(args.escape_boost_penalty_caps)

    penalty_scheduler: Optional[DensityPenaltyScheduler] = None
    penalty_gate_phase_allowance: List[int] = parse_penalty_gate_allowance(
        args.penalty_gate_phase_allowance,
        len(curriculum),
    )
    phase_plateaus: Optional[List[int]] = None
    if args.density_penalty_phase_plateaus:
        try:
            plateau_values = [int(chunk.strip()) for chunk in args.density_penalty_phase_plateaus.split(",") if chunk.strip()]
        except ValueError as exc:
            raise ValueError("density_penalty_phase_plateaus must be comma-separated integers") from exc
        if len(plateau_values) != len(curriculum):
            raise ValueError("density_penalty_phase_plateaus length must match curriculum phases")
        phase_plateaus = plateau_values
    penalty_gate: Optional[PenaltyStageGate] = None
    if args.density_penalty_phase_targets:
        try:
            phase_targets = [float(chunk.strip()) for chunk in args.density_penalty_phase_targets.split(",") if chunk.strip()]
        except ValueError as exc:
            raise ValueError("density_penalty_phase_targets must be comma-separated floats") from exc
        if len(phase_targets) != len(curriculum):
            raise ValueError("density_penalty_phase_targets length must match curriculum phases")
        ramp_phases: Set[int] = set()
        if args.density_penalty_ramp_phases:
            try:
                ramp_phases = {
                    int(chunk.strip())
                    for chunk in args.density_penalty_ramp_phases.split(",")
                    if chunk.strip()
                }
            except ValueError as exc:
                raise ValueError("density_penalty_ramp_phases must be comma-separated integers") from exc
            invalid = [idx for idx in ramp_phases if idx < 1 or idx > len(curriculum)]
            if invalid:
                raise ValueError(f"density_penalty_ramp_phases contains invalid indices: {invalid}")
        penalty_scheduler = DensityPenaltyScheduler(
            base_value=args.density_penalty_coef,
            phase_targets=phase_targets,
            phase_iterations=phase_iterations,
            ramp_phases=ramp_phases,
            phase_plateaus=phase_plateaus,
            lock_value=(args.density_penalty_lock_value if args.density_penalty_lock_until > 0 else None),
            lock_until_iteration=(args.density_penalty_lock_until if args.density_penalty_lock_until > 0 else None),
        )
        if penalty_gate_phase_allowance and args.penalty_gate_required_successes > 0:
            lock_iter = args.penalty_gate_lock_iteration if args.penalty_gate_lock_iteration > 0 else args.density_penalty_lock_until
            penalty_gate = PenaltyStageGate(
                phase_allowance=penalty_gate_phase_allowance,
                lock_until=lock_iter,
                required_successes=args.penalty_gate_required_successes,
                freeze_iterations=args.penalty_gate_freeze_iterations,
                success_step_one=args.penalty_gate_success_step_one,
                success_step_one_ratio=(
                    args.penalty_gate_success_step_one_ratio if args.penalty_gate_success_step_one_ratio > 0 else None
                ),
                success_early_death=args.penalty_gate_success_early_death,
                success_early_death_window=args.penalty_gate_success_early_death_window,
                success_early_death_by_stage=success_early_death_overrides,
                success_early_death_window_by_stage=success_window_overrides or None,
                success_first_death_p10=(
                    args.penalty_gate_success_p10 if args.penalty_gate_success_p10 > 0 else None
                ),
                success_first_death_p10_by_stage=success_p10_overrides or None,
                success_freeze_iterations=args.penalty_gate_success_freeze_iterations,
                failure_p10=args.penalty_gate_failure_p10,
                failure_min_final=args.penalty_gate_failure_min_final,
                success_p10_median_window=args.penalty_gate_success_median_window,
                success_p10_median_window_by_stage=success_p10_window_overrides or None,
                failure_tolerance=args.penalty_gate_failure_tolerance,
                failure_tolerance_by_stage=failure_tolerance_overrides or None,
            )
            penalty_scheduler.set_max_active_phase(penalty_gate.current_phase_limit())

    initial_density_penalty = penalty_scheduler.initial_value() if penalty_scheduler else args.density_penalty_coef
    entropy_scheduler: Optional[EntropyCoefScheduler] = None
    if args.entropy_coef_schedule:
        schedule_windows = parse_ent_coef_schedule(args.entropy_coef_schedule)
        if schedule_windows:
            entropy_scheduler = EntropyCoefScheduler(args.ent_coef, schedule_windows)
    initial_entropy_coef = args.ent_coef
    initial_phase_floor = escape_boost_phase_floors.get(1, args.escape_boost_floor_default)
    initial_escape_boost_speed = max(
        escape_boost_phase_speeds.get(1, args.escape_boost_speed),
        initial_phase_floor,
    )
    initial_effective_boost = apply_penalty_caps(
        initial_escape_boost_speed,
        initial_density_penalty,
        escape_boost_penalty_caps,
    )

    run_name = args.run_name or datetime.now().strftime("dev_v35_%Y%m%d_%H%M%S")
    run_checkpoint_dir = CHECKPOINT_DIR / run_name
    run_log_path = LOG_DIR / f"{run_name}.log"
    stats_path = run_checkpoint_dir / "training_stats.pkl"
    tb_run_dir = TB_LOG_DIR / run_name
    tb_post_eval_dir = tb_run_dir / "post_eval"
    tb_post_eval_dir.mkdir(parents=True, exist_ok=True)
    post_eval_writer = SummaryWriter(log_dir=str(tb_post_eval_dir))

    multi_eval_kwargs: Optional[Dict[str, Any]] = None
    multi_eval_history_path: Optional[Path] = None
    multi_eval_plot_path: Optional[Path] = None
    multi_eval_hist_dir: Optional[Path] = None
    multi_eval_tb_writer: Optional[SummaryWriter] = None
    if args.multi_eval_interval > 0 and args.eval_multi_fish and args.eval_multi_fish > 1:
        multi_eval_kwargs = {
            "num_fish": args.eval_multi_fish,
            "episodes": args.multi_eval_probe_episodes,
            "include_neighbor_features": args.include_neighbor_features,
            "neighbor_radius": args.neighbor_radius,
            "neighbor_average_count": args.neighbor_average_count,
            "initial_escape_boost": args.initial_escape_boost,
            "escape_boost_speed": initial_effective_boost,
            "escape_jitter_std": args.escape_jitter_std,
            "divergence_reward_coef": args.divergence_reward_coef,
            "density_penalty_coef": initial_density_penalty,
            "density_target": args.density_target,
            "predator_spawn_jitter_radius": args.predator_spawn_jitter_radius,
            "predator_pre_roll_steps": args.predator_pre_roll_steps,
            "predator_pre_roll_angle_jitter": args.predator_pre_roll_angle_jitter,
            "predator_pre_roll_speed_jitter": args.predator_pre_roll_speed_jitter,
            "predator_heading_bias": predator_heading_bias,
            "predator_pre_roll_speed_bias": predator_speed_bias,
        }
        multi_eval_history_path = run_checkpoint_dir / "eval_multi_history.jsonl"
        multi_eval_plot_path = PLOTS_DIR / f"{run_name}_multi_eval_timeline.png"
        multi_eval_hist_dir = PLOTS_DIR / f"{run_name}_multi_eval_hist"
        probe_tb_dir = tb_run_dir / "multi_eval_probes"
        probe_tb_dir.mkdir(parents=True, exist_ok=True)
        multi_eval_tb_writer = SummaryWriter(log_dir=str(probe_tb_dir))

    penalty_stage_log_path = run_checkpoint_dir / "penalty_stage_debug.jsonl"
    pre_roll_stats_path = run_checkpoint_dir / "pre_roll_stats.jsonl"
    step_one_cluster_path = run_checkpoint_dir / "step_one_clusters.jsonl"

    lr_schedule = make_lr_schedule(args.learning_rate, args.lr_schedule)

    adaptive_boost_controller: Optional[AdaptiveBoostController] = None
    if not args.disable_adaptive_boost:
        adaptive_boost_controller = AdaptiveBoostController(
            window=args.adaptive_boost_window,
            lower_p10=args.adaptive_boost_lower_p10,
            upper_p10=args.adaptive_boost_upper_p10,
            low_speed=args.adaptive_boost_low_speed,
            high_speed=args.adaptive_boost_high_speed,
            min_iteration=args.adaptive_boost_min_iteration,
            use_median=args.adaptive_boost_use_median,
            stage_speed_floors=stage_speed_floors,
        )

    best_checkpoint_gate: Dict[str, float] = {}
    if args.best_checkpoint_max_step_one > 0:
        best_checkpoint_gate["max_step_one"] = float(args.best_checkpoint_max_step_one)
    if args.best_checkpoint_max_early_death > 0:
        best_checkpoint_gate["max_early_death"] = float(args.best_checkpoint_max_early_death)

    tail_seed_payloads = tail_seed_cycler.payloads() if tail_seed_cycler else tail_seed_sequences
    vec_env = build_vectorized_env(
        args.num_envs,
        curriculum[0][0],
        args.seed,
        args.obs_sampling,
        args.include_neighbor_features,
        args.neighbor_radius,
        args.neighbor_average_count,
        args.initial_escape_boost,
        initial_effective_boost,
        args.escape_jitter_std,
        args.divergence_reward_coef,
        initial_density_penalty,
        args.density_target,
        args.predator_spawn_jitter_radius,
        args.predator_pre_roll_steps,
        args.predator_pre_roll_angle_jitter,
        args.predator_pre_roll_speed_jitter,
        predator_heading_bias,
        predator_speed_bias,
        tail_seed_payloads,
        args.tail_force_reset_steps,
    )
    if penalty_scheduler:
        vec_env.env_method("set_density_penalty", initial_density_penalty, args.density_target)
    vec_env.env_method("set_escape_boost_speed", initial_effective_boost)

    training_signals = TrainingSignals(initial_escape_boost_speed)
    training_signals.phase_base_escape_boost_speed = initial_escape_boost_speed
    training_signals.phase_floor_escape_boost_speed = initial_phase_floor

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
        run_name=run_name,
        stats_path=stats_path,
        log_path=run_log_path,
        checkpoint_dir=run_checkpoint_dir,
        save_every=args.checkpoint_interval,
        density_penalty_scheduler=penalty_scheduler,
        density_target=args.density_target,
        initial_density_penalty=initial_density_penalty,
        entropy_scheduler=entropy_scheduler,
        initial_entropy_coef=initial_entropy_coef,
        signals=training_signals,
        initial_escape_boost_speed=initial_escape_boost_speed,
        schedule_trace_path=run_checkpoint_dir / "schedule_trace.json",
        escape_boost_penalty_caps=escape_boost_penalty_caps,
        multi_eval_every=args.multi_eval_interval,
        multi_eval_kwargs=multi_eval_kwargs,
        multi_eval_history_path=multi_eval_history_path,
        multi_eval_plot_path=multi_eval_plot_path,
        multi_eval_hist_dir=multi_eval_hist_dir,
        multi_eval_tb_writer=multi_eval_tb_writer,
        adaptive_boost_controller=adaptive_boost_controller,
        best_checkpoint_gate=(best_checkpoint_gate or None),
        penalty_gate=penalty_gate,
        penalty_stage_log_path=penalty_stage_log_path,
        pre_roll_stats_path=pre_roll_stats_path,
        step_one_cluster_path=step_one_cluster_path,
        tail_replay_count=args.tail_replay_count,
        tail_replay_video_fps=args.video_fps,
        tail_replay_max_steps=args.video_max_steps,
        media_dir=MEDIA_DIR,
        multi_eval_seed_base=(args.multi_eval_seed_base if args.multi_eval_seed_base is not None else args.seed),
        tail_stage_tracker=tail_stage_tracker,
        tail_seed_cycler=tail_seed_cycler,
        tail_stage_warn_ratio=args.tail_stage_warn_ratio,
        tail_stage_warn_patience=args.tail_stage_warn_patience,
    )

    current_density_penalty = initial_density_penalty

    for idx, (phase_num_fish, phase_iterations) in enumerate(curriculum):
        phase_idx = idx + 1
        phase_escape_boost_speed = escape_boost_phase_speeds.get(phase_idx, args.escape_boost_speed)
        phase_floor_speed = escape_boost_phase_floors.get(phase_idx, args.escape_boost_floor_default)
        phase_escape_boost_speed = max(phase_escape_boost_speed, phase_floor_speed)
        training_signals.phase_base_escape_boost_speed = phase_escape_boost_speed
        training_signals.phase_floor_escape_boost_speed = phase_floor_speed
        training_signals.escape_boost_speed = phase_escape_boost_speed
        phase_effective_boost = apply_penalty_caps(
            phase_escape_boost_speed,
            current_density_penalty,
            escape_boost_penalty_caps,
        )
        if idx > 0:
            vec_env.close()
            current_density_penalty = initial_density_penalty
            if penalty_scheduler and penalty_scheduler.current_value is not None:
                current_density_penalty = penalty_scheduler.current_value
            phase_effective_boost = apply_penalty_caps(
                phase_escape_boost_speed,
                current_density_penalty,
                escape_boost_penalty_caps,
            )
            tail_seed_payloads = None
            if tail_seed_cycler:
                tail_seed_payloads = tail_seed_cycler.payloads()
            vec_env = build_vectorized_env(
                args.num_envs,
                phase_num_fish,
                args.seed,
                args.obs_sampling,
                args.include_neighbor_features,
                args.neighbor_radius,
                args.neighbor_average_count,
                args.initial_escape_boost,
                phase_effective_boost,
                args.escape_jitter_std,
                args.divergence_reward_coef,
                current_density_penalty,
                args.density_target,
                args.predator_spawn_jitter_radius,
                args.predator_pre_roll_steps,
                args.predator_pre_roll_angle_jitter,
                args.predator_pre_roll_speed_jitter,
                predator_heading_bias,
                predator_speed_bias,
                tail_seed_payloads,
                args.tail_force_reset_steps,
            )
            if penalty_scheduler:
                vec_env.env_method("set_density_penalty", current_density_penalty, args.density_target)
            model.set_env(vec_env)
        vec_env.env_method("set_escape_boost_speed", phase_effective_boost)
        phase_timesteps = phase_iterations * args.n_steps * args.num_envs
        print(
            f"[dev_v32] Phase {idx + 1}/{len(curriculum)}: num_fish={phase_num_fish}, iterations={phase_iterations},"
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
    final_escape_boost_speed = callback.current_escape_boost_speed

    plot_path = PLOTS_DIR / f"{run_name}_survival.png"
    gif_path = MEDIA_DIR / f"{run_name}_curve.gif"
    plot_stats(callback.stats, run_name, plot_path)
    first_death_plot_path = PLOTS_DIR / f"{run_name}_first_death.png"
    plot_first_death_curve(callback.stats, run_name, first_death_plot_path)
    penalty_alignment_plot_path = PLOTS_DIR / f"{run_name}_penalty_vs_first_death.png"
    plot_penalty_alignment(callback.stats, run_name, penalty_alignment_plot_path)
    penalty_entropy_plot_path = PLOTS_DIR / f"{run_name}_penalty_vs_entropy.png"
    plot_penalty_entropy(callback.stats, run_name, penalty_entropy_plot_path)
    tail_warning_plot_path = PLOTS_DIR / f"{run_name}_tail_stage_warning.png"
    plot_tail_stage_warnings(callback.stats, run_name, tail_warning_plot_path, threshold=args.tail_stage_warn_ratio)
    stats_gif(callback.stats, run_name, gif_path)

    tail_polar_plot_path = PLOTS_DIR / f"{run_name}_step_one_heading_polar.png"
    tail_summary_path = run_checkpoint_dir / "step_one_worst_seeds.json"
    stage_summary_path = run_checkpoint_dir / "step_one_stage_summary.json"
    stage_plot_path = PLOTS_DIR / f"{run_name}_step_one_stage_distribution.png"
    export_tail_diagnostics(
        callback.step_one_records_all,
        run_name,
        tail_polar_plot_path,
        tail_summary_path,
        stage_summary_path=stage_summary_path,
        stage_plot_path=stage_plot_path,
    )

    eval_density_penalty = penalty_scheduler.final_value if penalty_scheduler else args.density_penalty_coef

    eval_results_single = None
    eval_path_single = None
    eval_multi = None
    eval_path_multi = None
    eval_multi_summary = None
    eval_multi_summary_path = None
    death_plot_path = None
    death_stats_path = None
    death_stats_summary = None
    step_one_death_path = None
    best_checkpoint_iteration: Optional[int] = None
    best_checkpoint_entry_path: Optional[Path] = None
    best_eval_path: Optional[Path] = None
    best_eval_summary: Optional[Dict[str, Any]] = None
    best_eval_summary_path: Optional[Path] = None
    best_video_best_path: Optional[Path] = None
    history_updated = False
    video_path = None

    if not args.skip_final_eval:
        eval_results_single = evaluate_single_fish(
            model,
            num_fish=curriculum[-1][0],
            episodes=args.eval_episodes,
            sampling_mode=args.obs_sampling,
            include_neighbor_features=args.include_neighbor_features,
            neighbor_radius=args.neighbor_radius,
            neighbor_average_count=args.neighbor_average_count,
            initial_escape_boost=args.initial_escape_boost,
            escape_boost_speed=final_escape_boost_speed,
            escape_jitter_std=args.escape_jitter_std,
            divergence_reward_coef=args.divergence_reward_coef,
            density_penalty_coef=eval_density_penalty,
            density_target=args.density_target,
            predator_spawn_jitter_radius=args.predator_spawn_jitter_radius,
            predator_pre_roll_steps=args.predator_pre_roll_steps,
            predator_pre_roll_angle_jitter=args.predator_pre_roll_angle_jitter,
            predator_pre_roll_speed_jitter=args.predator_pre_roll_speed_jitter,
            predator_heading_bias=predator_heading_bias,
            predator_pre_roll_speed_bias=predator_speed_bias,
        )
        eval_path_single = run_checkpoint_dir / "eval_single_fish.json"
        with open(eval_path_single, "w", encoding="utf-8") as f:
            json.dump({"results": eval_results_single}, f, indent=2)

        if args.eval_multi_fish and args.eval_multi_fish > 1:
            eval_multi = evaluate_multi_fish(
                model,
                num_fish=args.eval_multi_fish,
                episodes=args.eval_multi_episodes,
                include_neighbor_features=args.include_neighbor_features,
                neighbor_radius=args.neighbor_radius,
                neighbor_average_count=args.neighbor_average_count,
                initial_escape_boost=args.initial_escape_boost,
                escape_boost_speed=final_escape_boost_speed,
                escape_jitter_std=args.escape_jitter_std,
                divergence_reward_coef=args.divergence_reward_coef,
                density_penalty_coef=eval_density_penalty,
                density_target=args.density_target,
                predator_spawn_jitter_radius=args.predator_spawn_jitter_radius,
                predator_pre_roll_steps=args.predator_pre_roll_steps,
                predator_pre_roll_angle_jitter=args.predator_pre_roll_angle_jitter,
                predator_pre_roll_speed_jitter=args.predator_pre_roll_speed_jitter,
                predator_heading_bias=predator_heading_bias,
                predator_pre_roll_speed_bias=predator_speed_bias,
            )
            eval_path_multi = run_checkpoint_dir / "eval_multi_fish.json"
            with open(eval_path_multi, "w", encoding="utf-8") as f:
                json.dump({
                    "num_fish": args.eval_multi_fish,
                    "results": eval_multi
                }, f, indent=2)
            eval_multi_summary = summarize_multi_eval(eval_multi, args.eval_multi_fish)
            if eval_multi_summary:
                eval_multi_summary_path = run_checkpoint_dir / "eval_multi_summary.json"
                with open(eval_multi_summary_path, "w", encoding="utf-8") as f:
                    json.dump(eval_multi_summary, f, indent=2)

        if eval_multi:
            death_timesteps = collect_death_timesteps(eval_multi, MAX_STEPS)
            death_plot_path = PLOTS_DIR / f"{run_name}_death_histogram.png"
            death_stats_path = run_checkpoint_dir / "death_stats.json"
            plot_death_histogram(death_timesteps, run_name, death_plot_path, MAX_STEPS)
            death_stats_summary = save_death_stats(death_timesteps, death_stats_path, MAX_STEPS)
            step_one_records = extract_step_one_deaths(eval_multi)
            if step_one_records:
                step_one_death_path = run_checkpoint_dir / "eval_step_one_deaths.json"
                with open(step_one_death_path, "w", encoding="utf-8") as f:
                    json.dump(step_one_records, f, indent=2)
                for record in step_one_records:
                    enriched = dict(record)
                    enriched["iteration"] = int(args.total_iterations) + 1
                    callback.step_one_records_all.append(enriched)

        if args.eval_multi_fish and args.eval_multi_fish > 1 and callback.multi_eval_history:
            max_step_one = args.best_checkpoint_max_step_one if args.best_checkpoint_max_step_one > 0 else None
            max_early_death = args.best_checkpoint_max_early_death if args.best_checkpoint_max_early_death > 0 else None
            candidate_iter, candidate_entry = select_best_multi_eval_entry(
                callback.multi_eval_history,
                max_step_one=max_step_one,
                max_early_death=max_early_death,
            )
            if candidate_iter is None or candidate_entry is None:
                gating_desc = []
                if max_step_one is not None:
                    gating_desc.append(f"step_one_death_count<={max_step_one}")
                if max_early_death is not None:
                    gating_desc.append(f"early_death_fraction_100<={max_early_death:.3f}")
                gate_msg = " & ".join(gating_desc) if gating_desc else "no gating"
                callback._log(f"best_checkpoint gating skipped (no entries satisfied {gate_msg}).")
            else:
                checkpoint_zip = run_checkpoint_dir / f"model_iter_{candidate_iter}.zip"
                entry_best_path = candidate_entry.get("best_model_path")
                if entry_best_path:
                    custom_zip = Path(entry_best_path)
                    if custom_zip.exists():
                        checkpoint_zip = custom_zip
                if checkpoint_zip.exists():
                    candidate_entry.setdefault("best_model_path", str(checkpoint_zip))
                    best_checkpoint_iteration = candidate_iter
                    best_checkpoint_entry_path = run_checkpoint_dir / f"multi_eval_best_iter_{candidate_iter}.json"
                    with open(best_checkpoint_entry_path, "w", encoding="utf-8") as f:
                        json.dump(candidate_entry, f, indent=2)
                    best_model = PPO.load(checkpoint_zip, env=model.get_env())
                    best_eval_path = run_checkpoint_dir / f"eval_multi_best_iter_{candidate_iter}.json"
                    best_eval_results = evaluate_multi_fish(
                        best_model,
                        num_fish=args.eval_multi_fish,
                        episodes=args.best_checkpoint_eval_episodes,
                        include_neighbor_features=args.include_neighbor_features,
                        neighbor_radius=args.neighbor_radius,
                        neighbor_average_count=args.neighbor_average_count,
                        initial_escape_boost=args.initial_escape_boost,
                        escape_boost_speed=callback.current_escape_boost_speed,
                        escape_jitter_std=args.escape_jitter_std,
                        divergence_reward_coef=args.divergence_reward_coef,
                        density_penalty_coef=eval_density_penalty,
                        density_target=args.density_target,
                        predator_spawn_jitter_radius=args.predator_spawn_jitter_radius,
                        predator_pre_roll_steps=args.predator_pre_roll_steps,
                        predator_pre_roll_angle_jitter=args.predator_pre_roll_angle_jitter,
                        predator_pre_roll_speed_jitter=args.predator_pre_roll_speed_jitter,
                        predator_heading_bias=predator_heading_bias,
                        predator_pre_roll_speed_bias=predator_speed_bias,
                    )
                    with open(best_eval_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "iteration": candidate_iter,
                            "results": best_eval_results,
                        }, f, indent=2)
                    best_eval_summary = summarize_multi_eval(best_eval_results, args.eval_multi_fish)
                    if best_eval_summary:
                        best_eval_summary_path = run_checkpoint_dir / f"eval_multi_best_iter_{candidate_iter}_summary.json"
                        with open(best_eval_summary_path, "w", encoding="utf-8") as f:
                            json.dump(best_eval_summary, f, indent=2)
                    best_video_target = MEDIA_DIR / f"{run_name}_best_iter_{candidate_iter:02d}_multi.mp4"
                    best_video_best_path = record_multi_fish_video(
                        best_model,
                        num_fish=(
                            args.video_num_fish
                            if args.video_num_fish and args.video_num_fish > 0
                            else args.eval_multi_fish
                        ),
                        video_path=best_video_target,
                        max_steps=args.video_max_steps,
                        fps=args.video_fps,
                        include_neighbor_features=args.include_neighbor_features,
                        neighbor_radius=args.neighbor_radius,
                        neighbor_average_count=args.neighbor_average_count,
                        initial_escape_boost=args.initial_escape_boost,
                        escape_boost_speed=callback.current_escape_boost_speed,
                        escape_jitter_std=args.escape_jitter_std,
                        divergence_reward_coef=args.divergence_reward_coef,
                        density_penalty_coef=eval_density_penalty,
                        density_target=args.density_target,
                        predator_spawn_jitter_radius=args.predator_spawn_jitter_radius,
                        predator_pre_roll_steps=args.predator_pre_roll_steps,
                        predator_pre_roll_angle_jitter=args.predator_pre_roll_angle_jitter,
                        predator_pre_roll_speed_jitter=args.predator_pre_roll_speed_jitter,
                        predator_heading_bias=predator_heading_bias,
                    )
                    history_updated = True
                else:
                    callback._log(
                        f"best_checkpoint iteration={candidate_iter:03d} skipped (missing checkpoint at {checkpoint_zip})"
                    )

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
                escape_boost_speed=final_escape_boost_speed,
                escape_jitter_std=args.escape_jitter_std,
                divergence_reward_coef=args.divergence_reward_coef,
                density_penalty_coef=eval_density_penalty,
                density_target=args.density_target,
                predator_spawn_jitter_radius=args.predator_spawn_jitter_radius,
                predator_pre_roll_steps=args.predator_pre_roll_steps,
                predator_pre_roll_angle_jitter=args.predator_pre_roll_angle_jitter,
                predator_pre_roll_speed_jitter=args.predator_pre_roll_speed_jitter,
                predator_heading_bias=predator_heading_bias,
                predator_pre_roll_speed_bias=predator_speed_bias,
            )

        if eval_results_single:
            single_rates = [episode.get("avg_num_alive", 0.0) for episode in eval_results_single]
            post_eval_writer.add_scalar("eval_single/avg_alive", float(np.mean(single_rates)), tb_global_step)
            single_steps = [episode.get("episode_length", 0.0) for episode in eval_results_single]
            post_eval_writer.add_scalar("eval_single/avg_steps", float(np.mean(single_steps)), tb_global_step)

        if eval_multi_summary:
            for key, value in eval_multi_summary.items():
                post_eval_writer.add_scalar(f"eval_multi/{key}", float(value), tb_global_step)

        if death_stats_summary:
            for key, value in death_stats_summary.items():
                if isinstance(value, (int, float)):
                    post_eval_writer.add_scalar(f"eval_multi_death/{key}", float(value), tb_global_step)
    else:
        callback._log("skip_final_eval enabled; final deterministic eval/video skipped.")

    if history_updated and callback.multi_eval_history_path:
        callback.multi_eval_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(callback.multi_eval_history_path, "w", encoding="utf-8") as f:
            for entry in callback.multi_eval_history:
                f.write(json.dumps(entry) + "\n")

    post_eval_writer.flush()
    post_eval_writer.close()

    if multi_eval_tb_writer:
        multi_eval_tb_writer.flush()
        multi_eval_tb_writer.close()

    return {
        "run_name": run_name,
        "stats": callback.stats,
        "plot_path": plot_path,
        "gif_path": gif_path,
        "first_death_plot": first_death_plot_path,
        "penalty_alignment_plot": penalty_alignment_plot_path,
        "penalty_entropy_plot": penalty_entropy_plot_path,
        "checkpoint_dir": run_checkpoint_dir,
        "eval_single": eval_path_single,
        "eval_multi": eval_path_multi,
        "eval_multi_summary": eval_multi_summary_path,
        "death_plot": death_plot_path,
        "death_stats": death_stats_path,
        "death_stats_summary": death_stats_summary,
        "step_one_deaths": step_one_death_path,
        "log_path": run_log_path,
        "tb_log_dir": tb_run_dir,
        "video_path": video_path,
        "multi_eval_history": multi_eval_history_path,
        "multi_eval_plot": multi_eval_plot_path,
        "multi_eval_hist_dir": multi_eval_hist_dir,
        "best_checkpoint_iteration": best_checkpoint_iteration,
        "best_checkpoint_entry": best_checkpoint_entry_path,
        "best_eval_path": best_eval_path,
        "best_eval_summary": best_eval_summary_path,
        "best_eval_video": best_video_best_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Fish RL dev_v32")
    parser.add_argument("--total_iterations", type=int, default=64, help="迭代次数 (rollout count)")
    parser.add_argument("--num_envs", type=int, default=128, help="并行环境数量 (>=64 per SOP)")
    parser.add_argument("--num_fish", type=int, default=25, help="默认训练时的鱼数量 (无 curriculum 时生效)")
    parser.add_argument(
        "--curriculum",
        type=str,
        default="15:12,20:14,25:8,25:8,25:10,25:4,25:4",
        help="逗号分隔的 num_fish:iterations 阶段定义，留空表示禁用"
    )
    parser.add_argument("--n_steps", type=int, default=512, help="每个环境的 rollout 步数")
    parser.add_argument("--batch_size", type=int, default=1024, help="PPO 批大小")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_schedule", type=str, choices=("constant", "cosine", "warm_cosine"), default="warm_cosine", help="学习率日程")
    parser.add_argument("--policy_hidden_sizes", type=str, default="384,384", help="逗号分隔的 MLP 隐层尺寸，如 '384,384'")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="PPO entropy 系数")
    parser.add_argument(
        "--entropy_coef_schedule",
        type=str,
        default="16-24:0.03,25-32:0.02",
        help="迭代区间的 entropy 系数覆盖，格式 start-end:value，例如 '16-24:0.03,25+:0.015'"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="保存 checkpoint 的 iteration 间隔")
    parser.add_argument("--eval_episodes", type=int, default=5, help="训练结束后 deterministic 评估 EP 数")
    parser.add_argument("--eval_multi_fish", type=int, default=25, help="多鱼评估时的鱼数量 (>1 启用)")
    parser.add_argument(
        "--eval_multi_episodes",
        type=int,
        default=40,
        help="多鱼 deterministic 评估 EP 数 (建议 >=32 以降低尾部方差)",
    )
    parser.add_argument("--best_checkpoint_eval_episodes", type=int, default=32, help="best checkpoint 复核用的 deterministic EP 数")
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="跳过训练收尾阶段的 deterministic eval、best checkpoint 复核与媒体导出",
    )
    parser.add_argument(
        "--best_checkpoint_max_step_one",
        type=int,
        default=8,
        help="仅当 step_one_death_count <= 该值时才更新 best checkpoint (<=0 表示关闭 gating)",
    )
    parser.add_argument(
        "--best_checkpoint_max_early_death",
        type=float,
        default=0.1,
        help="仅当 early_death_fraction_100 <= 该值时才更新 best checkpoint (<=0 表示关闭 gating)",
    )
    parser.add_argument(
        "--multi_eval_interval",
        type=int,
        default=16,
        help="训练过程中触发多鱼探针评估的 iteration 间隔 (<=0 表示关闭)"
    )
    parser.add_argument(
        "--multi_eval_probe_episodes",
        type=int,
        default=20,
        help="每次训练期中多鱼评估的 episodes 数 (>=10 推荐)"
    )
    parser.add_argument("--adaptive_boost_window", type=int, default=4, help="first_death_p10 滑窗长度 (iteration)")
    parser.add_argument("--adaptive_boost_lower_p10", type=float, default=12.0, help="低于该 first_death_p10 时强制降速")
    parser.add_argument("--adaptive_boost_upper_p10", type=float, default=18.0, help="高于该 first_death_p10 时放宽 clamp")
    parser.add_argument("--adaptive_boost_low_speed", type=float, default=0.76, help="降速后的 escape boost 上限")
    parser.add_argument("--adaptive_boost_high_speed", type=float, default=0.8, help="放宽后的 escape boost 上限")
    parser.add_argument("--adaptive_boost_min_iteration", type=int, default=12, help="自适应 clamp 生效的最小 iteration")
    parser.add_argument(
        "--adaptive_boost_stage_floor",
        type=str,
        default="",
        help="按 penalty stage 设定 escape boost 最小值 (格式 stage:value, 逗号分隔)",
    )
    parser.add_argument(
        "--adaptive_boost_use_median",
        dest="adaptive_boost_use_median",
        action="store_true",
        help="使用滑窗 median 驱动 escape boost 调整",
    )
    parser.add_argument(
        "--adaptive_boost_use_mean",
        dest="adaptive_boost_use_median",
        action="store_false",
        help="改回滑窗均值驱动 escape boost",
    )
    parser.set_defaults(adaptive_boost_use_median=True)
    parser.add_argument("--disable_adaptive_boost", action="store_true", help="禁用 first_death_p10 驱动的 escape boost 调整")
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
    parser.add_argument("--escape_boost_speed", type=float, default=0.75, help="初始逃逸脉冲系数 (乘以 FISH_MAX_SPEED)")
    parser.add_argument(
        "--escape_boost_phase_speeds",
        type=str,
        default="3:0.76,4:0.78,5:0.82,6:0.82,7:0.82",
        help="按阶段覆盖 escape boost 速度，例如 '3:0.76,4:0.78,5:0.82,6:0.82,7:0.82'"
    )
    parser.add_argument(
        "--escape_boost_phase_floors",
        type=str,
        default="3:0.76,4:0.78,5:0.80,6:0.80,7:0.80",
        help="按阶段设定 escape boost 的地板值，防止 gate 未晋级时跌回过低速度"
    )
    parser.add_argument(
        "--escape_boost_penalty_caps",
        type=str,
        default="0.05:0.8",
        help="当密度 penalty 超过阈值时的 escape boost 上限，格式 threshold:value，如 '0.05:0.8,0.075:0.78'"
    )
    parser.add_argument(
        "--escape_boost_floor_default",
        type=float,
        default=0.75,
        help="当 phase floor 未显式指定时使用的默认 escape boost 地板"
    )
    parser.add_argument("--escape_jitter_std", type=float, default=0.35, help="初始逃逸方向噪声 (弧度标准差)")
    parser.add_argument("--divergence_reward_coef", type=float, default=0.0, help="邻居散度正向奖励系数")
    parser.add_argument("--density_penalty_coef", type=float, default=0.0, help="邻居密度惩罚系数 (penalize > target)")
    parser.add_argument("--density_target", type=float, default=0.4, help="密度惩罚触发的归一化阈值 (0~1)")
    parser.add_argument("--predator_spawn_jitter_radius", type=float, default=1.6, help="初始化时大鱼在圆心附近的随机半径 (世界坐标)")
    parser.add_argument("--predator_pre_roll_steps", type=int, default=16, help="reset 后仅移动大鱼的预热步数，缓解 step=1 碰撞")
    parser.add_argument(
        "--predator_pre_roll_angle_jitter",
        type=float,
        default=0.3,
        help="pre-roll 期间给大鱼速度方向加入的随机弧度幅度 (0 表示关闭)",
    )
    parser.add_argument(
        "--predator_pre_roll_speed_jitter",
        type=float,
        default=0.2,
        help="pre-roll 期间对大鱼速度模长施加的对称比例抖动 (0 表示关闭)",
    )
    parser.add_argument(
        "--predator_heading_bias",
        type=str,
        default="0-60:1.1,60-90:0.8,90-120:0.08,120-150:0.05,150-210:0.9,210-270:1.3,270-330:1.2,330-360:0.95",
        help="pre-roll 前大鱼速度方向的加权范围 (start-end:weight, 逗号分隔)",
    )
    parser.add_argument(
        "--predator_pre_roll_speed_bias",
        type=str,
        default="0-1.0:1.6,1.0-1.6:1.3,1.6-2.0:0.7,2.0-2.4:0.35,2.4-3.0:0.2",
        help="pre-roll 速度区间加权，格式 'start-end:weight' 逗号分隔，例如 '0-1.2:1.0,1.2-3.0:2.0'",
    )
    parser.add_argument(
        "--density_penalty_phase_targets",
        type=str,
        default="0.0,0.02,0.04,0.05,0.055,0.06,0.065",
        help="逗号分隔的 per-phase penalty 目标值 (长度需等于 curriculum 阶段)"
    )
    parser.add_argument(
        "--density_penalty_ramp_phases",
        type=str,
        default="6,7",
        help="需要线性 ramp 的阶段编号 (1-based, 逗号分隔)"
    )
    parser.add_argument(
        "--density_penalty_phase_plateaus",
        type=str,
        default="0,0,6,6,4,0,0",
        help="per-phase plateau 迭代数 (逗号分隔, 与 curriculum 长度一致)，用于延迟 ramp"
    )
    parser.add_argument(
        "--density_penalty_lock_value",
        type=float,
        default=0.04,
        help="当 iteration <= lock_until 时，penalty 不超过该值 (<=0 表示关闭)",
    )
    parser.add_argument(
        "--density_penalty_lock_until",
        type=int,
        default=28,
        help="锁定 penalty 的最大 iteration (<=0 表示关闭)",
    )
    parser.add_argument(
        "--penalty_gate_phase_allowance",
        type=str,
        default="4,5,6",
        help="1-based phase indices（按 curriculum 阶段编号）逐级解锁，长度决定 gate 可进入的阶段数量",
    )
    parser.add_argument(
        "--penalty_gate_required_successes",
        type=int,
        default=1,
        help="连续多少次 multi-eval 满足条件后才解锁下一阶段 (<=0 禁用 gate)",
    )
    parser.add_argument(
        "--penalty_gate_freeze_iterations",
        type=int,
        default=4,
        help="失败回退后冻结 penalty 的 iteration 数，期间不允许再次提升",
    )
    parser.add_argument(
        "--penalty_gate_success_step_one",
        type=int,
        default=7,
        help="判定成功解锁所需的 step-one 死亡上限",
    )
    parser.add_argument(
        "--penalty_gate_success_step_one_ratio",
        type=float,
        default=0.0,
        help="判定成功解锁所需的 step-one 死亡占比上限 (step-one / (episodes*num_fish)，<=0 表示禁用)",
    )
    parser.add_argument(
        "--penalty_gate_success_early_death",
        type=float,
        default=0.10,
        help="判定成功解锁所需的 early_death_fraction_100 上限",
    )
    parser.add_argument(
        "--penalty_gate_success_early_death_window",
        type=int,
        default=2,
        help="early_death_fraction_100 需要连续满足的窗口长度（>=2 使用滚动 median）",
    )
    parser.add_argument(
        "--penalty_gate_success_early_death_by_stage",
        type=str,
        default="",
        help="按 tail_stage_type 指定 early-death 阈值，格式 'ne:0.13,main:0.10' (留空禁用)",
    )
    parser.add_argument(
        "--penalty_gate_success_early_death_window_by_stage",
        type=str,
        default="",
        help="按 stage 覆盖 early-death 滑窗长度 (stage:window)",
    )
    parser.add_argument(
        "--penalty_gate_success_p10",
        type=float,
        default=120.0,
        help="判定成功解锁所需的 first_death_p10 下限 (<=0 表示忽略)",
    )
    parser.add_argument(
        "--penalty_gate_success_first_death_p10_by_stage",
        type=str,
        default="",
        help="按 stage 覆盖 success_p10 阈值，例如 'main:100'",
    )
    parser.add_argument(
        "--penalty_gate_success_freeze_iterations",
        type=int,
        default=1,
        help="成功后冻结阶段的 iteration 数，避免刚达标就触发失败",
    )
    parser.add_argument(
        "--penalty_gate_success_median_window",
        type=int,
        default=4,
        help="计算 success_p10 滚动 median 时的窗口长度",
    )
    parser.add_argument(
        "--penalty_gate_success_p10_median_window_by_stage",
        type=str,
        default="",
        help="按 stage 指定 rolling_p10 median 的窗口大小 (stage:int)",
    )
    parser.add_argument(
        "--penalty_gate_failure_p10",
        type=float,
        default=95.0,
        help="若 multi-eval 中 first_death_p10 低于该阈值则立即回退阶段",
    )
    parser.add_argument(
        "--penalty_gate_failure_min_final",
        type=float,
        default=0.5,
        help="若 multi-eval 中 min_final_survival_rate 低于该阈值则立即回退阶段",
    )
    parser.add_argument(
        "--penalty_gate_failure_tolerance",
        type=int,
        default=2,
        help="允许连续多少次 failure 后才真正回退阶段",
    )
    parser.add_argument(
        "--penalty_gate_failure_tolerance_by_stage",
        type=str,
        default="",
        help="按 stage 设置 failure 容忍度，例如 'main:3'",
    )
    parser.add_argument(
        "--penalty_gate_lock_iteration",
        type=int,
        default=-1,
        help="gate 解锁前的最小 iteration，<=0 时复用 density_penalty_lock_until",
    )
    parser.add_argument("--video_num_fish", type=int, default=32, help="录制 mp4 时的鱼数量 (<=0 跳过)")
    parser.add_argument("--video_max_steps", type=int, default=500, help="视频最多录制的步数")
    parser.add_argument("--video_fps", type=int, default=20, help="输出 mp4 的帧率")
    parser.add_argument(
        "--tail_replay_count",
        type=int,
        default=0,
        help="每次 multi-eval 自动录制的 tail 最差样本数量 (0 表示关闭)",
    )
    parser.add_argument(
        "--tail_seed_replay_path",
        type=str,
        default="",
        help="用于预热的 step_one_worst_seeds.json 路径 (为空则禁用)",
    )
    parser.add_argument(
        "--tail_seed_stage_spec",
        type=str,
        default="",
        help="按阶段加载 tail seeds，格式 'iterations:path|type=ne|label=foo;...' (提供后覆盖传统单文件配置)",
    )
    parser.add_argument(
        "--tail_stage_warn_ratio",
        type=float,
        default=0.65,
        help="tail_queue_remaining_ratio 低于该阈值且连续满足 patience 次后触发告警 (<=0 关闭)",
    )
    parser.add_argument(
        "--tail_stage_warn_patience",
        type=int,
        default=3,
        help="触发 tail stage warning 所需的连续 iteration 数",
    )
    parser.add_argument(
        "--tail_seed_prewarm_iterations",
        type=int,
        default=0,
        help="使用 tail seeds 预热的 iteration 数量 (0 表示关闭)",
    )
    parser.add_argument(
        "--tail_force_reset_steps",
        type=int,
        default=0,
        help="若 >0，则在单鱼环境运行指定步数后强制截断 episode 以便消耗 tail seeds",
    )
    parser.add_argument(
        "--multi_eval_seed_base",
        type=int,
        default=None,
        help="固定 multi-eval RNG 的起始种子，便于 tail replay 复现 (默认继承 --seed)",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--run_name", type=str, default=None, help="自定义 run 名称 (可选)")

    args = parser.parse_args()
    train(args)
