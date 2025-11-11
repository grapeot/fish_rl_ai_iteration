import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, List


class FishEscapeEnv(gym.Env):
    """
    强化学习环境：小鱼躲避大鱼
    - 圆形舞台
    - 1条大鱼（确定性物理运动）
    - 250条小鱼（共享策略）
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_fish: int = 250,
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
    ):
        super().__init__()
        
        # 环境参数
        self.STAGE_RADIUS = 10.0
        self.NUM_FISH = num_fish
        self.MAX_TIMESTEPS = 500
        
        # 小鱼参数
        self.FISH_MAX_SPEED = 2.0
        self.FISH_MAX_TURN_ANGLE = np.pi / 6  # 30度
        self.FISH_VISION_RADIUS = self.STAGE_RADIUS / 3
        self.FISH_SIZE = 0.2
        self.FISH_ACCELERATION = 1.0
        
        # 大鱼参数
        self.PREDATOR_SIZE = 0.5
        self.PREDATOR_GRAVITY = 0.5  # 增加重力，让大鱼更快
        self.PREDATOR_INITIAL_VX = 1.5  # 增加初始速度
        self.PREDATOR_INITIAL_VY = 0.0
        self.PREDATOR_BOUNCE_DAMPING = 0.85
        
        # 时间步长
        self.dt = 0.1
        # 奖励缩放，降低单步奖励量级，便于稳定训练
        self.REWARD_SCALE = 0.1
        
        # 邻居特征 & 初始逃逸控制
        self.include_neighbor_features = include_neighbor_features
        self.neighbor_radius = max(neighbor_radius, 1e-3)
        self.neighbor_average_count = max(neighbor_average_count, 1)
        self.initial_escape_boost = initial_escape_boost
        self.escape_boost_speed = float(np.clip(escape_boost_speed, 0.0, 1.0))
        self.escape_jitter_std = float(max(escape_jitter_std, 0.0))
        self.divergence_reward_coef = float(divergence_reward_coef)
        self.density_penalty_coef = float(density_penalty_coef)
        self.density_target = float(np.clip(density_target, 0.0, 1.0))
        self.predator_spawn_jitter_radius = max(float(predator_spawn_jitter_radius), 0.0)
        self.predator_pre_roll_steps = max(int(predator_pre_roll_steps), 0)
        self.predator_pre_roll_angle_jitter = max(float(predator_pre_roll_angle_jitter), 0.0)

        # 定义观测空间（单条小鱼的观测）
        # [自身x, 自身y, 自身vx, 自身vy, 到边界距离,
        #  大鱼可见标志, 大鱼相对x, 大鱼相对y, 大鱼相对vx, 大鱼相对vy, 大鱼距离,
        #  (可选) 邻居密度、平均相对x、平均相对y、最近邻距离及速度/散度统计]
        base_dim = 11
        neighbor_dim = 7 if self.include_neighbor_features else 0
        self._obs_dim = base_dim + neighbor_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        
        # 定义动作空间（离散动作）
        # 0: 前进, 1: 左转, 2: 右转, 3: 减速, 4: 保持
        self.action_space = spaces.Discrete(5)
        
        # 渲染相关
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.screen_size = 800
        
        # 状态变量
        self.timestep = 0
        self.fish_positions = None
        self.fish_velocities = None
        self.fish_alive = None
        self.predator_pos = None
        self.predator_vel = None
        self.fish_death_timesteps = None
        self.step_one_death_records = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.timestep = 0
        
        # 初始化小鱼位置（随机分布在圆内）
        self.fish_positions = np.zeros((self.NUM_FISH, 2), dtype=np.float32)
        self.fish_velocities = np.zeros((self.NUM_FISH, 2), dtype=np.float32)
        self.fish_alive = np.ones(self.NUM_FISH, dtype=bool)
        self.fish_death_timesteps = np.full(self.NUM_FISH, -1, dtype=np.int32)
        self.step_one_death_records = []
        
        for i in range(self.NUM_FISH):
            # 随机角度和半径
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(0, self.STAGE_RADIUS * 0.8)
            self.fish_positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]

            # 随机初始速度
            v_angle = self.np_random.uniform(0, 2 * np.pi)
            v_mag = self.np_random.uniform(0, self.FISH_MAX_SPEED * 0.5)
            self.fish_velocities[i] = [v_mag * np.cos(v_angle), v_mag * np.sin(v_angle)]

        if self.initial_escape_boost:
            for i in range(self.NUM_FISH):
                radial = self.fish_positions[i].copy()
                norm = np.linalg.norm(radial)
                if norm < 1e-5:
                    radial = self.np_random.normal(0, 1, size=2)
                    norm = np.linalg.norm(radial)
                radial /= max(norm, 1e-6)
                jitter = self.np_random.normal(0.0, self.escape_jitter_std)
                cos_a, sin_a = np.cos(jitter), np.sin(jitter)
                rotated = np.array([
                    radial[0] * cos_a - radial[1] * sin_a,
                    radial[0] * sin_a + radial[1] * cos_a,
                ])
                boost_speed = self.FISH_MAX_SPEED * self.escape_boost_speed
                self.fish_velocities[i] = rotated * boost_speed
        
        # 初始化大鱼位置（圆心附近）
        self.predator_pos = np.array([0.0, 0.0], dtype=np.float32)
        if self.predator_spawn_jitter_radius > 0:
            max_radius = min(self.predator_spawn_jitter_radius, self.STAGE_RADIUS * 0.8)
            jitter_radius = self.np_random.uniform(0.0, max_radius)
            jitter_angle = self.np_random.uniform(0.0, 2 * np.pi)
            offset = np.array([
                jitter_radius * np.cos(jitter_angle),
                jitter_radius * np.sin(jitter_angle),
            ], dtype=np.float32)
            self.predator_pos = offset
        self.predator_vel = np.array([self.PREDATOR_INITIAL_VX, self.PREDATOR_INITIAL_VY], dtype=np.float32)

        if self.predator_pre_roll_steps > 0:
            self._apply_predator_pre_roll(self.predator_pre_roll_steps)
        
        # 获取初始观测
        observations = self._get_observations()
        
        if self.render_mode == "human":
            self._render_frame()

        return observations, {}

    def _apply_predator_pre_roll(self, steps: int):
        """Advance predator only before正式计时，避免 step=1 的空间重叠。"""
        for _ in range(max(int(steps), 0)):
            if self.predator_pre_roll_angle_jitter > 0.0 and self.predator_vel is not None:
                angle = self.np_random.uniform(
                    -self.predator_pre_roll_angle_jitter,
                    self.predator_pre_roll_angle_jitter,
                )
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                vx, vy = float(self.predator_vel[0]), float(self.predator_vel[1])
                rotated = np.array(
                    [vx * cos_a - vy * sin_a, vx * sin_a + vy * cos_a],
                    dtype=np.float32,
                )
                self.predator_vel = rotated
            self._update_predator()

    def set_density_penalty(self, coef: float, target: Optional[float] = None):
        """动态更新密度惩罚参数，供训练过程中调整 ramp。"""
        self.density_penalty_coef = float(coef)
        if target is not None:
            self.density_target = float(np.clip(target, 0.0, 1.0))
    
    def set_escape_boost_speed(self, speed: float):
        """允许训练过程中调整初始逃逸速度，便于做 gating 实验。"""
        self.escape_boost_speed = float(np.clip(speed, 0.0, 1.0))
    
    def step(self, actions):
        """
        执行一步仿真
        actions: 所有存活小鱼的动作数组
        """
        self.timestep += 1
        
        # 记录死亡前的状态
        prev_alive = self.fish_alive.copy()
        
        # 更新小鱼状态
        alive_indices = np.where(self.fish_alive)[0]
        for idx, fish_idx in enumerate(alive_indices):
            if idx < len(actions):
                action = actions[idx]
                self._update_fish(fish_idx, action)
        
        # 更新大鱼状态（确定性物理运动）
        self._update_predator()
        
        # 碰撞检测
        self._check_collisions()

        just_died = np.where(np.logical_and(prev_alive, np.logical_not(self.fish_alive)))[0]
        if self.timestep == 1 and just_died.size > 0:
            for fish_idx in just_died:
                self._record_step_one_death(fish_idx)
        
        # 计算奖励（包括死亡惩罚）
        rewards = self._compute_rewards()
        
        # 给刚刚死亡的鱼添加死亡惩罚
        for i in range(self.NUM_FISH):
            if prev_alive[i] and not self.fish_alive[i]:
                rewards[i] -= 50.0  # 死亡时的单次惩罚
        
        # 检查终止条件
        terminated = (np.sum(self.fish_alive) == 0) or (self.timestep >= self.MAX_TIMESTEPS)
        truncated = False
        
        # 获取观测
        observations = self._get_observations()
        
        if self.render_mode == "human":
            self._render_frame()
        
        info = {
            "num_alive": np.sum(self.fish_alive),
            "survival_rate": np.sum(self.fish_alive) / self.NUM_FISH,
            "timestep": self.timestep
        }

        if self.fish_death_timesteps is not None:
            valid = self.fish_death_timesteps[self.fish_death_timesteps >= 0]
            info["first_death_step"] = int(valid.min()) if valid.size > 0 else -1
        else:
            info["first_death_step"] = -1

        if terminated or truncated:
            info["death_timesteps"] = (
                self.fish_death_timesteps.tolist() if self.fish_death_timesteps is not None else None
            )

        info["step_one_deaths"] = [dict(record) for record in (self.step_one_death_records or [])]

        return observations, rewards, terminated, truncated, info
    
    def _update_fish(self, fish_idx: int, action: int):
        """更新单条小鱼的状态"""
        pos = self.fish_positions[fish_idx]
        vel = self.fish_velocities[fish_idx]
        
        # 当前速度方向
        speed = np.linalg.norm(vel)
        if speed > 0.01:
            direction = vel / speed
        else:
            direction = np.array([1.0, 0.0])
        
        # 根据动作更新速度
        if action == 0:  # 前进
            vel += direction * self.FISH_ACCELERATION * self.dt
        elif action == 1:  # 左转
            angle = self.FISH_MAX_TURN_ANGLE
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            new_dir = np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a
            ])
            vel = new_dir * speed
        elif action == 2:  # 右转
            angle = -self.FISH_MAX_TURN_ANGLE
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            new_dir = np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a
            ])
            vel = new_dir * speed
        elif action == 3:  # 减速
            vel *= 0.9
        # action == 4: 保持
        
        # 限制最大速度
        speed = np.linalg.norm(vel)
        if speed > self.FISH_MAX_SPEED:
            vel = vel / speed * self.FISH_MAX_SPEED
        
        # 更新位置
        pos += vel * self.dt
        
        # 边界约束（弹性反弹）
        dist = np.linalg.norm(pos)
        if dist > self.STAGE_RADIUS:
            # 法向量
            normal = pos / dist
            # 速度分解
            v_normal = np.dot(vel, normal) * normal
            v_tangent = vel - v_normal
            # 反弹
            vel = -v_normal * 0.5 + v_tangent
            # 修正位置
            pos = normal * self.STAGE_RADIUS
        
        self.fish_positions[fish_idx] = pos
        self.fish_velocities[fish_idx] = vel
    
    def _update_predator(self):
        """更新大鱼状态（确定性物理运动）"""
        # 施加重力
        self.predator_vel[1] += self.PREDATOR_GRAVITY * self.dt
        
        # 更新位置
        self.predator_pos += self.predator_vel * self.dt
        
        # 边界碰撞检测
        dist = np.linalg.norm(self.predator_pos)
        if dist > self.STAGE_RADIUS:
            # 法向量（从圆心指向碰撞点）
            normal = self.predator_pos / dist
            # 速度分解
            v_normal = np.dot(self.predator_vel, normal) * normal
            v_tangent = self.predator_vel - v_normal
            # 反弹（法向分量反向并施加阻尼）
            self.predator_vel = -v_normal * self.PREDATOR_BOUNCE_DAMPING + v_tangent
            # 修正位置
            self.predator_pos = normal * self.STAGE_RADIUS
    
    def _check_collisions(self):
        """检测大鱼与小鱼的碰撞"""
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                continue
            
            dist = np.linalg.norm(self.fish_positions[i] - self.predator_pos)
            if dist < (self.FISH_SIZE + self.PREDATOR_SIZE):
                self.fish_alive[i] = False
                if self.fish_death_timesteps is not None and self.fish_death_timesteps[i] < 0:
                    self.fish_death_timesteps[i] = self.timestep
    
    def _compute_rewards(self) -> np.ndarray:
        """计算每条小鱼的奖励"""
        rewards = np.zeros(self.NUM_FISH, dtype=np.float32)
        
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                # 死去的鱼不再获得奖励或惩罚（只在死亡时惩罚一次）
                rewards[i] = 0.0
                continue
            
            # 存活奖励（增加权重）
            rewards[i] += 2.0
            
            # 距离奖励（与大鱼保持距离）
            dist_to_predator = np.linalg.norm(self.fish_positions[i] - self.predator_pos)
            if dist_to_predator < self.FISH_VISION_RADIUS:
                # 在视野内，鼓励远离（增加奖励）
                rewards[i] += 10.0 * (dist_to_predator / self.FISH_VISION_RADIUS)
            else:
                # 不在视野内也给一些奖励
                rewards[i] += 5.0
            
            # 边界惩罚（减少惩罚）
            dist_to_center = np.linalg.norm(self.fish_positions[i])
            dist_to_boundary = self.STAGE_RADIUS - dist_to_center
            if dist_to_boundary < 1.0:
                rewards[i] -= 2.0 * (1.0 - dist_to_boundary)

            if (self.divergence_reward_coef != 0.0 or self.density_penalty_coef != 0.0):
                rel_array, dist_array, _, neighbor_divergence = self._collect_neighbor_stats(i)
                if rel_array is not None and dist_array is not None:
                    neighbor_count = len(dist_array)
                    density_ratio = neighbor_count / float(max(self.neighbor_average_count, 1))
                    if self.density_penalty_coef != 0.0:
                        over_density = max(density_ratio - self.density_target, 0.0)
                        rewards[i] -= self.density_penalty_coef * over_density
                    if self.divergence_reward_coef != 0.0 and neighbor_divergence.size > 0:
                        rewards[i] += self.divergence_reward_coef * float(np.mean(neighbor_divergence))

        return rewards * self.REWARD_SCALE

    def _record_step_one_death(self, fish_idx: int):
        if self.step_one_death_records is None:
            self.step_one_death_records = []
        fish_pos = self.fish_positions[fish_idx].astype(float).tolist()
        predator_pos = self.predator_pos.astype(float).tolist() if self.predator_pos is not None else [0.0, 0.0]
        predator_vel = self.predator_vel.astype(float).tolist() if self.predator_vel is not None else [0.0, 0.0]
        self.step_one_death_records.append({
            "fish_idx": int(fish_idx),
            "predator_index": 0,
            "fish_position": fish_pos,
            "predator_position": predator_pos,
            "predator_velocity": predator_vel,
            "timestep": int(self.timestep),
        })
    
    def _get_observations(self) -> np.ndarray:
        """获取所有存活小鱼的观测"""
        observations = []

        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                continue

            obs = np.zeros(self._obs_dim, dtype=np.float32)

            # 自身状态（归一化）
            obs[0] = self.fish_positions[i][0] / self.STAGE_RADIUS
            obs[1] = self.fish_positions[i][1] / self.STAGE_RADIUS
            obs[2] = self.fish_velocities[i][0] / self.FISH_MAX_SPEED
            obs[3] = self.fish_velocities[i][1] / self.FISH_MAX_SPEED

            # 到边界的距离
            dist_to_center = np.linalg.norm(self.fish_positions[i])
            obs[4] = (self.STAGE_RADIUS - dist_to_center) / self.STAGE_RADIUS

            # 大鱼相对信息
            relative_pos = self.predator_pos - self.fish_positions[i]
            dist_to_predator = np.linalg.norm(relative_pos)

            if dist_to_predator < self.FISH_VISION_RADIUS:
                obs[5] = 1.0  # 大鱼可见
                obs[6] = relative_pos[0] / self.FISH_VISION_RADIUS
                obs[7] = relative_pos[1] / self.FISH_VISION_RADIUS
                obs[8] = self.predator_vel[0] / self.FISH_MAX_SPEED
                obs[9] = self.predator_vel[1] / self.FISH_MAX_SPEED
                obs[10] = dist_to_predator / self.FISH_VISION_RADIUS
            else:
                obs[5] = 0.0  # 大鱼不可见

            if self.include_neighbor_features:
                offset = 11
                rel_array, dist_array, neighbor_speeds, neighbor_divergence = self._collect_neighbor_stats(i)

                if rel_array is not None and dist_array is not None:
                    top_k = min(len(dist_array), self.neighbor_average_count)
                    top_indices = np.argsort(dist_array)[:top_k]
                    rel_subset = rel_array[top_indices]
                    dist_subset = dist_array[top_indices]
                    obs[offset] = min(len(dist_array) / float(self.neighbor_average_count), 1.0)
                    obs[offset + 1] = float(np.mean(rel_subset[:, 0]) / self.neighbor_radius)
                    obs[offset + 2] = float(np.mean(rel_subset[:, 1]) / self.neighbor_radius)
                    obs[offset + 3] = float(np.min(dist_subset) / self.neighbor_radius)
                    obs[offset + 4] = float(np.mean(neighbor_speeds))
                    obs[offset + 5] = float(np.std(neighbor_speeds))
                    obs[offset + 6] = float(np.mean(neighbor_divergence))
                else:
                    obs[offset:offset + 7] = 0.0

            observations.append(obs)

            
        return np.array(observations, dtype=np.float32)

    def _collect_neighbor_stats(self, fish_idx: int):
        neighbor_vectors: List[np.ndarray] = []
        neighbor_distances: List[float] = []
        neighbor_speeds: List[float] = []
        neighbor_divergence: List[float] = []

        for j in range(self.NUM_FISH):
            if j == fish_idx or not self.fish_alive[j]:
                continue
            rel = self.fish_positions[j] - self.fish_positions[fish_idx]
            dist = np.linalg.norm(rel)
            if dist <= self.neighbor_radius:
                neighbor_vectors.append(rel)
                neighbor_distances.append(dist)
                speed = np.linalg.norm(self.fish_velocities[j])
                neighbor_speeds.append(float(speed / max(self.FISH_MAX_SPEED, 1e-6)))
                rel_dir = rel / max(dist, 1e-6)
                neighbor_divergence.append(
                    float(np.dot(self.fish_velocities[j], rel_dir) / max(self.FISH_MAX_SPEED, 1e-6))
                )

        if not neighbor_vectors:
            return None, None, None, None

        rel_array = np.array(neighbor_vectors, dtype=np.float32)
        dist_array = np.array(neighbor_distances, dtype=np.float32)
        speeds = np.array(neighbor_speeds, dtype=np.float32)
        divergences = np.array(neighbor_divergence, dtype=np.float32)
        return rel_array, dist_array, speeds, divergences
    
    def _render_frame(self):
        """渲染当前帧"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Fish Escape RL")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.screen_size, self.screen_size))
        canvas.fill((255, 255, 255))
        
        # 坐标转换函数
        def world_to_screen(pos):
            x = int((pos[0] / self.STAGE_RADIUS + 1) * self.screen_size / 2)
            y = int((1 - pos[1] / self.STAGE_RADIUS) * self.screen_size / 2)
            return (x, y)
        
        # 绘制舞台边界
        # 世界坐标边界 STAGE_RADIUS 应该映射到屏幕边缘
        # 坐标转换: x = (pos[0] / STAGE_RADIUS + 1) * screen_size / 2
        # 当 pos[0] = STAGE_RADIUS 时，x = screen_size
        # 当 pos[0] = 0 时，x = screen_size / 2
        # 所以边界圆的半径应该是 screen_size / 2
        center = (self.screen_size // 2, self.screen_size // 2)
        radius = int(self.screen_size / 2)
        pygame.draw.circle(canvas, (200, 200, 200), center, radius, 2)
        
        # 绘制小鱼
        for i in range(self.NUM_FISH):
            if self.fish_alive[i]:
                pos = world_to_screen(self.fish_positions[i])
                pygame.draw.circle(canvas, (0, 150, 255), pos, 4)
        
        # 绘制大鱼
        predator_screen_pos = world_to_screen(self.predator_pos)
        pygame.draw.circle(canvas, (255, 0, 0), predator_screen_pos, 10)
        
        # 显示信息
        if self.render_mode == "human":
            font = pygame.font.Font(None, 36)
            text = font.render(f"Alive: {np.sum(self.fish_alive)}/{self.NUM_FISH}  Step: {self.timestep}", True, (0, 0, 0))
            canvas.blit(text, (10, 10))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
