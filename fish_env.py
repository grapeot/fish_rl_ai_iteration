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
    
    def __init__(self, render_mode: Optional[str] = None, num_fish: int = 250):
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
        self.PREDATOR_GRAVITY = 0.3  # 降低重力，让大鱼慢一点
        self.PREDATOR_INITIAL_VX = 0.8  # 降低初始速度
        self.PREDATOR_INITIAL_VY = 0.0
        self.PREDATOR_BOUNCE_DAMPING = 0.85
        
        # 时间步长
        self.dt = 0.1
        
        # 定义观测空间（单条小鱼的观测）
        # [自身x, 自身y, 自身vx, 自身vy, 到边界距离, 
        #  大鱼可见标志, 大鱼相对x, 大鱼相对y, 大鱼相对vx, 大鱼相对vy, 大鱼距离]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
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
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.timestep = 0
        
        # 初始化小鱼位置（随机分布在圆内）
        self.fish_positions = np.zeros((self.NUM_FISH, 2), dtype=np.float32)
        self.fish_velocities = np.zeros((self.NUM_FISH, 2), dtype=np.float32)
        self.fish_alive = np.ones(self.NUM_FISH, dtype=bool)
        
        for i in range(self.NUM_FISH):
            # 随机角度和半径
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(0, self.STAGE_RADIUS * 0.8)
            self.fish_positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]
            
            # 随机初始速度
            v_angle = self.np_random.uniform(0, 2 * np.pi)
            v_mag = self.np_random.uniform(0, self.FISH_MAX_SPEED * 0.5)
            self.fish_velocities[i] = [v_mag * np.cos(v_angle), v_mag * np.sin(v_angle)]
        
        # 初始化大鱼位置（圆心附近）
        self.predator_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.predator_vel = np.array([self.PREDATOR_INITIAL_VX, self.PREDATOR_INITIAL_VY], dtype=np.float32)
        
        # 获取初始观测
        observations = self._get_observations()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, {}
    
    def step(self, actions):
        """
        执行一步仿真
        actions: 所有存活小鱼的动作数组
        """
        self.timestep += 1
        
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
        
        # 计算奖励
        rewards = self._compute_rewards()
        
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
    
    def _compute_rewards(self) -> np.ndarray:
        """计算每条小鱼的奖励"""
        rewards = np.zeros(self.NUM_FISH, dtype=np.float32)
        
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                rewards[i] = -100.0  # 死亡惩罚
                continue
            
            # 存活奖励（增加权重）
            rewards[i] += 1.0
            
            # 距离奖励（与大鱼保持距离）
            dist_to_predator = np.linalg.norm(self.fish_positions[i] - self.predator_pos)
            if dist_to_predator < self.FISH_VISION_RADIUS:
                # 在视野内，鼓励远离（增加奖励）
                rewards[i] += 5.0 * (dist_to_predator / self.FISH_VISION_RADIUS)
            else:
                # 不在视野内也给一些奖励
                rewards[i] += 2.0
            
            # 边界惩罚（减少惩罚）
            dist_to_center = np.linalg.norm(self.fish_positions[i])
            dist_to_boundary = self.STAGE_RADIUS - dist_to_center
            if dist_to_boundary < 1.0:
                rewards[i] -= 1.0 * (1.0 - dist_to_boundary)
        
        return rewards
    
    def _get_observations(self) -> np.ndarray:
        """获取所有存活小鱼的观测"""
        observations = []
        
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                continue
            
            obs = np.zeros(11, dtype=np.float32)
            
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
            
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
    
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
        center = (self.screen_size // 2, self.screen_size // 2)
        radius = int(self.screen_size / 2 * 0.95)
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
