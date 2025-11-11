import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional


class FishEscapeEnvEasy(gym.Env):
    """
    简化版小鱼躲避环境 - 用于验证学习机制
    - 更少的鱼
    - 更慢的大鱼
    - 更短的episode
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None, num_fish: int = 10):
        super().__init__()
        
        # 环境参数
        self.STAGE_RADIUS = 10.0
        self.NUM_FISH = num_fish
        self.MAX_TIMESTEPS = 300  # 缩短episode
        
        # 小鱼参数
        self.FISH_MAX_SPEED = 2.5  # 提高小鱼速度
        self.FISH_MAX_TURN_ANGLE = np.pi / 4  # 增加转向能力
        self.FISH_VISION_RADIUS = self.STAGE_RADIUS / 2  # 扩大视野
        self.FISH_SIZE = 0.15
        self.FISH_ACCELERATION = 1.5
        
        # 大鱼参数 - 大幅降低威胁
        self.PREDATOR_SIZE = 0.4
        self.PREDATOR_GRAVITY = 0.15  # 非常慢的重力
        self.PREDATOR_INITIAL_VX = 0.4  # 非常慢的初始速度
        self.PREDATOR_INITIAL_VY = 0.0
        self.PREDATOR_BOUNCE_DAMPING = 0.9
        
        # 时间步长
        self.dt = 0.1
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # 动作空间
        self.action_space = spaces.Discrete(5)
        
        # 渲染
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.screen_size = 800
        
        # 状态
        self.timestep = 0
        self.fish_positions = None
        self.fish_velocities = None
        self.fish_alive = None
        self.predator_pos = None
        self.predator_vel = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.timestep = 0
        
        # 初始化小鱼 - 分散在圆内
        self.fish_positions = np.zeros((self.NUM_FISH, 2), dtype=np.float32)
        self.fish_velocities = np.zeros((self.NUM_FISH, 2), dtype=np.float32)
        self.fish_alive = np.ones(self.NUM_FISH, dtype=bool)
        
        for i in range(self.NUM_FISH):
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(3, self.STAGE_RADIUS * 0.9)  # 远离中心
            self.fish_positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]
            
            v_angle = self.np_random.uniform(0, 2 * np.pi)
            v_mag = self.np_random.uniform(0, self.FISH_MAX_SPEED * 0.3)
            self.fish_velocities[i] = [v_mag * np.cos(v_angle), v_mag * np.sin(v_angle)]
        
        # 大鱼从中心开始
        self.predator_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.predator_vel = np.array([self.PREDATOR_INITIAL_VX, self.PREDATOR_INITIAL_VY], dtype=np.float32)
        
        observations = self._get_observations()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, {}
    
    def step(self, actions):
        self.timestep += 1
        
        # 更新小鱼
        alive_indices = np.where(self.fish_alive)[0]
        for idx, fish_idx in enumerate(alive_indices):
            if idx < len(actions):
                action = actions[idx]
                self._update_fish(fish_idx, action)
        
        # 更新大鱼
        self._update_predator()
        
        # 碰撞检测
        self._check_collisions()
        
        # 计算奖励
        rewards = self._compute_rewards()
        
        # 终止条件
        terminated = (np.sum(self.fish_alive) == 0) or (self.timestep >= self.MAX_TIMESTEPS)
        truncated = False
        
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
        pos = self.fish_positions[fish_idx]
        vel = self.fish_velocities[fish_idx]
        
        speed = np.linalg.norm(vel)
        if speed > 0.01:
            direction = vel / speed
        else:
            direction = np.array([1.0, 0.0])
        
        if action == 0:  # 加速
            vel += direction * self.FISH_ACCELERATION * self.dt
        elif action == 1:  # 左转
            angle = self.FISH_MAX_TURN_ANGLE
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            new_dir = np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a
            ])
            vel = new_dir * max(speed, 1.0)  # 保持一定速度
        elif action == 2:  # 右转
            angle = -self.FISH_MAX_TURN_ANGLE
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            new_dir = np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a
            ])
            vel = new_dir * max(speed, 1.0)
        elif action == 3:  # 减速
            vel *= 0.85
        # action == 4: 保持
        
        speed = np.linalg.norm(vel)
        if speed > self.FISH_MAX_SPEED:
            vel = vel / speed * self.FISH_MAX_SPEED
        
        pos += vel * self.dt
        
        # 边界处理
        dist = np.linalg.norm(pos)
        if dist > self.STAGE_RADIUS:
            normal = pos / dist
            v_normal = np.dot(vel, normal) * normal
            v_tangent = vel - v_normal
            vel = -v_normal * 0.5 + v_tangent
            pos = normal * self.STAGE_RADIUS
        
        self.fish_positions[fish_idx] = pos
        self.fish_velocities[fish_idx] = vel
    
    def _update_predator(self):
        self.predator_vel[1] += self.PREDATOR_GRAVITY * self.dt
        self.predator_pos += self.predator_vel * self.dt
        
        dist = np.linalg.norm(self.predator_pos)
        if dist > self.STAGE_RADIUS:
            normal = self.predator_pos / dist
            v_normal = np.dot(self.predator_vel, normal) * normal
            v_tangent = self.predator_vel - v_normal
            self.predator_vel = -v_normal * self.PREDATOR_BOUNCE_DAMPING + v_tangent
            self.predator_pos = normal * self.STAGE_RADIUS
    
    def _check_collisions(self):
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                continue
            
            dist = np.linalg.norm(self.fish_positions[i] - self.predator_pos)
            if dist < (self.FISH_SIZE + self.PREDATOR_SIZE):
                self.fish_alive[i] = False
    
    def _compute_rewards(self) -> np.ndarray:
        rewards = np.zeros(self.NUM_FISH, dtype=np.float32)
        
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                rewards[i] = -50.0  # 降低死亡惩罚
                continue
            
            # 基础存活奖励
            rewards[i] += 2.0
            
            # 距离奖励
            dist_to_predator = np.linalg.norm(self.fish_positions[i] - self.predator_pos)
            
            if dist_to_predator < self.FISH_VISION_RADIUS:
                # 在视野内，大幅奖励远离
                normalized_dist = dist_to_predator / self.FISH_VISION_RADIUS
                rewards[i] += 10.0 * normalized_dist
                
                # 额外奖励：如果正在远离大鱼
                relative_pos = self.fish_positions[i] - self.predator_pos
                if np.linalg.norm(relative_pos) > 0.01:
                    relative_dir = relative_pos / np.linalg.norm(relative_pos)
                    fish_dir = self.fish_velocities[i]
                    if np.linalg.norm(fish_dir) > 0.01:
                        fish_dir = fish_dir / np.linalg.norm(fish_dir)
                        # 如果鱼的运动方向与远离大鱼的方向一致
                        alignment = np.dot(fish_dir, relative_dir)
                        if alignment > 0:
                            rewards[i] += 5.0 * alignment
            else:
                # 不在视野内
                rewards[i] += 5.0
            
            # 轻微的边界惩罚
            dist_to_center = np.linalg.norm(self.fish_positions[i])
            dist_to_boundary = self.STAGE_RADIUS - dist_to_center
            if dist_to_boundary < 0.5:
                rewards[i] -= 2.0 * (0.5 - dist_to_boundary)
        
        return rewards
    
    def _get_observations(self) -> np.ndarray:
        observations = []
        
        for i in range(self.NUM_FISH):
            if not self.fish_alive[i]:
                continue
            
            obs = np.zeros(11, dtype=np.float32)
            
            obs[0] = self.fish_positions[i][0] / self.STAGE_RADIUS
            obs[1] = self.fish_positions[i][1] / self.STAGE_RADIUS
            obs[2] = self.fish_velocities[i][0] / self.FISH_MAX_SPEED
            obs[3] = self.fish_velocities[i][1] / self.FISH_MAX_SPEED
            
            dist_to_center = np.linalg.norm(self.fish_positions[i])
            obs[4] = (self.STAGE_RADIUS - dist_to_center) / self.STAGE_RADIUS
            
            relative_pos = self.predator_pos - self.fish_positions[i]
            dist_to_predator = np.linalg.norm(relative_pos)
            
            if dist_to_predator < self.FISH_VISION_RADIUS:
                obs[5] = 1.0
                obs[6] = relative_pos[0] / self.FISH_VISION_RADIUS
                obs[7] = relative_pos[1] / self.FISH_VISION_RADIUS
                obs[8] = self.predator_vel[0] / self.FISH_MAX_SPEED
                obs[9] = self.predator_vel[1] / self.FISH_MAX_SPEED
                obs[10] = dist_to_predator / self.FISH_VISION_RADIUS
            else:
                obs[5] = 0.0
            
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Fish Escape RL (Easy)")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.screen_size, self.screen_size))
        canvas.fill((255, 255, 255))
        
        def world_to_screen(pos):
            x = int((pos[0] / self.STAGE_RADIUS + 1) * self.screen_size / 2)
            y = int((1 - pos[1] / self.STAGE_RADIUS) * self.screen_size / 2)
            return (x, y)
        
        center = (self.screen_size // 2, self.screen_size // 2)
        radius = int(self.screen_size / 2 * 0.95)
        pygame.draw.circle(canvas, (200, 200, 200), center, radius, 2)
        
        for i in range(self.NUM_FISH):
            if self.fish_alive[i]:
                pos = world_to_screen(self.fish_positions[i])
                pygame.draw.circle(canvas, (0, 150, 255), pos, 5)
        
        predator_screen_pos = world_to_screen(self.predator_pos)
        pygame.draw.circle(canvas, (255, 0, 0), predator_screen_pos, 12)
        
        if self.render_mode == "human":
            font = pygame.font.Font(None, 36)
            text = font.render(f"Alive: {np.sum(self.fish_alive)}/{self.NUM_FISH}  Step: {self.timestep}", 
                             True, (0, 0, 0))
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
