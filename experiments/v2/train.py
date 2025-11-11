import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import pickle
from fish_env import FishEscapeEnv


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
LOG_DIR = ARTIFACTS_DIR / "logs"
TB_LOG_DIR = ARTIFACTS_DIR / "tb_logs"

for _dir in (CHECKPOINT_DIR, LOG_DIR, TB_LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class SingleFishEnv(gym.Env):
    """
    单条鱼的环境包装
    多个这样的环境并行训练，所有鱼共享策略
    """
    def __init__(self, num_fish=50, fish_id=0):
        super().__init__()
        self.base_env = FishEscapeEnv(num_fish=num_fish)
        self.fish_id = fish_id
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.current_obs = None
        self.fish_index_map = {}
        
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self.current_obs = obs
        
        # 找到我的鱼的索引
        if len(obs) > self.fish_id:
            return obs[self.fish_id], info
        elif len(obs) > 0:
            return obs[0], info
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), info
    
    def step(self, action):
        # 为所有鱼生成随机动作，但用我们的动作替换对应的鱼
        num_alive = len(self.current_obs)
        if num_alive == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32), -100.0, True, False, {}
        
        # 为所有鱼生成动作（其他鱼随机，我们的鱼用学到的策略）
        actions = [self.action_space.sample() for _ in range(num_alive)]
        if self.fish_id < num_alive:
            actions[self.fish_id] = action
        elif num_alive > 0:
            actions[0] = action  # 如果我的鱼已死，控制第一条活鱼
        
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        self.current_obs = obs
        
        # 返回我的鱼的观测和奖励
        if len(obs) > self.fish_id:
            return obs[self.fish_id], float(rewards[self.fish_id]), terminated, truncated, info
        elif len(obs) > 0:
            return obs[0], float(rewards[0]), terminated, truncated, info
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), -100.0, True, False, info
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()


class VectorizedFishEnv(gym.Env):
    """
    向量化的鱼环境 - 每个时间步处理所有活着的鱼
    """
    def __init__(self, num_fish=50):
        super().__init__()
        self.base_env = FishEscapeEnv(num_fish=num_fish)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.num_fish = num_fish
        self.current_obs = None
        self.step_count = 0
        self.episode_reward = 0
        # 用于跟踪整个episode的存活率
        self.episode_survival_sum = 0.0
        self.episode_steps = 0
        
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self.current_obs = obs
        self.step_count = 0
        self.episode_reward = 0
        self.episode_survival_sum = 0.0
        self.episode_steps = 0
        
        # 返回第一条鱼的观测
        if len(obs) > 0:
            return obs[0], info
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), info
    
    def step(self, action):
        """
        这里的action是单个动作，我们会应用到所有活着的鱼上
        """
        num_alive = len(self.current_obs)
        
        if num_alive == 0:
            # 计算平均存活率
            avg_survival_rate = self.episode_survival_sum / max(self.episode_steps, 1)
            avg_num_alive = avg_survival_rate * self.num_fish
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {
                'num_alive': 0,
                'survival_rate': avg_survival_rate,
                'avg_survival_rate': avg_survival_rate,
                'avg_num_alive': avg_num_alive,
                'final_num_alive': 0
            }
        
        # 所有活着的鱼使用相同的策略，但观测不同
        # 为了训练，我们让所有鱼都执行相同的动作（这是简化，实际应该每条鱼根据自己的观测决策）
        # 更好的做法是：每条鱼根据自己的观测独立决策
        actions = [action] * num_alive  # 简化：所有鱼执行相同动作
        
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        real_num_alive = info.get('num_alive', len(obs))
        self.current_obs = obs
        self.step_count += 1
        
        # 累计存活率（用于计算平均值）
        current_survival_rate = info.get('num_alive', 0) / self.num_fish
        self.episode_survival_sum += current_survival_rate
        self.episode_steps += 1
        
        # 计算平均奖励
        avg_reward = np.mean(rewards) if len(rewards) > 0 else -100.0
        self.episode_reward += avg_reward
        
        # 返回第一条活鱼的观测
        if len(obs) > 0:
            next_obs = obs[0]
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 更新info - 使用平均存活率而不是当前存活率
        avg_survival_rate = self.episode_survival_sum / self.episode_steps
        avg_num_alive = avg_survival_rate * self.num_fish
        info['episode_reward'] = self.episode_reward
        info['avg_survival_rate'] = avg_survival_rate
        info['avg_num_alive'] = avg_num_alive
        info['num_alive'] = real_num_alive
        if terminated or truncated:
            info['final_num_alive'] = real_num_alive

        return next_obs, avg_reward, terminated, truncated, info
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()


class CheckpointCallback(BaseCallback):
    """在特定iteration保存模型和统计信息"""
    def __init__(self, save_path: str, checkpoint_iterations: list, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.checkpoint_iterations = checkpoint_iterations
        self.iteration = 0
        self.stats = {
            'iterations': [],
            'survival_rates': [],
            'avg_rewards': [],
            'avg_timesteps': [],
            'num_alive': [],
            'avg_num_alive': [],
            'final_num_alive': []
        }
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        self.iteration += 1
        
        if len(self.model.ep_info_buffer) > 0:
            # 从episode info中提取信息
            recent_episodes = list(self.model.ep_info_buffer)
            
            survival_rates = []
            num_alives = []
            avg_num_alives = []
            final_num_alives = []
            for ep in recent_episodes:
                if 'survival_rate' in ep:
                    survival_rates.append(ep['survival_rate'])
                if 'num_alive' in ep:
                    num_alives.append(ep['num_alive'])
                if 'avg_num_alive' in ep:
                    avg_num_alives.append(ep['avg_num_alive'])
                if 'final_num_alive' in ep and ep['final_num_alive'] is not None:
                    final_num_alives.append(ep['final_num_alive'])
            
            survival_rate = np.mean(survival_rates) if survival_rates else 0.0
            avg_reward = np.mean([ep['r'] for ep in recent_episodes])
            avg_timestep = np.mean([ep['l'] for ep in recent_episodes])
            avg_num_alive = np.mean(num_alives) if num_alives else 0.0
            mean_avg_num_alive = np.mean(avg_num_alives) if avg_num_alives else avg_num_alive
            final_num_alive = np.mean(final_num_alives) if final_num_alives else avg_num_alive
            
            self.stats['iterations'].append(self.iteration)
            self.stats['survival_rates'].append(survival_rate)
            self.stats['avg_rewards'].append(avg_reward)
            self.stats['avg_timesteps'].append(avg_timestep)
            self.stats['num_alive'].append(avg_num_alive)
            self.stats['avg_num_alive'].append(mean_avg_num_alive)
            self.stats['final_num_alive'].append(final_num_alive)

            # 记录到 TensorBoard/logger，方便后续调试
            self.logger.record("custom/avg_survival_rate", float(survival_rate))
            self.logger.record("custom/avg_num_alive", float(mean_avg_num_alive))
            self.logger.record("custom/final_num_alive", float(final_num_alive))
            
            if self.verbose > 0 and self.iteration % 5 == 0:
                # 调试：打印第一个episode的详细信息
                if len(recent_episodes) > 0:
                    first_ep = recent_episodes[0]
                    debug_info = f" [Debug: ep_keys={list(first_ep.keys())}, "
                    if 'survival_rate' in first_ep:
                        debug_info += f"sr={first_ep['survival_rate']:.3f}, "
                    if 'num_alive' in first_ep:
                        debug_info += f"na={first_ep['num_alive']}]"
                    print(f"Iter {self.iteration:3d}: Survival(avg)={survival_rate:6.2%}, "
                          f"Alive(avg)={mean_avg_num_alive:5.1f}, Final={final_num_alive:5.1f}, "
                          f"Reward={avg_reward:8.2f}, Steps={avg_timestep:5.1f}{debug_info}")
                else:
                    print(f"Iter {self.iteration:3d}: Survival(avg)={survival_rate:6.2%}, "
                          f"Alive(avg)={mean_avg_num_alive:5.1f}, Final={final_num_alive:5.1f}, "
                          f"Reward={avg_reward:8.2f}, Steps={avg_timestep:5.1f}")
        
        if self.iteration in self.checkpoint_iterations:
            model_path = os.path.join(self.save_path, f"model_iter_{self.iteration}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"✓ Checkpoint saved at iteration {self.iteration}")
        
        # 实时保存统计信息（每5个iteration保存一次，避免频繁IO）
        if self.iteration % 5 == 0:
            stats_path = os.path.join(self.save_path, "training_stats.pkl")
            with open(stats_path, "wb") as f:
                pickle.dump(self.stats, f)
        
        return True


def train_ppo_v2(total_iterations=100, num_fish=50, num_envs=4):
    """
    改进的PPO训练 - 使用多进程并行环境
    
    使用 SubprocVecEnv 实现真正的多进程并行，可以充分利用多核CPU，
    相比 DummyVecEnv 在CPU密集型任务上有更好的性能。
    """
    checkpoint_iterations = list(range(10, total_iterations + 1, 10))
    
    print(f"Creating {num_envs} parallel environments with {num_fish} fish each...")
    
    # 创建多个并行环境（使用多进程）
    def make_env():
        return VectorizedFishEnv(num_fish=num_fish)
    
    envs = SubprocVecEnv([make_env for _ in range(num_envs)], start_method='spawn')
    envs = VecMonitor(
        envs,
        info_keywords=('survival_rate', 'num_alive', 'final_num_alive', 'avg_survival_rate', 'avg_num_alive')
    )
    
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        n_steps=1024,  # 增加步数，减少进程间同步频率，提高CPU利用率
        batch_size=256,  # 增加batch size以匹配更多数据
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # 增加探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(TB_LOG_DIR),
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )
    
    print(f"Starting training for {total_iterations} iterations...")
    print("=" * 80)
    
    callback = CheckpointCallback(
        save_path=str(CHECKPOINT_DIR),
        checkpoint_iterations=checkpoint_iterations,
        verbose=1
    )
    
    # 总时间步 = iterations * n_steps * num_envs
    total_timesteps = total_iterations * 1024 * num_envs
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    model.save(str(CHECKPOINT_DIR / "model_final"))
    
    with open(CHECKPOINT_DIR / "training_stats.pkl", "wb") as f:
        pickle.dump(callback.stats, f)
    
    print("=" * 80)
    print("Training completed!")
    if len(callback.stats['survival_rates']) > 0:
        print(f"Final survival rate: {callback.stats['survival_rates'][-1]:.2%}")
        print(f"Final avg alive: {callback.stats['num_alive'][-1]:.1f}")
        print(f"Final avg reward: {callback.stats['avg_rewards'][-1]:.2f}")
    
    return model, callback.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for FishEscapeEnv")
    parser.add_argument("--total_iterations", type=int, default=100, help="训练迭代次数")
    parser.add_argument("--num_fish", type=int, default=25, help="每个环境中的小鱼数量")
    parser.add_argument("--num_envs", type=int, default=32, help="并行环境数量")
    args = parser.parse_args()

    model, stats = train_ppo_v2(
        total_iterations=args.total_iterations,
        num_fish=args.num_fish,
        num_envs=args.num_envs
    )
