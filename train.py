import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import pickle
from fish_env import FishEscapeEnv


class MultiAgentWrapper(gym.Wrapper):
    """
    将多智能体环境包装成单智能体环境
    所有小鱼共享同一个策略
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_fish_idx = 0
        self.all_observations = None
        self.all_rewards = None
        self.episode_rewards = []
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.all_observations = obs
        self.current_fish_idx = 0
        self.episode_rewards = []
        
        if len(obs) > 0:
            return obs[0], info
        else:
            # 如果没有存活的鱼，返回零观测
            return np.zeros(self.observation_space.shape, dtype=np.float32), info
    
    def step(self, action):
        # 收集所有存活小鱼的动作
        num_alive = len(self.all_observations)
        
        if num_alive == 0:
            # 没有存活的鱼
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
        
        # 为所有小鱼执行相同的策略（但观测不同）
        # 这里我们需要为每条鱼获取动作
        # 简化处理：先只处理一条鱼的训练
        actions = [action]  # 只处理第一条鱼
        
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        self.all_observations = obs
        self.all_rewards = rewards
        
        # 返回第一条存活小鱼的观测和奖励
        if len(obs) > 0:
            return obs[0], float(rewards[0]), terminated, truncated, info
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, info


class CheckpointCallback(BaseCallback):
    """
    在特定iteration保存模型和统计信息
    """
    def __init__(self, check_freq: int, save_path: str, checkpoint_iterations: list, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.checkpoint_iterations = checkpoint_iterations
        self.iteration = 0
        self.stats = {
            'iterations': [],
            'survival_rates': [],
            'avg_rewards': [],
            'avg_timesteps': []
        }
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        self.iteration += 1
        
        # 记录统计信息
        if len(self.model.ep_info_buffer) > 0:
            survival_rate = np.mean([ep_info.get('survival_rate', 0) for ep_info in self.model.ep_info_buffer])
            avg_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            avg_timestep = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            
            self.stats['iterations'].append(self.iteration)
            self.stats['survival_rates'].append(survival_rate)
            self.stats['avg_rewards'].append(avg_reward)
            self.stats['avg_timesteps'].append(avg_timestep)
            
            if self.verbose > 0:
                print(f"Iteration {self.iteration}: Survival Rate = {survival_rate:.2%}, "
                      f"Avg Reward = {avg_reward:.2f}, Avg Timesteps = {avg_timestep:.1f}")
        
        # 在检查点保存模型
        if self.iteration in self.checkpoint_iterations:
            model_path = os.path.join(self.save_path, f"model_iter_{self.iteration}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved checkpoint at iteration {self.iteration}")
        
        return True


def make_env():
    """创建环境"""
    env = FishEscapeEnv(num_fish=250)
    env = MultiAgentWrapper(env)
    return env


def train_ppo(total_iterations=300, checkpoint_iterations=None):
    """
    训练PPO模型
    """
    if checkpoint_iterations is None:
        checkpoint_iterations = list(range(30, total_iterations + 1, 30))
    
    print("Creating environment...")
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 鼓励探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/",
    )
    
    print(f"Starting training for {total_iterations} iterations...")
    
    # 创建回调
    callback = CheckpointCallback(
        check_freq=1,
        save_path="./checkpoints/",
        checkpoint_iterations=checkpoint_iterations,
        verbose=1
    )
    
    # 训练
    # 每个iteration大约是一个rollout
    total_timesteps = total_iterations * 2048  # n_steps per iteration
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # 保存最终模型
    model.save("./checkpoints/model_final")
    
    # 保存统计信息
    with open("./checkpoints/training_stats.pkl", "wb") as f:
        pickle.dump(callback.stats, f)
    
    print("Training completed!")
    
    return model, callback.stats


if __name__ == "__main__":
    # 训练参数
    TOTAL_ITERATIONS = 300
    CHECKPOINT_ITERATIONS = list(range(30, TOTAL_ITERATIONS + 1, 30))
    
    # 开始训练
    model, stats = train_ppo(
        total_iterations=TOTAL_ITERATIONS,
        checkpoint_iterations=CHECKPOINT_ITERATIONS
    )
    
    print("\nTraining statistics saved to ./checkpoints/training_stats.pkl")
    print("Model checkpoints saved to ./checkpoints/")
