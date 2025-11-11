import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import pickle
from fish_env_easy import FishEscapeEnvEasy


class VectorizedFishEnv(gym.Env):
    """向量化的鱼环境"""
    def __init__(self, num_fish=10):
        super().__init__()
        self.base_env = FishEscapeEnvEasy(num_fish=num_fish)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.num_fish = num_fish
        self.current_obs = None
        self.step_count = 0
        self.episode_reward = 0
        
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self.current_obs = obs
        self.step_count = 0
        self.episode_reward = 0
        
        if len(obs) > 0:
            return obs[0], info
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), info
    
    def step(self, action):
        num_alive = len(self.current_obs)
        
        if num_alive == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {
                'num_alive': 0,
                'survival_rate': 0.0
            }
        
        # 所有活着的鱼使用相同的策略
        actions = [action] * num_alive
        
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        self.current_obs = obs
        self.step_count += 1
        
        avg_reward = np.mean(rewards) if len(rewards) > 0 else -50.0
        self.episode_reward += avg_reward
        
        if len(obs) > 0:
            next_obs = obs[0]
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info['episode_reward'] = self.episode_reward
        info['survival_rate'] = info.get('num_alive', 0) / self.num_fish
        
        return next_obs, avg_reward, terminated, truncated, info
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()


class CheckpointCallback(BaseCallback):
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
            'num_alive': []
        }
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        self.iteration += 1
        
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = list(self.model.ep_info_buffer)
            
            survival_rates = []
            num_alives = []
            for ep in recent_episodes:
                if 'survival_rate' in ep:
                    survival_rates.append(ep['survival_rate'])
                if 'num_alive' in ep:
                    num_alives.append(ep['num_alive'])
            
            survival_rate = np.mean(survival_rates) if survival_rates else 0.0
            avg_reward = np.mean([ep['r'] for ep in recent_episodes])
            avg_timestep = np.mean([ep['l'] for ep in recent_episodes])
            avg_num_alive = np.mean(num_alives) if num_alives else 0.0
            
            self.stats['iterations'].append(self.iteration)
            self.stats['survival_rates'].append(survival_rate)
            self.stats['avg_rewards'].append(avg_reward)
            self.stats['avg_timesteps'].append(avg_timestep)
            self.stats['num_alive'].append(avg_num_alive)
            
            if self.verbose > 0 and self.iteration % 3 == 0:
                print(f"Iter {self.iteration:3d}: Survival={survival_rate:6.2%}, "
                      f"Alive={avg_num_alive:5.2f}, Reward={avg_reward:8.2f}, Steps={avg_timestep:5.1f}")
        
        if self.iteration in self.checkpoint_iterations:
            model_path = os.path.join(self.save_path, f"model_iter_{self.iteration}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"✓ Checkpoint saved at iteration {self.iteration}")
        
        return True


def train_easy(total_iterations=60, num_fish=10, num_envs=4):
    """简化版训练"""
    checkpoint_iterations = list(range(10, total_iterations + 1, 10))
    
    print(f"Creating {num_envs} parallel environments with {num_fish} fish each...")
    
    def make_env():
        return VectorizedFishEnv(num_fish=num_fish)
    
    envs = DummyVecEnv([make_env for _ in range(num_envs)])
    envs = VecMonitor(envs)
    
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # 更多探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]))
    )
    
    print(f"Starting training for {total_iterations} iterations...")
    print("=" * 80)
    
    callback = CheckpointCallback(
        save_path="./checkpoints/",
        checkpoint_iterations=checkpoint_iterations,
        verbose=1
    )
    
    total_timesteps = total_iterations * 512 * num_envs
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    model.save("./checkpoints/model_final")
    
    with open("./checkpoints/training_stats.pkl", "wb") as f:
        pickle.dump(callback.stats, f)
    
    print("=" * 80)
    print("Training completed!")
    if len(callback.stats['survival_rates']) > 0:
        print(f"Final survival rate: {callback.stats['survival_rates'][-1]:.2%}")
        print(f"Final avg alive: {callback.stats['num_alive'][-1]:.2f}/{num_fish}")
        print(f"Final avg reward: {callback.stats['avg_rewards'][-1]:.2f}")
    
    return model, callback.stats


if __name__ == "__main__":
    model, stats = train_easy(
        total_iterations=60,
        num_fish=10,
        num_envs=4
    )
