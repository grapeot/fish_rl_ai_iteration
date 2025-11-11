import numpy as np
from stable_baselines3 import PPO
from fish_env_easy import FishEscapeEnvEasy
import matplotlib.pyplot as plt
import sys


def test_model(model_path, num_episodes=5, render=False):
    """测试模型并收集统计信息"""
    model = PPO.load(model_path)
    env = FishEscapeEnvEasy(num_fish=10, render_mode="human" if render else None)
    
    episode_stats = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0
        timestep = 0
        death_times = []  # 记录每条鱼死亡的时间
        initial_num_fish = 10
        
        while not done:
            # 为所有活着的鱼获取动作
            actions = []
            for ob in obs:
                action, _ = model.predict(ob, deterministic=True)
                actions.append(action)
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            total_reward += np.mean(rewards) if len(rewards) > 0 else 0
            timestep += 1
            
            # 记录死亡
            if len(obs) < initial_num_fish:
                num_dead = initial_num_fish - len(obs)
                for _ in range(num_dead):
                    death_times.append(timestep)
                initial_num_fish = len(obs)
            
            done = terminated or truncated
        
        episode_stats.append({
            'episode': ep,
            'total_reward': total_reward,
            'final_alive': info['num_alive'],
            'survival_rate': info['survival_rate'],
            'timesteps': timestep,
            'death_times': death_times
        })
        
        print(f"Episode {ep+1}: Alive={info['num_alive']}/10, "
              f"Survival={info['survival_rate']:.1%}, "
              f"Steps={timestep}, Reward={total_reward:.1f}")
    
    env.close()
    return episode_stats


def test_random_baseline(num_episodes=5):
    """测试随机策略作为基线"""
    env = FishEscapeEnvEasy(num_fish=10)
    
    episode_stats = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0
        timestep = 0
        
        while not done:
            actions = [env.action_space.sample() for _ in range(len(obs))]
            obs, rewards, terminated, truncated, info = env.step(actions)
            total_reward += np.mean(rewards) if len(rewards) > 0 else 0
            timestep += 1
            done = terminated or truncated
        
        episode_stats.append({
            'episode': ep,
            'total_reward': total_reward,
            'final_alive': info['num_alive'],
            'survival_rate': info['survival_rate'],
            'timesteps': timestep
        })
        
        print(f"Random Episode {ep+1}: Alive={info['num_alive']}/10, "
              f"Steps={timestep}, Reward={total_reward:.1f}")
    
    env.close()
    return episode_stats


def compare_models(checkpoint_iters):
    """比较不同checkpoint的性能"""
    results = {}
    
    # 测试随机策略
    print("\n" + "="*60)
    print("Testing RANDOM policy (baseline)...")
    print("="*60)
    random_stats = test_random_baseline(num_episodes=10)
    results['random'] = {
        'avg_reward': np.mean([s['total_reward'] for s in random_stats]),
        'avg_alive': np.mean([s['final_alive'] for s in random_stats]),
        'avg_steps': np.mean([s['timesteps'] for s in random_stats])
    }
    
    # 测试训练的模型
    for iter_num in checkpoint_iters:
        model_path = f"./checkpoints/model_iter_{iter_num}.zip"
        print(f"\n" + "="*60)
        print(f"Testing model at iteration {iter_num}...")
        print("="*60)
        
        try:
            stats = test_model(model_path, num_episodes=10)
            results[f'iter_{iter_num}'] = {
                'avg_reward': np.mean([s['total_reward'] for s in stats]),
                'avg_alive': np.mean([s['final_alive'] for s in stats]),
                'avg_steps': np.mean([s['timesteps'] for s in stats])
            }
        except Exception as e:
            print(f"Error loading model: {e}")
    
    return results


if __name__ == "__main__":
    import os
    
    # 检查是否有训练好的模型
    if not os.path.exists("./checkpoints/model_iter_10.zip"):
        print("No trained models found. Please run training first.")
        sys.exit(1)
    
    # 比较不同阶段的模型
    checkpoint_iters = [10, 20, 30, 40, 50, 60]
    results = compare_models(checkpoint_iters)
    
    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, value in results.items():
        print(f"{key:12s}: Reward={value['avg_reward']:7.1f}, "
              f"Alive={value['avg_alive']:.2f}, Steps={value['avg_steps']:.1f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    labels = list(results.keys())
    rewards = [results[k]['avg_reward'] for k in labels]
    alives = [results[k]['avg_alive'] for k in labels]
    steps = [results[k]['avg_steps'] for k in labels]
    
    axes[0].bar(range(len(labels)), rewards, color='skyblue')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Average Episode Reward')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(len(labels)), alives, color='lightgreen')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Average Fish Alive')
    axes[1].set_title('Average Fish Survival')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 10])
    
    axes[2].bar(range(len(labels)), steps, color='lightcoral')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].set_ylabel('Average Timesteps')
    axes[2].set_title('Average Episode Length')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison chart saved to ./model_comparison.png")
