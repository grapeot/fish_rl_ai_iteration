import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from stable_baselines3 import PPO
from fish_env import FishEscapeEnv
import imageio
from tqdm import tqdm


def plot_training_curves(stats_path="./checkpoints/training_stats.pkl", output_path="./training_curves.png"):
    """绘制训练曲线"""
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fish Escape RL Training Progress', fontsize=16, fontweight='bold')
    
    iterations = stats['iterations']
    
    # 存活率
    ax = axes[0, 0]
    ax.plot(iterations, stats['survival_rates'], 'b-', linewidth=2, label='Survival Rate')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Survival Rate', fontsize=12)
    ax.set_title('Fish Survival Rate Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.legend()
    
    # 平均奖励
    ax = axes[0, 1]
    ax.plot(iterations, stats['avg_rewards'], 'g-', linewidth=2, label='Avg Reward')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Average Episode Reward', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 平均时间步
    ax = axes[1, 0]
    ax.plot(iterations, stats['avg_timesteps'], 'r-', linewidth=2, label='Avg Timesteps')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Timesteps', fontsize=12)
    ax.set_title('Average Episode Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 综合指标
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(iterations, stats['survival_rates'], 'b-', linewidth=2, label='Survival Rate')
    l2 = ax2.plot(iterations, stats['avg_rewards'], 'g-', linewidth=2, label='Avg Reward')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Survival Rate', fontsize=12, color='b')
    ax2.set_ylabel('Average Reward', fontsize=12, color='g')
    ax.set_title('Combined Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # 合并图例
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def render_episode(model_path, output_path, num_fish=250, max_steps=500):
    """渲染一个episode并保存为GIF"""
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建环境
    env = FishEscapeEnv(num_fish=num_fish, render_mode="rgb_array")
    
    obs, _ = env.reset()
    frames = []
    
    print(f"Rendering episode for {model_path}...")
    for step in tqdm(range(max_steps)):
        frame = env.render()
        frames.append(frame)
        
        # 为所有存活的鱼获取动作
        if len(obs) == 0:
            break
        
        actions = []
        for ob in obs:
            action, _ = model.predict(ob, deterministic=True)
            actions.append(action)
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # 保存为GIF
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Episode animation saved to {output_path}")
    
    return info


def create_comparison_visualization(checkpoint_iterations, output_dir="./visualizations"):
    """为多个检查点创建可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    for iteration in checkpoint_iterations:
        model_path = f"./checkpoints/model_iter_{iteration}"
        if os.path.exists(model_path + ".zip"):
            output_path = os.path.join(output_dir, f"episode_iter_{iteration}.gif")
            info = render_episode(model_path, output_path, num_fish=250, max_steps=500)
            print(f"Iteration {iteration}: {info}")


def create_single_frame_comparison(checkpoint_iterations, output_path="./comparison.png"):
    """创建多个检查点的单帧对比图"""
    num_checkpoints = len(checkpoint_iterations)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Fish Escape Learning Progress - Snapshots at Different Iterations', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, iteration in enumerate(checkpoint_iterations):
        model_path = f"./checkpoints/model_iter_{iteration}"
        if not os.path.exists(model_path + ".zip"):
            continue
        
        # 加载模型
        model = PPO.load(model_path)
        env = FishEscapeEnv(num_fish=250, render_mode="rgb_array")
        
        obs, _ = env.reset(seed=42)  # 固定种子以便对比
        
        # 运行100步
        for step in range(100):
            if len(obs) == 0:
                break
            
            actions = []
            for ob in obs:
                action, _ = model.predict(ob, deterministic=True)
                actions.append(action)
            
            obs, _, terminated, truncated, info = env.step(actions)
            
            if terminated or truncated:
                break
        
        # 获取当前帧
        frame = env.render()
        env.close()
        
        # 显示
        ax = axes[idx]
        ax.imshow(frame)
        ax.set_title(f'Iteration {iteration}\nAlive: {info.get("num_alive", 0)}/250', 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(checkpoint_iterations), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # 绘制训练曲线
    if os.path.exists("./checkpoints/training_stats.pkl"):
        plot_training_curves()
    
    # 创建检查点可视化
    checkpoint_iterations = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    
    # 创建单帧对比图
    if all(os.path.exists(f"./checkpoints/model_iter_{i}.zip") for i in checkpoint_iterations):
        create_single_frame_comparison(checkpoint_iterations)
    
    # 创建完整动画（可选，比较耗时）
    # create_comparison_visualization(checkpoint_iterations)
