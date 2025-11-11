#!/usr/bin/env python3
"""
实时观看小鱼躲避大鱼的行为
"""
import sys
from stable_baselines3 import PPO
from fish_env import FishEscapeEnv
import argparse


def watch_model(model_path="./checkpoints/model_final", num_fish=25, num_episodes=1):
    """
    加载模型并实时观看小鱼的行为
    
    Args:
        model_path: 模型路径（不需要 .zip 后缀）
        num_fish: 小鱼数量
        num_episodes: 观看的 episode 数量
    """
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print(f"\nCreating environment with {num_fish} fish...")
    print("Press Ctrl+C to stop\n")
    
    # 创建环境（使用 human 模式实时显示）
    env = FishEscapeEnv(num_fish=num_fish, render_mode="human")
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}\n")
        
        obs, info = env.reset()
        step = 0
        
        while True:
            if len(obs) == 0:
                print("All fish are dead!")
                break
            
            # 为所有存活的鱼生成动作
            actions = []
            for ob in obs:
                action, _ = model.predict(ob, deterministic=True)
                actions.append(action)
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1
            
            # 每50步打印一次状态
            if step % 50 == 0:
                num_alive = info.get('num_alive', len(obs))
                survival_rate = info.get('survival_rate', num_alive / num_fish)
                print(f"Step {step:3d}: {num_alive:2d}/{num_fish} fish alive ({survival_rate:.1%})")
            
            if terminated or truncated:
                num_alive = info.get('num_alive', len(obs))
                survival_rate = info.get('survival_rate', num_alive / num_fish)
                print(f"\nEpisode finished!")
                print(f"Final: {num_alive}/{num_fish} fish alive ({survival_rate:.1%})")
                print(f"Total steps: {step}")
                break
    
    env.close()
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch fish escape behavior")
    parser.add_argument(
        "--model",
        type=str,
        default="./checkpoints/model_final",
        help="Path to model (default: ./checkpoints/model_final)"
    )
    parser.add_argument(
        "--num-fish",
        type=int,
        default=25,
        help="Number of fish (default: 25)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to watch (default: 1)"
    )
    
    args = parser.parse_args()
    
    watch_model(
        model_path=args.model,
        num_fish=args.num_fish,
        num_episodes=args.episodes
    )

