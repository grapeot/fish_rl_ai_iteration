from fish_env import FishEscapeEnv
import numpy as np

# 测试环境
print("Testing FishEscapeEnv...")
env = FishEscapeEnv(num_fish=10)  # 先用少量鱼测试

obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Number of alive fish: {len(obs)}")

# 运行几步
for step in range(10):
    actions = [env.action_space.sample() for _ in range(len(obs))]
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"Step {step+1}: Alive={info['num_alive']}, Survival Rate={info['survival_rate']:.2%}")
    
    if terminated:
        print("Episode terminated!")
        break

print("\nEnvironment test passed!")
env.close()
