import gymnasium as gym

# Create a simple environment perfect for beginners
env = gym.make('Humanoid-v5', render_mode="human")

observation, info = env.reset()

print(f"Obs: {observation}")
print(f"Action space: {env.action_space}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info  = env.step(action)

    total_reward += reward
    # episode_over = terminated or truncated

print(f"Episode finished. Total reward: {total_reward}")
env.close()