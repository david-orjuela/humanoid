import gymnasium as gym
import torch
import numpy as np

from model import SimpleNeuralNetwork
# Create a simple environment perfect for beginners
env = gym.make('Humanoid-v5', render_mode="human")

observation, info = env.reset()

print(f"Obs: {observation}")
print(f"Action space: {env.action_space}")

episode_over = False
total_reward = 0

prev_obs = None
model = SimpleNeuralNetwork(input_dim=348, output_dim=17)
i = 0
while not episode_over:
    i += 1
    if i == 5: episode_over = True
    if prev_obs is not None:
        action = model(torch.tensor(prev_obs, dtype=torch.float32))
        action = action.detach().cpu().numpy()
        print(f"Model action: {action}")
    else:
        action = env.action_space.sample()
        print(f"Sampled action: {action}")

    obs, reward, terminated, truncated, info  = env.step(action)
    
    prev_obs = obs

    total_reward += reward
    # episode_over = terminated or truncated

print(f"Episode finished. Total reward: {total_reward}")
env.close()