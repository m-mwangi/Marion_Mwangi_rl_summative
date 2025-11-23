import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from environment.custom_fire_env import FireRescueEnv
from stable_baselines3.common.monitor import Monitor


# Same Action Wrapper as other algorithms
class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steering_actions = 5
        self.throttle_actions = 3
        self.action_space = gym.spaces.Discrete(self.steering_actions * self.throttle_actions)

    def action(self, act):
        steering = act // self.throttle_actions
        throttle = act % self.throttle_actions
        return {"steering": steering, "throttle": throttle}


def make_env(render_mode=None):
    env = FireRescueEnv(grid_size=(10, 10), render_mode=render_mode)
    env = FlattenActionWrapper(env)
    return Monitor(env)


# REINFORCE Policy Network
class ReinforcePolicy(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def obs_to_tensor(obs):
    if isinstance(obs, dict):
        flat = []
        for v in obs.values():
            flat.extend(np.array(v, dtype=np.float32).flatten())
        return torch.tensor(flat, dtype=torch.float32)
    return torch.tensor(obs, dtype=torch.float32)


# REINFORCE Training
def train_reinforce(total_episodes=800, gamma=0.99, lr=1e-3):

    print("\nTraining REINFORCE from scratch...")

    env = make_env()
    n_actions = env.action_space.n

    # Get input vector length
    tmp_obs, _ = env.reset()
    input_dim = len(obs_to_tensor(tmp_obs))

    policy = ReinforcePolicy(input_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_episode_rewards = []
    all_losses = []

    for episode in range(total_episodes):
        obs, _ = env.reset()
        done = False

        log_probs = []
        rewards = []

        while not done:
            obs_t = obs_to_tensor(obs)

            probs = policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # Compute returns (Monte Carlo)
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Policy gradient loss
        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_episode_rewards.append(sum(rewards))
        all_losses.append(loss.item())

        if episode % 20 == 0:
            print(f"Episode {episode}/{total_episodes} | Reward={sum(rewards):.1f} | Loss={loss.item():.4f}")

    # SAVE LOG FILES
    os.makedirs("reinforce_logs", exist_ok=True)

    # save policy
    torch.save(policy.state_dict(), "reinforce_logs/final_model.pth")

    # save rewards
    with open("reinforce_logs/reinforce_rewards.txt", "w") as f:
        for r in all_episode_rewards:
            f.write(f"{r}\n")

    # save losses
    with open("reinforce_logs/reinforce_losses.txt", "w") as f:
        for l in all_losses:
            f.write(f"{l}\n")

    print("Saved REINFORCE logs → reinforce_logs/")

    return {
        "rewards": all_episode_rewards,
        "losses": all_losses
    }


# Run
def main():
    results = train_reinforce(total_episodes=800)
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(results["rewards"])
    plt.title("REINFORCE Rewards")

    plt.subplot(1,2,2)
    plt.plot(results["losses"])
    plt.title("REINFORCE Losses")
    plt.show()


if __name__ == "__main__":
    main()
