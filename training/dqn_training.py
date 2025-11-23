import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import torch
from typing import Dict, Any

# Your environment
from environment.custom_fire_env import FireRescueEnv


# Action Wrapper
class FlattenActionWrapper(gym.ActionWrapper):
    """
    Converts Dict(steering, throttle) → Discrete(n)
    """
    def __init__(self, env):
        super().__init__(env)
        self.steering_actions = 5   # 0-4
        self.throttle_actions = 3   # 0-2
        self.action_space = gym.spaces.Discrete(self.steering_actions * self.throttle_actions)

    def action(self, act):
        steering = act // self.throttle_actions
        throttle = act % self.throttle_actions
        return {"steering": steering, "throttle": throttle}


def make_env(render_mode=None):
    env = FireRescueEnv(grid_size=(10, 10), render_mode=render_mode)
    env = FlattenActionWrapper(env)
    env = Monitor(env)
    return env


# Callback for reward tracking + model saving
class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            if len(self.model.ep_info_buffer) > 0:
                last_ep = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(last_ep['r'])
                self.episode_lengths.append(last_ep['l'])

                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose:
                        print(f"New best model saved! Mean reward = {mean_reward:.2f}")
                    self.model.save(self.save_path)

            # Track training loss
            if "train/loss" in self.model.logger.name_to_value:
                self.losses.append(self.model.logger.name_to_value["train/loss"])

        return True


# Hyperparameters — You can add your own variants

hyperparams = {
    "DQN-1": {
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "buffer_size": 120000,
        "batch_size": 32,
        "exploration_fraction": 0.25,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 8000,
        "learning_starts": 10000,
        "train_freq": (4, "step")
    },

    "DQN-2": {
        "learning_rate": 1e-3,
        "gamma": 0.98,
        "buffer_size": 100000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.1,
        "target_update_interval": 6000,
        "learning_starts": 5000,
        "train_freq": (4, "step")
    },

    "DQN-3": {
        "learning_rate": 2.5e-4,
        "gamma": 0.995,
        "buffer_size": 150000,
        "batch_size": 32,
        "exploration_fraction": 0.30,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "target_update_interval": 10000,
        "learning_starts": 15000,
        "train_freq": (2, "step")
    },

    "DQN-4": {
        "learning_rate": 7e-4,
        "gamma": 0.97,
        "buffer_size": 90000,
        "batch_size": 16,
        "exploration_fraction": 0.35,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.1,
        "target_update_interval": 4000,
        "learning_starts": 5000,
        "train_freq": (1, "step")
    },

    "DQN-5": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 200000,
        "batch_size": 128,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 15000,
        "learning_starts": 20000,
        "train_freq": (8, "step")
    },

    "DQN-6": {
        "learning_rate": 5e-4,
        "gamma": 0.97,
        "buffer_size": 50000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.2,
        "target_update_interval": 3000,
        "learning_starts": 3000,
        "train_freq": (4, "step")
    },

    "DQN-7": {
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "buffer_size": 160000,
        "batch_size": 32,
        "exploration_fraction": 0.25,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 12000,
        "learning_starts": 15000,
        "train_freq": (4, "step")
    },

    "DQN-8": {
        "learning_rate": 8e-4,
        "gamma": 0.96,
        "buffer_size": 110000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.15,
        "target_update_interval": 8000,
        "learning_starts": 7000,
        "train_freq": (2, "step")
    },

    "DQN-9": {
        "learning_rate": 6e-4,
        "gamma": 0.985,
        "buffer_size": 130000,
        "batch_size": 32,
        "exploration_fraction": 0.28,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 9000,
        "learning_starts": 9000,
        "train_freq": (4, "step")
    },

    "DQN-10": {
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 256,
        "exploration_fraction": 0.18,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.03,
        "target_update_interval": 14000,
        "learning_starts": 20000,
        "train_freq": (4, "step")
    },
}



# Training Function
def train_and_evaluate(config: Dict[str, Any],
                       config_name: str,
                       total_timesteps: int = 700000):

    print(f"\nTraining configuration: {config_name}")
    print(config)

    # Logging
    log_dir = f"logs_{config_name}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()
    eval_env = make_env()

    # Network architecture
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        exploration_fraction=config["exploration_fraction"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_final_eps=config["exploration_final_eps"],
        target_update_interval=config["target_update_interval"],
        learning_starts=config["learning_starts"],
        train_freq=config["train_freq"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"
    )

    # Callbacks
    reward_callback = RewardCallback(check_freq=1000, log_dir=log_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=15000,
        n_eval_episodes=5,
        deterministic=True
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([reward_callback, eval_callback]),
        tb_log_name=config_name
    )

    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

    return {
        "model": model,
        "episode_rewards": reward_callback.episode_rewards,
        "losses": reward_callback.losses,
        "log_dir": log_dir,
        "config_name": config_name
    }


# Plotting utilities
def plot_rewards(results_dict):
    plt.figure(figsize=(12,6))
    for cfg_name, result in results_dict.items():
        episodes = range(1, len(result["episode_rewards"]) + 1)
        plt.plot(episodes, result["episode_rewards"], label=cfg_name)

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("DQN Reward Progression")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_losses(results_dict):
    plt.figure(figsize=(12,6))
    for cfg_name, result in results_dict.items():
        steps = range(1, len(result["losses"]) + 1)
        plt.plot(steps, result["losses"], label=cfg_name)

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("DQN Loss Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_individual_training(result):
    plt.figure(figsize=(12,5))

    # Rewards
    plt.subplot(1,2,1)
    plt.plot(result["episode_rewards"])
    plt.title(f"{result['config_name']} - Episode Rewards")
    plt.grid(True)

    # Loss
    plt.subplot(1,2,2)
    plt.plot(result["losses"])
    plt.title(f"{result['config_name']} - Training Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Convergence Analysis
def analyze_convergence(results_dict, window_size=10, threshold=0.05):
    convergence_data = {}

    for cfg, result in results_dict.items():
        rewards = result["episode_rewards"]

        if len(rewards) < window_size:
            continue

        moving_avg = np.convolve(
            rewards, np.ones(window_size)/window_size, mode="valid"
        )

        stable_episode = None
        for i in range(len(moving_avg) - window_size):
            segment = moving_avg[i:i+window_size]
            if np.std(segment) < threshold * np.mean(segment):
                stable_episode = i + window_size
                break

        convergence_data[cfg] = {
            "stable_episode": stable_episode,
            "final_reward": np.mean(rewards[-window_size:]),
            "max_reward": np.max(rewards),
            "total_episodes": len(rewards)
        }

    return convergence_data


# Main Training Loop
def main():
    os.makedirs("comparison_plots", exist_ok=True)

    results = {}

    for cfg_name, cfg in hyperparams.items():
        result = train_and_evaluate(cfg, cfg_name)
        results[cfg_name] = result
        plot_individual_training(result)

    plot_rewards(results)
    plot_losses(results)

    convergence = analyze_convergence(results)
    print("\nConvergence ")
    for cfg, stats in convergence.items():
        print(f"\nConfig: {cfg}")
        print(f"  Stabilized after episode: {stats['stable_episode']}")
        print(f"  Final reward: {stats['final_reward']:.2f}")
        print(f"  Max reward: {stats['max_reward']:.2f}")
        print(f"  Episodes: {stats['total_episodes']}")


if __name__ == "__main__":
    main()
