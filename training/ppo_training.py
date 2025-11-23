import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# YOUR environment
from environment.custom_fire_env import FireRescueEnv


# Action Wrapper
class FlattenActionWrapper(gym.ActionWrapper):
    """
    Convert Dict(steering, throttle) → Discrete(n)
    """
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
    env = Monitor(env)
    return env


# Reward + Entropy Tracking Callback
class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            self.rewards.append(mean_reward)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose:
                    print(f"New best PPO model with mean reward {mean_reward:.2f}")
                self.model.save(self.save_path)

        return True


class EntropyCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000):
        super().__init__()
        self.check_freq = check_freq
        self.entropies = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            entropy = float(self.model.logger.name_to_value.get("train/entropy_loss", 0))
            self.entropies.append(entropy)
        return True


# PPO Hyperparameter Configurations
ppo_hyperparams = {
    "PPO-1": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
    },

    "PPO-2": {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 0.98,
        "gae_lambda": 0.92,
        "clip_range": 0.15,
        "ent_coef": 0.02,
        "max_grad_norm": 0.5,
    },

    "PPO-3": {
        "learning_rate": 5e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "clip_range": 0.25,
        "ent_coef": 0.005,
        "max_grad_norm": 0.4,
    },

    "PPO-4": {
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 15,
        "gamma": 0.97,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.03,
        "max_grad_norm": 0.5,
    },

    "PPO-5": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.985,
        "gae_lambda": 0.90,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "max_grad_norm": 0.5,
    },

    "PPO-6": {
        "learning_rate": 7e-4,
        "n_steps": 2048,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "ent_coef": 0.01,
        "max_grad_norm": 0.3,
    },

    "PPO-7": {
        "learning_rate": 4e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 20,
        "gamma": 0.96,
        "gae_lambda": 0.90,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "max_grad_norm": 0.5,
    },

    "PPO-8": {
        "learning_rate": 3e-4,
        "n_steps": 3072,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
    },

    "PPO-9": {
        "learning_rate": 1e-3,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "max_grad_norm": 0.5,
    },

    "PPO-10": {
        "learning_rate": 2e-4,
        "n_steps": 4096,
        "batch_size": 64,
        "n_epochs": 8,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "clip_range": 0.15,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
    }
}

# PPO Training Function
def train_and_evaluate_ppo(config: Dict[str, Any],
                           config_name: str,
                           total_timesteps: int = 150000):

    print(f"\nTraining PPO ({config_name})...")

    log_dir = f"ppo_logs_{config_name}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()
    eval_env = make_env()

    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
        **config
    )

    reward_callback = RewardCallback(check_freq=2000, log_dir=log_dir)
    entropy_callback = EntropyCallback(check_freq=2000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([reward_callback, entropy_callback, eval_callback]),
        tb_log_name=config_name
    )

    model.save(os.path.join(log_dir, "final_model"))

    return {
        "model": model,
        "rewards": reward_callback.rewards,
        "entropies": entropy_callback.entropies,
        "log_dir": log_dir,
        "config_name": config_name,
    }

# Plotting
def plot_individual_training_ppo(result):
    plt.figure(figsize=(12, 5))

    # Rewards
    plt.subplot(1, 2, 1)
    plt.plot(result["rewards"])
    plt.title(f"{result['config_name']} - PPO Mean Rewards")
    plt.grid(True)

    # Entropy
    plt.subplot(1, 2, 2)
    plt.plot(result["entropies"])
    plt.title(f"{result['config_name']} - Policy Entropy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_entropy(results):
    plt.figure(figsize=(10, 5))
    for cfg, res in results.items():
        plt.plot(res["entropies"], label=cfg)
    plt.title("PPO Policy Entropy Progression")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main
def main():
    results = {}

    for cfg_name, cfg in ppo_hyperparams.items():
        results[cfg_name] = train_and_evaluate_ppo(
            cfg,
            cfg_name,
            total_timesteps=750000
        )

        plot_individual_training_ppo(results[cfg_name])

    plot_entropy(results)


if __name__ == "__main__":
    main()
