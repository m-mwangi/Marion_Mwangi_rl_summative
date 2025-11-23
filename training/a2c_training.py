import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from typing import Dict, Any

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# Your FireRescue Environment
from environment.custom_fire_env import FireRescueEnv


# Action Wrapper
class FlattenActionWrapper(gym.ActionWrapper):
    """
    Converts Dict(steering, throttle) → Discrete(n)
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


# Reward Tracking Callback
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
                    print(f"New best A2C model with mean reward {mean_reward:.2f}")
                self.model.save(self.save_path)

        return True


# A2C Hyperparameters
a2c_hyperparams = {
    "A2C-1": {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_rms_prop": True
    },

    "A2C-2": {
        "learning_rate": 3e-4,
        "gamma": 0.98,
        "gae_lambda": 0.90,
        "ent_coef": 0.02,
        "vf_coef": 0.4,
        "max_grad_norm": 0.5,
        "use_rms_prop": True
    },

    "A2C-3": {
        "learning_rate": 1e-3,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "ent_coef": 0.005,
        "vf_coef": 0.6,
        "max_grad_norm": 0.3,
        "use_rms_prop": True
    },

    "A2C-4": {
        "learning_rate": 5e-4,
        "gamma": 0.97,
        "gae_lambda": 0.92,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.7,
        "use_rms_prop": False
    },

    "A2C-5": {
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.02,
        "vf_coef": 0.3,
        "max_grad_norm": 0.5,
        "use_rms_prop": True
    },

    "A2C-6": {
        "learning_rate": 8e-4,
        "gamma": 0.985,
        "gae_lambda": 0.90,
        "ent_coef": 0.01,
        "vf_coef": 0.4,
        "max_grad_norm": 0.4,
        "use_rms_prop": False
    },

    "A2C-7": {
        "learning_rate": 4e-4,
        "gamma": 0.96,
        "gae_lambda": 0.88,
        "ent_coef": 0.03,
        "vf_coef": 0.7,
        "max_grad_norm": 0.5,
        "use_rms_prop": True
    },

    "A2C-8": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "ent_coef": 0.005,
        "vf_coef": 0.8,
        "max_grad_norm": 0.3,
        "use_rms_prop": False
    },

    "A2C-9": {
        "learning_rate": 6e-4,
        "gamma": 0.98,
        "gae_lambda": 0.93,
        "ent_coef": 0.02,
        "vf_coef": 0.4,
        "max_grad_norm": 0.5,
        "use_rms_prop": True
    },

    "A2C-10": {
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.6,
        "max_grad_norm": 0.5,
        "use_rms_prop": False
    },
}

# A2C Training Function
def train_and_evaluate_a2c(config: Dict[str, Any],
                           config_name: str,
                           total_timesteps: int = 600000):

    print(f"\nTraining A2C ({config_name})...")

    log_dir = f"a2c_logs_{config_name}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()
    eval_env = make_env()

    # Neural net architecture
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )

    model = A2C(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
        **config
    )

    reward_callback = RewardCallback(check_freq=2000, log_dir=log_dir)

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
        callback=CallbackList([reward_callback, eval_callback]),
        tb_log_name=config_name
    )

    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

    return {
        "model": model,
        "rewards": reward_callback.rewards,
        "log_dir": log_dir,
        "config_name": config_name
    }


# Plotting
def plot_individual_training_a2c(result):
    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 1, 1)
    plt.plot(result["rewards"])
    plt.xlabel("Training Steps / Check Frequency")
    plt.ylabel("Mean Episode Reward")
    plt.title(f"{result['config_name']} - A2C Rewards")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_rewards(results):
    plt.figure(figsize=(12, 6))
    for cfg, res in results.items():
        plt.plot(res["rewards"], label=cfg)
    plt.xlabel("Training Checkpoints")
    plt.ylabel("Mean Episode Reward")
    plt.title("A2C Reward Progression")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Training Loop
def main():
    results = {}

    for cfg_name, cfg in a2c_hyperparams.items():
        results[cfg_name] = train_and_evaluate_a2c(
            cfg,
            cfg_name,
            total_timesteps=150000
        )
        plot_individual_training_a2c(results[cfg_name])

    plot_rewards(results)


if __name__ == "__main__":
    main()
