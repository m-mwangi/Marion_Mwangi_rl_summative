import time
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Tuple
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_fire_env import FireRescueEnv


# ACTION MEANINGS
ACTION_MEANINGS = {
    "steering": {
        0: "Straight",
        1: "Soft Left",
        2: "Hard Left",
        3: "Soft Right",
        4: "Hard Right"
    },
    "throttle": {
        0: "Brake",
        1: "Maintain",
        2: "Accelerate"
    }
}

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steering_actions = 5
        self.throttle_actions = 3
        self.action_space = gym.spaces.Discrete(self.steering_actions * self.throttle_actions)

    def action(self, act):
        act = int(act)
        steering = act // self.throttle_actions
        throttle = act % self.throttle_actions
        return {"steering": int(steering), "throttle": int(throttle)}


# ACTION CONVERSION
def action_idx_to_dict(action_idx: int):
    if isinstance(action_idx, np.ndarray):
        action_idx = action_idx.item()

    steering = action_idx // 3
    throttle = action_idx % 3

    action_dict = {"steering": steering, "throttle": throttle}
    action_str = f"{ACTION_MEANINGS['steering'][steering]} + {ACTION_MEANINGS['throttle'][throttle]}"
    return action_dict, action_str


# LOAD REINFORCE POLICY
def load_reinforce_policy(model_path: str, input_dim: int, n_actions: int):
    class ReinforcePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, n_actions)
            )

        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    policy = ReinforcePolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    return policy


def obs_to_tensor(obs):
    flat = []
    for v in obs.values():
        flat.extend(np.array(v, dtype=np.float32).flatten())
    return torch.tensor(flat, dtype=torch.float32)

# VISUALIZATION
def visualize_model(model_path: str, algorithm: str = "DQN", num_episodes: int = 3):

    env = FireRescueEnv(grid_size=(10, 10), render_mode="human")
    env = FlattenActionWrapper(env)

    print("\n=== Loading Trained Model ===")

    is_reinforce = algorithm.upper() == "REINFORCE"

    if is_reinforce:
        obs, _ = env.reset()
        input_dim = len(obs_to_tensor(obs))
        n_actions = env.action_space.n
        policy = load_reinforce_policy(model_path, input_dim, n_actions)
        print("Loaded REINFORCE policy.")
    else:
        if algorithm.upper() == "DQN":
            model = DQN.load(model_path)
        elif algorithm.upper() == "PPO":
            model = PPO.load(model_path)
        elif algorithm.upper() == "A2C":
            model = A2C.load(model_path)
        else:
            raise ValueError("Unknown algorithm")
        print(f"Loaded {algorithm} model.")

    # RUN EPISODES
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\nEpisode {ep + 1}")
        print(f"Initial target: {env.unwrapped.target_location}")
        print(f"Water: {env.unwrapped.water_level:.1f}")

        while not done:

            if is_reinforce:
                obs_t = obs_to_tensor(obs)
                with torch.no_grad():
                    probs = policy(obs_t)
                action_idx = torch.argmax(probs).item()
            else:
                action_idx, _ = model.predict(obs, deterministic=True)

            action_dict, action_str = action_idx_to_dict(action_idx)

            next_obs, reward, terminated, truncated, info = env.step(action_idx)

            done = terminated or truncated
            steps += 1
            total_reward += reward

            env.render()

            print(
                f"Step {steps:03d} | {action_str:20s} | "
                f"Pos: {next_obs['position']} | "
                f"Water: {env.unwrapped.water_level:.1f} | "
                f"Reward: {reward:+6.2f} (Total {total_reward:.2f}) | "
                f"Status: {info.get('status', '')}"
            )

            obs = next_obs
            time.sleep(0.05)

        true_extinguished = env.unwrapped.fires_extinguished

        print(f"\nEpisode finished in {steps} steps")
        print(f"Total reward: {total_reward:.1f}")
        print(f"Fires extinguished: {true_extinguished}/{env.unwrapped.total_fires}")

        # Mission success logic
        if env.unwrapped.mission_complete and true_extinguished == env.unwrapped.total_fires:
            print("MISSION SUCCESS! All fires extinguished!")
        else:
            print("Mission ended - not all fires extinguished.")

    env.close()

if __name__ == "__main__":
    visualize_model(
        model_path="logs_DQN-6/best_model.zip",
        algorithm="DQN",
        num_episodes=3
    )
