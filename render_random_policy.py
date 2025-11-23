import gymnasium as gym
import numpy as np
import time
from environment.custom_fire_env import FireRescueEnv

# ---------------------------------------------------------
#  SAME ACTION WRAPPER AS TRAINING
# ---------------------------------------------------------
class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steering_actions = 5
        self.throttle_actions = 3
        self.action_space = gym.spaces.Discrete(
            self.steering_actions * self.throttle_actions
        )

    def action(self, act):
        # Convert INT -> Dict
        steering = act // self.throttle_actions
        throttle = act % self.throttle_actions
        return {"steering": steering, "throttle": throttle}

def make_env():
    env = FireRescueEnv(grid_size=(10, 10), render_mode="human")
    env = FlattenActionWrapper(env)
    return env

# ---------------------------------------------------------
#  RANDOM AGENT DEMO
# ---------------------------------------------------------
def run_random_demo(steps=300):
    print("\n🔥 STATIC DEMONSTRATION: RANDOM AGENT")
    print("Agent is taking random actions (NO RL model)")
    print("Showing environment functionality & OpenGL components\n")

    env = make_env()
    obs, _ = env.reset()

    for t in range(steps):
        action = env.action_space.sample()     # <--- INT, not dict
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.05)

        if terminated or truncated:
            print("\n⚠ Episode ended early:", info.get("status", ""))
            time.sleep(1)
            break

    env.close()


if __name__ == "__main__":
    run_random_demo()
