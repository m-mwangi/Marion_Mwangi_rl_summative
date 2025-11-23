import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Optional, Dict


class FireRescueEnv(gym.Env):
    """Custom Firefighting Robot Environment"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size=(10, 10), render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        # Map settings
        self.grid_size = grid_size
        self.time_limit = 900

        # Robot physics
        self.wheel_angle = 0.0
        self.max_wheel_angle = math.radians(35)
        self.robot_length = 2.0
        self.drift_factor = 0.94

        self.current_speed = 4.0
        self.min_speed = 0.8
        self.max_speed = 7.0
        self.acceleration = 0.25
        self.braking = 0.28

        # Robot collision box
        self.robot_body_length = 0.8
        self.robot_body_width = 0.4

        # Bounds
        self.min_x = self.robot_body_width / 2
        self.max_x = grid_size[0] - self.robot_body_width / 2
        self.min_y = self.robot_body_length / 2
        self.max_y = grid_size[1] - self.robot_body_length / 2

        # Fire mission settings
        self.fires_extinguished = 0
        self.total_fires = 3
        self.current_fire_id = 0
        self.water_level = 100.0
        self.max_water = 100.0
        self.water_usage_rate = 30.0
        self.mission_complete = False

        self._setup_fixed_elements()

        # Action space
        self.action_space = spaces.Dict({
            "steering": spaces.Discrete(5),
            "throttle": spaces.Discrete(3)
        })

        # Observation space
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            "direction": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "destination": spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            "water_level": spaces.Box(low=0, high=self.max_water, shape=(1,), dtype=np.float32),
            "time_elapsed": spaces.Box(low=0, high=self.time_limit, shape=(1,), dtype=np.float32),
            "obstacles": spaces.Box(low=0, high=max(grid_size), shape=(5, 2), dtype=np.float32),
            "wheel_angle": spaces.Box(low=-self.max_wheel_angle, high=self.max_wheel_angle, shape=(1,), dtype=np.float32),
            "fires_remaining": spaces.Discrete(self.total_fires + 1),
        })

    def _setup_fixed_elements(self):
        self.fire_locations = [(3.0, 7.0), (6.5, 4.5), (8.0, 2.0)]
        self.water_stations = [(1.5, 1.5), (8.5, 8.5)]
        self.robot_start = (1.0, 1.0)

        self.debris_fields = [
            (2.0, 6.0), (2.0, 7.0),
            (5.0, 3.0), (7.0, 3.0),
            (4.5, 5.5), (5.0, 7.5)
        ]

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self.fires_extinguished = 0
        self.current_fire_id = 0
        self.time_elapsed = 0
        self.water_level = self.max_water

        self.robot_pos = np.array(self.robot_start, dtype=np.float32)
        self.robot_dir = np.array([0.0, 1.0])
        self.wheel_angle = 0.0
        self.current_speed = 4.0

        self.active_fires = self.fire_locations.copy()
        self.obstacles = [np.array(p, dtype=np.float32) for p in self.debris_fields]
        self.water_sources = self.water_stations.copy()

        self.target_location = self.active_fires[self.current_fire_id]
        self.last_distance = self._distance(self.robot_pos, self.target_location)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action: Dict):

        # Apply action
        self._apply_action(action)
        self.time_elapsed += 1

        reward = 0.0
        terminated = False
        truncated = False
        status = ""

        # 1. Direction alignment
        dir_to_target = self.target_location - self.robot_pos
        if np.linalg.norm(dir_to_target) > 0.1:
            dir_to_target /= np.linalg.norm(dir_to_target)
            alignment = np.dot(self.robot_dir, dir_to_target)
            reward += 2.2 * max(0, alignment)

        # Fire extinguish
        if self._at_location(self.robot_pos, [self.target_location]):

            if self.water_level >= self.water_usage_rate:
                self.water_level -= self.water_usage_rate
                self.fires_extinguished += 1

                reward += 650 + (200 * self.fires_extinguished)
                status = f"Fire {self.fires_extinguished} extinguished!"

                extinguished_fire = self.target_location
                self.active_fires.remove(extinguished_fire)

                # Next fire selection
                if self.active_fires:
                    self.target_location = self._nearest_fire()
                else:
                    terminated = True
                    self.mission_complete = True
                    reward += 900 + (self.water_level * 1.5)
                    status = "MISSION COMPLETE"

                if self.active_fires:
                    self.last_distance = self._distance(self.robot_pos, self.target_location)

            else:
                reward -= 6
                status = "NO WATER"

        # Water refill
        elif self._at_location(self.robot_pos, self.water_sources):
            refill_reward = 250 * (1 - self.water_level / self.max_water)
            reward += refill_reward
            self.water_level = self.max_water
            status = "REFILLED"

        # Distance improvement
        dist = self._distance(self.robot_pos, self.target_location)
        improvement = self.last_distance - dist
        if improvement > 0:
            reward += 14 * improvement
        self.last_distance = dist

        # Collision
        if self._collision():
            reward -= 150
            terminated = True
            status = "COLLISION"

        # Time penalty
        reward -= 0.05

        # Timeout
        if self.time_elapsed >= self.time_limit:
            truncated = True
            reward += 120 * self.fires_extinguished
            status = "TIMEOUT"

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"status": status}

    def _apply_action(self, action):
        steering = int(action["steering"])
        throttle = int(action["throttle"])

        steering_map = {0: 0, 1: 15, 2: 30, 3: -15, 4: -30}
        angle_step = math.radians(steering_map[steering])

        if angle_step != 0:
            self.wheel_angle = np.clip(
                self.wheel_angle + angle_step,
                -self.max_wheel_angle,
                self.max_wheel_angle
            )
        else:
            self.wheel_angle *= self.drift_factor

        if throttle == 0:
            self.current_speed = max(self.min_speed, self.current_speed - self.braking)
        elif throttle == 2:
            self.current_speed = min(self.max_speed, self.current_speed + self.acceleration)

        # Turning update
        if abs(self.wheel_angle) > 0.01:
            turn_radius = self.robot_length / math.tan(self.wheel_angle)
            ang_vel = (self.current_speed * 0.05) / max(0.1, abs(turn_radius))
            ang_vel *= -1 if self.wheel_angle < 0 else 1

            cur_angle = math.atan2(self.robot_dir[1], self.robot_dir[0])
            new_angle = cur_angle + ang_vel

            self.robot_dir = np.array([math.cos(new_angle), math.sin(new_angle)])
            self.robot_dir /= np.linalg.norm(self.robot_dir)

        new_pos = self.robot_pos + self.robot_dir * self.current_speed * 0.05
        if self._valid_pos(new_pos):
            self.robot_pos = new_pos

    def _get_obs(self):
        return {
            "position": np.array(self.robot_pos, dtype=np.float32),
            "direction": np.array(self.robot_dir, dtype=np.float32),
            "destination": np.array(self.target_location, dtype=np.float32),
            "water_level": np.array([self.water_level], dtype=np.float32),
            "time_elapsed": np.array([self.time_elapsed], dtype=np.float32),
            "obstacles": np.array(self.obstacles[:5], dtype=np.float32),
            "wheel_angle": np.array([self.wheel_angle], dtype=np.float32),

            "fires_remaining": np.array(
                [len(self.active_fires)], dtype=np.int32
            ),
        }

    def _valid_pos(self, pos):
        return (
            self.min_x <= pos[0] <= self.max_x and
            self.min_y <= pos[1] <= self.max_y
        )

    def _collision(self):
        return any(self._distance(self.robot_pos, obs) < 0.7 for obs in self.obstacles)

    def _at_location(self, pos, locs):
        return any(self._distance(pos, loc) < 0.5 for loc in locs)

    def _distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _nearest_fire(self):
        dists = [self._distance(self.robot_pos, f) for f in self.active_fires]
        return self.active_fires[np.argmin(dists)]

    def render(self):
        if self.render_mode == "human":
            from environment.rendering import FireRescueVisualizer
            if self.viewer is None:
                self.viewer = FireRescueVisualizer(self)
            return self.viewer.render()
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
