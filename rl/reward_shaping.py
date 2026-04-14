import gymnasium as gym


class SafeRewardWrapper(gym.Wrapper):
    def __init__(self, env, shaping_config):
        super().__init__(env)
        self.shaping_config = {
            "crash_penalty": 0.0,
            "lane_change_penalty": 0.0,
            "overspeed_threshold": None,
            "overspeed_penalty": 0.0,
            **(shaping_config or {}),
        }
        self._last_lane_id = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_lane_id = self._get_lane_id()

        info = dict(info)
        info["shaping"] = {
            "base_reward": 0.0,
            "shaped_reward": 0.0,
            "crash_penalty": 0.0,
            "lane_change_penalty": 0.0,
            "overspeed_penalty": 0.0,
            "overspeed_fraction": 0.0,
            "lane_changed": False,
        }
        return obs, info

    def step(self, action):
        previous_lane_id = self._last_lane_id
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lane_id = self._get_lane_id()
        lane_changed = (
            previous_lane_id is not None
            and current_lane_id is not None
            and current_lane_id != previous_lane_id
        )

        crash_penalty = self.shaping_config["crash_penalty"] if info.get("crashed", False) else 0.0
        lane_change_penalty = (
            self.shaping_config["lane_change_penalty"] if lane_changed else 0.0
        )

        speed = float(info.get("speed", 0.0))
        overspeed_fraction = 0.0
        overspeed_threshold = self.shaping_config["overspeed_threshold"]
        if overspeed_threshold is not None:
            reward_speed_range = self.unwrapped.config.get("reward_speed_range", [overspeed_threshold, overspeed_threshold + 1.0])
            max_speed = reward_speed_range[1]
            denominator = max(1.0, float(max_speed) - float(overspeed_threshold))
            overspeed_fraction = max(0.0, speed - float(overspeed_threshold)) / denominator

        overspeed_penalty = self.shaping_config["overspeed_penalty"] * overspeed_fraction
        shaped_reward = reward - crash_penalty - lane_change_penalty - overspeed_penalty

        info = dict(info)
        info["shaping"] = {
            "base_reward": float(reward),
            "shaped_reward": float(shaped_reward),
            "crash_penalty": float(crash_penalty),
            "lane_change_penalty": float(lane_change_penalty),
            "overspeed_penalty": float(overspeed_penalty),
            "overspeed_fraction": float(overspeed_fraction),
            "lane_changed": bool(lane_changed),
        }

        self._last_lane_id = current_lane_id
        return obs, shaped_reward, terminated, truncated, info

    def _get_lane_id(self):
        vehicle = getattr(self.unwrapped, "vehicle", None)
        if vehicle is None:
            return None

        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            return None

        if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
            return lane_index[2]

        return lane_index
