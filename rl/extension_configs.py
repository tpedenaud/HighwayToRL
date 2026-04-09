from copy import deepcopy

from config import SHARED_CORE_CONFIG


def make_config(**updates):
    cfg = deepcopy(SHARED_CORE_CONFIG)
    cfg.update(updates)
    return cfg

def build_config(traffic_name="shared", reward_name="baseline"):
    cfg = deepcopy(TRAFFIC_CONFIGS[traffic_name])
    cfg.update(REWARD_VARIANTS[reward_name])
    return cfg


TRAFFIC_CONFIGS = {
    "shared": make_config(),
    "dense": make_config(
        vehicles_count=60,
        vehicles_density=1.3,
    ),
    "sparse": make_config(
        vehicles_count=30,
        vehicles_density=0.7,
    ),
    "narrow": make_config(
        lanes_count=3,
        vehicles_count=45,
        vehicles_density=1.0,
    ),
    "wide": make_config(
        lanes_count=5,
        vehicles_count=55,
        vehicles_density=1.0,
    ),
    "stress": make_config(
        lanes_count=3,
        vehicles_count=60,
        vehicles_density=1.3,
    ),
}

REWARD_VARIANTS = {
    "baseline": {},
    "safe": {
        "collision_reward": -3.0,
        "lane_change_reward": -0.05,
        "right_lane_reward": 0.05,
        "high_speed_reward": 0.7,
    },
}

TRAIN_VARIANTS = {
    "baseline_shared": {
        "reward_name": "baseline",
        "traffic_names": ["shared"],
    },
    "safe_shared": {
        "reward_name": "safe",
        "traffic_names": ["shared"],
    },
    "safe_mixed": {
        "reward_name": "safe",
        "traffic_names": ["shared", "dense", "sparse", "narrow", "wide"],
    },
}

