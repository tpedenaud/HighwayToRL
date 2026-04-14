from copy import deepcopy

from config import SHARED_CORE_CONFIG
from .utils import get_observation_config


MIXED_TRAFFIC_NAMES = ["shared", "dense", "sparse", "narrow", "wide"]
SAFE_PILOT_EVAL_ENVS = ["shared", "dense", "stress"]


def make_config(observation_mode="kinematics", **updates):
    cfg = deepcopy(SHARED_CORE_CONFIG)
    cfg["observation"] = get_observation_config(observation_mode)
    cfg.update(updates)
    return cfg


TRAFFIC_UPDATES = {
    "shared": {},
    "dense": {
        "vehicles_count": 60,
        "vehicles_density": 1.3,
    },
    "sparse": {
        "vehicles_count": 30,
        "vehicles_density": 0.7,
    },
    "narrow": {
        "lanes_count": 3,
        "vehicles_count": 45,
        "vehicles_density": 1.0,
    },
    "wide": {
        "lanes_count": 5,
        "vehicles_count": 55,
        "vehicles_density": 1.0,
    },
    "stress": {
        "lanes_count": 3,
        "vehicles_count": 60,
        "vehicles_density": 1.3,
    },
}


TRAFFIC_CONFIGS = {
    traffic_name: make_config(**traffic_updates)
    for traffic_name, traffic_updates in TRAFFIC_UPDATES.items()
}


SAFE_REWARD_PRESETS = {
    "safe_v1_mild": {
        "env_updates": {
            "collision_reward": -2.0,
            "right_lane_reward": 0.05,
        },
        "wrapper": {
            "crash_penalty": 1.0,
            "lane_change_penalty": 0.02,
            "overspeed_threshold": 29.0,
            "overspeed_penalty": 0.01,
        },
    },
    "safe_v2_balanced": {
        "env_updates": {
            "collision_reward": -2.0,
            "right_lane_reward": 0.05,
        },
        "wrapper": {
            "crash_penalty": 1.5,
            "lane_change_penalty": 0.03,
            "overspeed_threshold": 28.0,
            "overspeed_penalty": 0.03,
        },
    },
    "safe_v3_strong": {
        "env_updates": {
            "collision_reward": -2.5,
            "right_lane_reward": 0.10,
        },
        "wrapper": {
            "crash_penalty": 2.0,
            "lane_change_penalty": 0.05,
            "overspeed_threshold": 27.0,
            "overspeed_penalty": 0.05,
        },
    },
    "safe_v4_crash_only": {
        "env_updates": {
            "collision_reward": -2.5,
            "right_lane_reward": 0.05,
        },
        "wrapper": {
            "crash_penalty": 2.0,
            "lane_change_penalty": 0.0,
            "overspeed_threshold": None,
            "overspeed_penalty": 0.0,
        },
    },
}


def build_eval_config(traffic_name="shared", observation_mode="kinematics"):
    return make_config(observation_mode=observation_mode, **TRAFFIC_UPDATES[traffic_name])


def build_training_config(traffic_name="shared", safe_preset=None, observation_mode="kinematics"):
    cfg = build_eval_config(traffic_name=traffic_name, observation_mode=observation_mode)
    if safe_preset is not None:
        cfg.update(deepcopy(SAFE_REWARD_PRESETS[safe_preset]["env_updates"]))
    return cfg


def get_reward_wrapper_config(safe_preset=None):
    if safe_preset is None:
        return None
    return deepcopy(SAFE_REWARD_PRESETS[safe_preset]["wrapper"])


def build_eval_config_suite(eval_env_names, observation_mode="kinematics"):
    return {
        eval_env: build_eval_config(eval_env, observation_mode=observation_mode)
        for eval_env in eval_env_names
    }


def make_observation_variants():
    return {
        "kinematics_baseline": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": "kinematics",
            "double_dqn": False,
            "prioritized_replay": False,
        },
        "occupancy_grid_baseline": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": "occupancy_grid",
            "double_dqn": False,
            "prioritized_replay": False,
        },
    }


def make_algorithm_variants(observation_mode="kinematics"):
    return {
        f"{observation_mode}_baseline": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": False,
        },
        f"{observation_mode}_double": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": True,
            "prioritized_replay": False,
        },
        f"{observation_mode}_per": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": True,
        },
        f"{observation_mode}_per_double": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": True,
            "prioritized_replay": True,
        },
    }


def make_pilot_variants(observation_mode="kinematics"):
    variants = {
        "baseline_shared": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": False,
        }
    }

    for safe_preset in SAFE_REWARD_PRESETS:
        variants[safe_preset] = {
            "safe_preset": safe_preset,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": False,
        }

    return variants


def make_main_variants(selected_safe_preset, observation_mode="kinematics"):
    return {
        "baseline_shared": {
            "safe_preset": None,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": False,
        },
        "safe_shared": {
            "safe_preset": selected_safe_preset,
            "traffic_names": ["shared"],
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": False,
        },
        "safe_mixed": {
            "safe_preset": selected_safe_preset,
            "traffic_names": MIXED_TRAFFIC_NAMES,
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": False,
        },
        "safe_mixed_double": {
            "safe_preset": selected_safe_preset,
            "traffic_names": MIXED_TRAFFIC_NAMES,
            "observation_mode": observation_mode,
            "double_dqn": True,
            "prioritized_replay": False,
        },
        "safe_mixed_per": {
            "safe_preset": selected_safe_preset,
            "traffic_names": MIXED_TRAFFIC_NAMES,
            "observation_mode": observation_mode,
            "double_dqn": False,
            "prioritized_replay": True,
        },
        "safe_mixed_per_double": {
            "safe_preset": selected_safe_preset,
            "traffic_names": MIXED_TRAFFIC_NAMES,
            "observation_mode": observation_mode,
            "double_dqn": True,
            "prioritized_replay": True,
        },
    }


def make_safety_variants(best_backbone_name, best_backbone_variant, selected_safe_preset):
    shared_backbone = {
        "observation_mode": best_backbone_variant["observation_mode"],
        "double_dqn": best_backbone_variant["double_dqn"],
        "prioritized_replay": best_backbone_variant["prioritized_replay"],
        "source_backbone": best_backbone_name,
    }

    return {
        "bestalgo_shared_baseline_reward": {
            **shared_backbone,
            "safe_preset": None,
            "traffic_names": ["shared"],
        },
        "bestalgo_shared_safe_reward": {
            **shared_backbone,
            "safe_preset": selected_safe_preset,
            "traffic_names": ["shared"],
        },
        "bestalgo_mixed_safe_reward": {
            **shared_backbone,
            "safe_preset": selected_safe_preset,
            "traffic_names": MIXED_TRAFFIC_NAMES,
        },
    }


def make_training_kwargs(variant_name, variants):
    variant = variants[variant_name]
    train_env_configs = [
        build_training_config(
            traffic_name,
            variant["safe_preset"],
            observation_mode=variant["observation_mode"],
        )
        for traffic_name in variant["traffic_names"]
    ]

    kwargs = {
        "observation_mode": variant["observation_mode"],
        "reward_shaping": get_reward_wrapper_config(variant["safe_preset"]),
        "double_dqn": variant["double_dqn"],
        "prioritized_replay": variant["prioritized_replay"],
        "safe_preset": variant["safe_preset"],
        "training_traffic_names": list(variant["traffic_names"]),
        "variant_name": variant_name,
    }

    if len(train_env_configs) == 1:
        kwargs["env_config"] = train_env_configs[0]
    else:
        kwargs["train_env_configs"] = train_env_configs

    return kwargs
