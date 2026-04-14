from copy import deepcopy

import numpy as np

from config import SHARED_CORE_CONFIG


OCCUPANCY_GRID_OBSERVATION_CONFIG = {
    "type": "OccupancyGrid",
    "features": ["presence", "vx", "vy"],
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    "grid_step": [5, 5],
    "as_image": False,
    "align_to_vehicle_axes": True,
    "absolute": False,
    "normalize": True,
    "clip": True,
}


OBSERVATION_CONFIGS = {
    "kinematics": deepcopy(SHARED_CORE_CONFIG["observation"]),
    "occupancy_grid": deepcopy(OCCUPANCY_GRID_OBSERVATION_CONFIG),
}


OBSERVATION_TYPE_TO_MODE = {
    "Kinematics": "kinematics",
    "OccupancyGrid": "occupancy_grid",
}


def get_observation_config(observation_mode="kinematics"):
    if observation_mode not in OBSERVATION_CONFIGS:
        raise ValueError(f"Unknown observation mode: {observation_mode}")
    return deepcopy(OBSERVATION_CONFIGS[observation_mode])


def resolve_observation_mode(observation_mode="kinematics", env_config=None):
    if env_config is None:
        return observation_mode

    observation_type = env_config.get("observation", {}).get("type")
    config_mode = OBSERVATION_TYPE_TO_MODE.get(observation_type)

    if config_mode is None:
        raise ValueError(f"Unsupported observation type: {observation_type}")

    if observation_mode is not None and observation_mode != config_mode:
        raise ValueError(
            f"Observation mode '{observation_mode}' does not match config type '{observation_type}'"
        )

    return config_mode


def preprocess_observation(obs, observation_mode="kinematics"):
    array = np.asarray(obs, dtype=np.float32)

    if observation_mode == "kinematics":
        return array.reshape(-1)

    if observation_mode == "occupancy_grid":
        if array.ndim == 3 and array.shape[0] not in (1, 3, 5):
            return np.moveaxis(array, -1, 0)
        return array

    raise ValueError(f"Unknown observation mode: {observation_mode}")
