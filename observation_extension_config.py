from shared_core_config import SHARED_CORE_CONFIG

OBSERVATION_MODES = {
    "kinematics": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
            "clip": True,
            "see_behind": True,
            "observe_intentions": False,
        }
    },
    "occupancy_grid": {
        "observation": {
            "type": "OccupancyGrid",
            "features": ["presence", "x", "y", "vx", "vy"],
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "as_image": False,
            "align_to_vehicle_axes": True,
            "absolute": False,
            "normalize": True,
            "clip": True,
        }
    },
}


def build_config(observation_mode):
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"Unknown observation mode: {observation_mode}")

    config = dict(SHARED_CORE_CONFIG)
    config["observation"] = OBSERVATION_MODES[observation_mode]["observation"]
    return config
