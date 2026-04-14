import copy
from pathlib import Path

import gymnasium as gym
import numpy as np

from common import dump_json, resolve_device, seed_env, set_seed
from observation_extension_config import build_config
from shared_core_config import SHARED_CORE_ENV_ID

PROJECT_ROOT = Path(__file__).resolve().parent
EXT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "observation_extension"

DEFAULT_EXTENSION_TRAIN_SEEDS = [0, 1, 2]
DEFAULT_EXTENSION_EVAL_SEEDS = list(range(3000, 3050))
SMOKE_EXTENSION_SEED = 123

DEFAULT_EXTENSION_SETTINGS = {
    "gamma": 0.99,
    "learning_rate": 5e-4,
    "buffer_size": 100_000,
    "batch_size": 64,
    "learning_starts": 10_000,
    "train_freq": 4,
    "target_update_interval": 1_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 80_000,
    "checkpoint_interval": 25_000,
    "gradient_clip_norm": 10.0,
    "mlp_hidden_sizes": [256, 256],
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [5, 3],
    "cnn_strides": [2, 1],
    "cnn_head_hidden": 128,
}

SMOKE_EXTENSION_SETTINGS = {
    **DEFAULT_EXTENSION_SETTINGS,
    "buffer_size": 2_048,
    "batch_size": 32,
    "learning_starts": 32,
    "train_freq": 1,
    "target_update_interval": 64,
    "epsilon_decay_steps": 256,
    "checkpoint_interval": 128,
}


def preprocess_observation(observation, observation_mode):
    array = np.asarray(observation, dtype=np.float32)

    if observation_mode == "kinematics":
        return array.reshape(-1)

    if observation_mode == "occupancy_grid":
        if array.ndim == 2:
            return array[np.newaxis, :, :]
        if array.ndim != 3:
            raise ValueError(f"Unexpected occupancy-grid shape: {array.shape}")

        # Convert to channel-first expected by CNN.
        if array.shape[0] <= 8:
            return array
        return np.transpose(array, (2, 0, 1))

    raise ValueError(f"Unknown observation mode: {observation_mode}")


def make_extension_env(observation_mode, render_mode=None):
    import highway_env  # noqa: F401

    config = copy.deepcopy(build_config(observation_mode))
    if render_mode == "rgb_array":
        config["offscreen_rendering"] = True

    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(config)
    return env


def get_extension_run_paths(observation_mode, seed):
    run_dir = EXT_ARTIFACTS_DIR / observation_mode / f"seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "evaluation_dir": evaluation_dir,
        "training_log_path": run_dir / "training_episodes.csv",
        "metadata_path": run_dir / "metadata.json",
        "final_model_path": checkpoints_dir / "final_model.pt",
        "evaluation_episodes_path": evaluation_dir / "episodes.csv",
        "evaluation_summary_path": evaluation_dir / "summary.json",
    }


def ensure_extension_run_dirs(observation_mode, seed):
    paths = get_extension_run_paths(observation_mode, seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
    paths["evaluation_dir"].mkdir(parents=True, exist_ok=True)
    EXT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return paths


def extension_comparison_csv_path():
    return EXT_ARTIFACTS_DIR / "comparison_summary.csv"


def extension_comparison_markdown_path():
    return EXT_ARTIFACTS_DIR / "comparison_summary.md"


def get_observation_shape_and_num_actions(observation_mode):
    env = make_extension_env(observation_mode)
    try:
        observation, _ = seed_env(env, 0)
        processed = preprocess_observation(observation, observation_mode)
        return tuple(processed.shape), int(env.action_space.n)
    finally:
        env.close()
