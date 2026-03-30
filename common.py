import copy
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import ObservationWrapper, spaces

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

CUSTOM_MODEL_NAME = "custom_dqn"
SB3_MODEL_NAME = "sb3_dqn"

DEFAULT_TRAIN_SEEDS = [0, 1, 2]
PILOT_SEED = 42
SMOKE_SEED = 123
DEFAULT_EVAL_SEEDS = list(range(1000, 1050))

ACTION_LABELS = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

DEFAULT_DQN_SETTINGS = {
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
    "hidden_sizes": [256, 256],
}

SMOKE_DQN_SETTINGS = {
    **DEFAULT_DQN_SETTINGS,
    "buffer_size": 2_048,
    "batch_size": 32,
    "learning_starts": 32,
    "train_freq": 1,
    "target_update_interval": 64,
    "epsilon_decay_steps": 256,
    "checkpoint_interval": 128,
}


os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sample_observation, _ = env.reset()
        flattened = self._flatten(sample_observation)
        bounds = np.full(flattened.shape, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-bounds, high=bounds, dtype=np.float32)

    @staticmethod
    def _flatten(observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def observation(self, observation):
        return self._flatten(observation)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device):
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def make_env(render_mode=None, monitor_path=None):
    import highway_env  # noqa: F401

    config = copy.deepcopy(SHARED_CORE_CONFIG)
    if render_mode == "rgb_array":
        config["offscreen_rendering"] = True

    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(config)
    env = FlattenObservation(env)

    if monitor_path is not None:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env, filename=str(monitor_path), info_keywords=("crashed", "speed"))

    return env


def seed_env(env, seed):
    env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env.reset(seed=seed)


def get_observation_dim_and_num_actions():
    env = make_env()
    try:
        observation, _ = seed_env(env, 0)
        return int(observation.shape[0]), int(env.action_space.n)
    finally:
        env.close()


def get_run_paths(model_name, seed):
    run_dir = ARTIFACTS_DIR / model_name / f"seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    final_name = "final_model.pt" if model_name == CUSTOM_MODEL_NAME else "final_model.zip"
    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "evaluation_dir": evaluation_dir,
        "training_log_path": run_dir / "training_episodes.csv",
        "metadata_path": run_dir / "metadata.json",
        "final_model_path": checkpoints_dir / final_name,
        "monitor_path": run_dir / "sb3_training.monitor.csv",
        "evaluation_episodes_path": evaluation_dir / "episodes.csv",
        "evaluation_summary_path": evaluation_dir / "summary.json",
    }


def ensure_run_dirs(model_name, seed):
    paths = get_run_paths(model_name, seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
    paths["evaluation_dir"].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    return paths


def comparison_csv_path():
    return EVALUATION_DIR / "comparison_summary.csv"


def comparison_markdown_path():
    return EVALUATION_DIR / "comparison_summary.md"


def training_curve_path():
    return REPORTS_DIR / "training_curves.png"


def success_contact_sheet_path():
    return REPORTS_DIR / "success_contact_sheet.png"


def failure_contact_sheet_path():
    return REPORTS_DIR / "failure_contact_sheet.png"


def behavior_analysis_path():
    return REPORTS_DIR / "behavior_analysis.md"


def core_summary_path():
    return REPORTS_DIR / "core_task_summary.md"
