from pathlib import Path

import gymnasium as gym
import highway_env
import imageio.v2 as imageio
import numpy as np
import torch

from config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from .dqn import DQN
from .utils import preprocess_observation


def make_env(render_mode=None, env_config=None):
    if env_config is None:
        env_config = SHARED_CORE_CONFIG

    return gym.make(
        SHARED_CORE_ENV_ID,
        config=env_config,
        render_mode=render_mode,
    )



def load_custom_agent(checkpoint_path, device=None, env_config=None):
    env = make_env(render_mode="rgb_array", env_config=env_config)

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    agent = DQN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        buffer_capacity=10000,
        batch_size=32,
        learning_rate=0.001,
        gamma=0.99,
        target_update_freq=1000,
        epsilon=1.0,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.net.load_state_dict(checkpoint["net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    env.close()
    return agent


def evaluate_custom_agent(agent, seeds, env_config=None):
    returns = []
    lengths = []
    crashes = []
    mean_speeds = []
    lane_change_rates = []

    for seed in seeds:
        env = make_env(env_config=env_config)
        lane_change_actions = get_lane_change_actions(env)

        obs, info = env.reset(seed=seed)
        obs = preprocess_observation(obs)

        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        speed_sum = 0.0
        lane_change_count = 0

        while not (done or truncated):
            action = agent.act(obs, epsilon=0.0)

            next_obs, reward, done, truncated, info = env.step(action)
            obs = preprocess_observation(next_obs)

            episode_return += reward
            episode_length += 1
            speed_sum += float(info.get("speed", 0.0))

            if action in lane_change_actions:
                lane_change_count += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(int(info["crashed"]))
        mean_speeds.append(speed_sum / episode_length if episode_length > 0 else 0.0)
        lane_change_rates.append(lane_change_count / episode_length if episode_length > 0 else 0.0)

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(mean_speeds)),
        "lane_change_rate": float(np.mean(lane_change_rates)),
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "crashes": np.array(crashes, dtype=np.int32),
        "mean_speeds": np.array(mean_speeds, dtype=np.float32),
        "lane_change_rates": np.array(lane_change_rates, dtype=np.float32),
    }



def evaluate_sb3_model(model, seeds, env_config=None):
    returns = []
    lengths = []
    crashes = []
    mean_speeds = []
    lane_change_rates = []

    for seed in seeds:
        env = make_env(env_config=env_config)
        lane_change_actions = get_lane_change_actions(env)

        obs, info = env.reset(seed=seed)

        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        speed_sum = 0.0
        lane_change_count = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, truncated, info = env.step(action)

            episode_return += reward
            episode_length += 1
            speed_sum += float(info.get("speed", 0.0))

            if action in lane_change_actions:
                lane_change_count += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(int(info["crashed"]))
        mean_speeds.append(speed_sum / episode_length if episode_length > 0 else 0.0)
        lane_change_rates.append(lane_change_count / episode_length if episode_length > 0 else 0.0)

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(mean_speeds)),
        "lane_change_rate": float(np.mean(lane_change_rates)),
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "crashes": np.array(crashes, dtype=np.int32),
        "mean_speeds": np.array(mean_speeds, dtype=np.float32),
        "lane_change_rates": np.array(lane_change_rates, dtype=np.float32),
    }



def record_custom_rollout(agent, seed, gif_path, env_config=None):
    env = make_env(render_mode="rgb_array", env_config=env_config)
    obs, info = env.reset(seed=seed)
    obs = preprocess_observation(obs)

    frames = [env.render()]
    done = False
    truncated = False
    episode_return = 0.0

    while not (done or truncated):
        action = agent.act(obs, epsilon=0.0)
        next_obs, reward, done, truncated, info = env.step(action)

        obs = preprocess_observation(next_obs)
        episode_return += reward
        frames.append(env.render())

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=5)

    env.close()

    return {
        "gif_path": str(gif_path),
        "return": float(episode_return),
        "crashed": bool(info["crashed"]),
        "seed": seed,
    }


def record_sb3_rollout(model, seed, gif_path, env_config=None):
    env = make_env(render_mode="rgb_array", env_config=env_config)
    obs, info = env.reset(seed=seed)

    frames = [env.render()]
    done = False
    truncated = False
    episode_return = 0.0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        episode_return += reward
        frames.append(env.render())

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=5)

    env.close()

    return {
        "gif_path": str(gif_path),
        "return": float(episode_return),
        "crashed": bool(info["crashed"]),
        "seed": seed,
    }


def find_custom_failure_case(agent, seeds, env_config=None):
    for seed in seeds:
        result = record_custom_rollout(
            agent,
            seed,
            f"artifacts/failures/custom_seed_{seed}.gif",
            env_config=env_config,
        )
        if result["crashed"]:
            return result
    return None


def find_sb3_failure_case(model, seeds, env_config=None):
    for seed in seeds:
        result = record_sb3_rollout(
            model,
            seed,
            f"artifacts/failures/sb3_seed_{seed}.gif",
            env_config=env_config,
        )
        if result["crashed"]:
            return result
    return None

def get_lane_change_actions(env):
    action_indexes = env.unwrapped.action_type.actions_indexes
    lane_change_actions = set()

    if "LANE_LEFT" in action_indexes:
        lane_change_actions.add(action_indexes["LANE_LEFT"])
    if "LANE_RIGHT" in action_indexes:
        lane_change_actions.add(action_indexes["LANE_RIGHT"])

    return lane_change_actions
