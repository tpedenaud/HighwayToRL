from pathlib import Path

import gymnasium as gym
import highway_env
import imageio.v2 as imageio
import numpy as np
import torch

from config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from .dqn import DQN
from .utils import preprocess_observation


def make_env(render_mode=None):
    return gym.make(
        SHARED_CORE_ENV_ID,
        config=SHARED_CORE_CONFIG,
        render_mode=render_mode,
    )


def load_custom_agent(checkpoint_path, device=None):
    env = make_env(render_mode="rgb_array")

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


def evaluate_custom_agent(agent, seeds):
    returns = []
    lengths = []
    crashes = []

    for seed in seeds:
        env = make_env()
        obs, info = env.reset(seed=seed)
        obs = preprocess_observation(obs)

        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (done or truncated):
            action = agent.act(obs, epsilon=0.0)
            next_obs, reward, done, truncated, info = env.step(action)

            obs = preprocess_observation(next_obs)
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(int(info["crashed"]))

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "crashes": np.array(crashes, dtype=np.int32),
    }


def evaluate_sb3_model(model, seeds):
    returns = []
    lengths = []
    crashes = []

    for seed in seeds:
        env = make_env()
        obs, info = env.reset(seed=seed)

        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(int(info["crashed"]))

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "crashes": np.array(crashes, dtype=np.int32),
    }


def record_custom_rollout(agent, seed, gif_path):
    env = make_env(render_mode="rgb_array")
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


def record_sb3_rollout(model, seed, gif_path):
    env = make_env(render_mode="rgb_array")
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


def find_custom_failure_case(agent, seeds):
    for seed in seeds:
        result = record_custom_rollout(agent, seed, f"artifacts/failures/custom_seed_{seed}.gif")
        if result["crashed"]:
            return result
    return None


def find_sb3_failure_case(model, seeds):
    for seed in seeds:
        result = record_sb3_rollout(model, seed, f"artifacts/failures/sb3_seed_{seed}.gif")
        if result["crashed"]:
            return result
    return None
