from pathlib import Path
from copy import deepcopy

import gymnasium as gym
import highway_env
import imageio.v2 as imageio
import numpy as np
import torch

from config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from .dqn import DQN
from .reward_shaping import SafeRewardWrapper
from .utils import get_observation_config, preprocess_observation, resolve_observation_mode


def make_env(render_mode=None, env_config=None, reward_shaping=None, observation_mode="kinematics"):
    if env_config is None:
        env_config = deepcopy(SHARED_CORE_CONFIG)
        env_config["observation"] = get_observation_config(observation_mode)
    else:
        env_config = deepcopy(env_config)

    resolve_observation_mode(observation_mode, env_config)

    env = gym.make(
        SHARED_CORE_ENV_ID,
        config=env_config,
        render_mode=render_mode,
    )

    if reward_shaping is not None:
        env = SafeRewardWrapper(env, reward_shaping)

    return env


def load_custom_agent(checkpoint_path, device=None, env_config=None, observation_mode=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_config = checkpoint.get("agent_config", {})
    checkpoint_observation_mode = agent_config.get("observation_mode", "kinematics")

    if env_config is None:
        env_config = deepcopy(SHARED_CORE_CONFIG)
    else:
        env_config = deepcopy(env_config)

    if "observation_config" in agent_config:
        env_config["observation"] = deepcopy(agent_config["observation_config"])

    resolved_observation_mode = resolve_observation_mode(
        checkpoint_observation_mode if observation_mode is None else observation_mode,
        env_config,
    )
    env = make_env(env_config=env_config, observation_mode=resolved_observation_mode)

    agent = DQN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        buffer_capacity=agent_config.get("buffer_capacity", 10000),
        batch_size=agent_config.get("batch_size", 32),
        learning_rate=agent_config.get("learning_rate", 0.001),
        gamma=agent_config.get("gamma", 0.99),
        target_update_freq=agent_config.get("target_update_freq", 1000),
        epsilon=agent_config.get("epsilon", 1.0),
        device=device,
        observation_mode=resolved_observation_mode,
        double_dqn=agent_config.get("double_dqn", False),
        prioritized_replay=agent_config.get("prioritized_replay", False),
        priority_alpha=agent_config.get("priority_alpha", 0.6),
        priority_eps=agent_config.get("priority_eps", 1e-5),
    )

    agent.net.load_state_dict(checkpoint["net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    env.close()
    return agent


def get_current_lane_id(env):
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return None

    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        return None

    if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
        return lane_index[2]

    return lane_index


def evaluate_custom_agent(agent, seeds, env_config=None, reward_shaping=None):
    returns = []
    lengths = []
    crashes = []
    mean_speeds = []
    actual_lane_change_rates = []

    for seed in seeds:
        resolved_observation_mode = (
            resolve_observation_mode(agent.observation_mode, env_config)
            if env_config is not None
            else agent.observation_mode
        )
        env = make_env(
            env_config=env_config,
            reward_shaping=reward_shaping,
            observation_mode=resolved_observation_mode,
        )

        obs, info = env.reset(seed=seed)
        obs = preprocess_observation(obs, resolved_observation_mode)

        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        speed_sum = 0.0
        lane_change_count = 0
        previous_lane_id = get_current_lane_id(env)

        while not (done or truncated):
            action = agent.act(obs, epsilon=0.0)

            next_obs, reward, done, truncated, info = env.step(action)
            obs = preprocess_observation(next_obs, resolved_observation_mode)

            current_lane_id = get_current_lane_id(env)
            if (
                previous_lane_id is not None
                and current_lane_id is not None
                and current_lane_id != previous_lane_id
            ):
                lane_change_count += 1

            previous_lane_id = current_lane_id
            episode_return += reward
            episode_length += 1
            speed_sum += float(info.get("speed", 0.0))

        actual_lane_change_rates.append(
            lane_change_count / episode_length if episode_length > 0 else 0.0
        )
        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(int(info["crashed"]))
        mean_speeds.append(speed_sum / episode_length if episode_length > 0 else 0.0)

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(mean_speeds)),
        "actual_lane_change_rate": float(np.mean(actual_lane_change_rates)),
        "lane_change_rate": float(np.mean(actual_lane_change_rates)),
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "crashes": np.array(crashes, dtype=np.int32),
        "mean_speeds": np.array(mean_speeds, dtype=np.float32),
        "actual_lane_change_rates": np.array(actual_lane_change_rates, dtype=np.float32),
        "lane_change_rates": np.array(actual_lane_change_rates, dtype=np.float32),
    }


def evaluate_sb3_model(model, seeds, env_config=None, reward_shaping=None, observation_mode="kinematics"):
    returns = []
    lengths = []
    crashes = []
    mean_speeds = []
    actual_lane_change_rates = []

    for seed in seeds:
        env = make_env(
            env_config=env_config,
            reward_shaping=reward_shaping,
            observation_mode=observation_mode,
        )

        obs, info = env.reset(seed=seed)

        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        speed_sum = 0.0
        lane_change_count = 0
        previous_lane_id = get_current_lane_id(env)

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, truncated, info = env.step(action)

            current_lane_id = get_current_lane_id(env)
            if (
                previous_lane_id is not None
                and current_lane_id is not None
                and current_lane_id != previous_lane_id
            ):
                lane_change_count += 1

            previous_lane_id = current_lane_id
            episode_return += reward
            episode_length += 1
            speed_sum += float(info.get("speed", 0.0))

        actual_lane_change_rates.append(
            lane_change_count / episode_length if episode_length > 0 else 0.0
        )
        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(int(info["crashed"]))
        mean_speeds.append(speed_sum / episode_length if episode_length > 0 else 0.0)

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(mean_speeds)),
        "actual_lane_change_rate": float(np.mean(actual_lane_change_rates)),
        "lane_change_rate": float(np.mean(actual_lane_change_rates)),
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "crashes": np.array(crashes, dtype=np.int32),
        "mean_speeds": np.array(mean_speeds, dtype=np.float32),
        "actual_lane_change_rates": np.array(actual_lane_change_rates, dtype=np.float32),
        "lane_change_rates": np.array(actual_lane_change_rates, dtype=np.float32),
    }


def evaluate_custom_agent_suite(
    agent,
    seeds,
    eval_configs,
    variant_name=None,
    train_seed=None,
    reward_shaping=None,
):
    rows = []

    for eval_env, env_config in eval_configs.items():
        result = evaluate_custom_agent(
            agent,
            seeds=seeds,
            env_config=env_config,
            reward_shaping=reward_shaping,
        )
        rows.append(
            {
                "variant": variant_name,
                "train_seed": train_seed,
                "eval_env": eval_env,
                "mean_return": result["mean_return"],
                "std_return": result["std_return"],
                "mean_length": result["mean_length"],
                "crash_rate": result["crash_rate"],
                "mean_speed": result["mean_speed"],
                "actual_lane_change_rate": result["actual_lane_change_rate"],
            }
        )

    return rows


def record_custom_rollout(agent, seed, gif_path, env_config=None, reward_shaping=None):
    resolved_observation_mode = (
        resolve_observation_mode(agent.observation_mode, env_config)
        if env_config is not None
        else agent.observation_mode
    )
    env = make_env(
        render_mode="rgb_array",
        env_config=env_config,
        reward_shaping=reward_shaping,
        observation_mode=agent.observation_mode,
    )
    obs, info = env.reset(seed=seed)
    obs = preprocess_observation(obs, resolved_observation_mode)

    frames = [env.render()]
    done = False
    truncated = False
    episode_return = 0.0

    while not (done or truncated):
        action = agent.act(obs, epsilon=0.0)
        next_obs, reward, done, truncated, info = env.step(action)

        obs = preprocess_observation(next_obs, resolved_observation_mode)
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


def record_sb3_rollout(
    model,
    seed,
    gif_path,
    env_config=None,
    reward_shaping=None,
    observation_mode="kinematics",
):
    env = make_env(
        render_mode="rgb_array",
        env_config=env_config,
        reward_shaping=reward_shaping,
        observation_mode=observation_mode,
    )
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


def find_custom_failure_case(agent, seeds, env_config=None, reward_shaping=None):
    for seed in seeds:
        result = record_custom_rollout(
            agent,
            seed,
            f"artifacts/failures/custom_seed_{seed}.gif",
            env_config=env_config,
            reward_shaping=reward_shaping,
        )
        if result["crashed"]:
            return result
    return None


def find_sb3_failure_case(model, seeds, env_config=None, reward_shaping=None):
    for seed in seeds:
        result = record_sb3_rollout(
            model,
            seed,
            f"artifacts/failures/sb3_seed_{seed}.gif",
            env_config=env_config,
            reward_shaping=reward_shaping,
        )
        if result["crashed"]:
            return result
    return None
