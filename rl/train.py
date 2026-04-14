import json
import random
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import torch

from config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from .dqn import DQN
from .reward_shaping import SafeRewardWrapper
from .utils import get_observation_config, preprocess_observation, resolve_observation_mode


def train(
    seed=0,
    run_dir=None,
    env_config=None,
    train_env_configs=None,
    observation_mode="kinematics",
    reward_shaping=None,
    double_dqn=False,
    prioritized_replay=False,
    total_steps=20000,
    learning_starts=1000,
    buffer_capacity=10000,
    batch_size=32,
    learning_rate=0.001,
    gamma=0.99,
    target_update_freq=1000,
    priority_alpha=0.6,
    beta_start=0.4,
    beta_final=1.0,
    priority_eps=1e-5,
    safe_preset=None,
    training_traffic_names=None,
    variant_name=None,
    verbose=True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if train_env_configs is None:
        if env_config is None:
            default_config = deepcopy(SHARED_CORE_CONFIG)
            default_config["observation"] = get_observation_config(observation_mode)
            train_env_configs = [default_config]
        else:
            train_env_configs = [env_config]

    resolved_observation_mode = resolve_observation_mode(observation_mode, train_env_configs[0])
    for candidate_config in train_env_configs[1:]:
        resolve_observation_mode(resolved_observation_mode, candidate_config)

    base_env = gym.make(SHARED_CORE_ENV_ID, config=train_env_configs[0])
    env = base_env if reward_shaping is None else SafeRewardWrapper(base_env, reward_shaping)
    env.action_space.seed(seed)

    config_rng = random.Random(seed)

    def reset_episode(reset_seed):
        episode_config = config_rng.choice(train_env_configs)
        env.unwrapped.configure(episode_config)
        obs, info = env.reset(seed=reset_seed)
        obs = preprocess_observation(obs, resolved_observation_mode)
        return obs, info

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    agent_init_config = {
        "buffer_capacity": buffer_capacity,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "target_update_freq": target_update_freq,
        "epsilon": 1.0,
        "observation_mode": resolved_observation_mode,
        "double_dqn": double_dqn,
        "prioritized_replay": prioritized_replay,
        "priority_alpha": priority_alpha,
        "priority_eps": priority_eps,
    }

    checkpoint_agent_config = {
        **agent_init_config,
        "observation_config": deepcopy(train_env_configs[0]["observation"]),
    }

    agent = DQN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        **agent_init_config,
    )

    epsilon_start = 1.0
    epsilon = epsilon_start
    epsilon_min = 0.05

    def get_beta(step):
        if step < learning_starts:
            return beta_start

        fraction = min(1.0, (step - learning_starts) / max(1, total_steps - learning_starts))
        return beta_start + fraction * (beta_final - beta_start)

    obs, info = reset_episode(seed)

    episode_return = 0.0
    episode_returns = []
    episode_end_steps = []
    losses = []

    for step in range(1, total_steps + 1):
        action = agent.act(obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_observation(next_obs, resolved_observation_mode)

        done = terminated or truncated
        agent.buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_return += reward

        if len(agent.buffer) >= agent.batch_size and step >= learning_starts:
            beta = get_beta(step) if prioritized_replay else None
            batch, indices, weights = agent.buffer.sample(agent.batch_size, beta=beta)
            loss, td_errors = agent.update(batch, weights=weights)
            agent.buffer.update_priorities(indices, td_errors)
            losses.append(loss)

        if step % agent.target_update_freq == 0:
            agent.sync_target()

        if step >= learning_starts:
            fraction = min(1.0, (step - learning_starts) / max(1, total_steps - learning_starts))
            epsilon = epsilon_start + fraction * (epsilon_min - epsilon_start)

        if done:
            if verbose:
                print(f"step={step}, return={episode_return:.2f}, epsilon={epsilon:.3f}")

            episode_returns.append(episode_return)
            episode_end_steps.append(step)

            obs, info = reset_episode(seed + step)
            episode_return = 0.0

    env.close()

    metrics = {
        "episode_returns": np.array(episode_returns, dtype=np.float32),
        "episode_end_steps": np.array(episode_end_steps, dtype=np.int32),
        "losses": np.array(losses, dtype=np.float32),
    }

    run_metadata = {
        "seed": seed,
        "variant_name": variant_name,
        "safe_preset": safe_preset,
        "reward_shaping": reward_shaping,
        "double_dqn": double_dqn,
        "prioritized_replay": prioritized_replay,
        "priority_alpha": priority_alpha,
        "beta_start": beta_start,
        "beta_final": beta_final,
        "priority_eps": priority_eps,
        "training_traffic_names": training_traffic_names,
        "num_training_configs": len(train_env_configs),
        "total_steps": total_steps,
        "learning_starts": learning_starts,
        "observation_mode": resolved_observation_mode,
        "buffer_capacity": buffer_capacity,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "target_update_freq": target_update_freq,
    }

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "net_state_dict": agent.net.state_dict(),
                "target_net_state_dict": agent.target_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "seed": seed,
                "agent_config": checkpoint_agent_config,
                "metadata": run_metadata,
            },
            run_dir / "checkpoint.pt",
        )

        np.savez(run_dir / "metrics.npz", **metrics)
        (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    return agent, metrics
