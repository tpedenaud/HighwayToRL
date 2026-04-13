from pathlib import Path
import random

import gymnasium as gym
import highway_env
import numpy as np
import torch

from .dqn import DQN
from .ddqn_per import DDQN_PER
from .utils import preprocess_observation
from config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG


def train(seed=0, run_dir=None, double_dqn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode="rgb_array")
    env.action_space.seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if double_dqn:
        agent = DDQN_PER(
            observation_space=env.observation_space,
            action_space=env.action_space,
            buffer_capacity=10000,
            batch_size=32,
            learning_rate=0.001,
            gamma=0.99,
            target_update_freq=1000,
            epsilon=1.0,
            device=device,
            double_dqn=True,
        )
    else:
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

    total_steps = 20000
    learning_starts = 1000
    epsilon_start = 1.0
    epsilon = epsilon_start
    epsilon_min = 0.05


    obs, info = env.reset(seed=seed)
    obs = preprocess_observation(obs)

    episode_return = 0.0
    episode_returns = []
    episode_end_steps = []
    losses = []

    beta = 0.4
    for step in range(1, total_steps + 1):
        action = agent.act(obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_observation(next_obs)

        done = terminated or truncated
        agent.buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_return += reward

        if len(agent.buffer) >= agent.batch_size and step >= learning_starts:
            if double_dqn:

                batch_data = agent.buffer.sample(agent.batch_size, beta)
                loss = agent.update(batch_data)
                beta = min(1.0, beta + 1e-4)

            else:
                batch = agent.buffer.sample(agent.batch_size)
                loss = agent.update(batch)

            losses.append(loss)

        if step % agent.target_update_freq == 0:
            agent.sync_target()

        if step >= learning_starts:
            fraction = min(1.0, (step - learning_starts) / (total_steps - learning_starts))
            epsilon = epsilon_start + fraction * (epsilon_min - epsilon_start)

        if done:
            print(f"step={step}, return={episode_return:.2f}, epsilon={epsilon:.3f}")

            episode_returns.append(episode_return)
            episode_end_steps.append(step)

            obs, info = env.reset(seed=seed + step)
            obs = preprocess_observation(obs)
            episode_return = 0.0


    env.close()

    metrics = {
        "episode_returns": np.array(episode_returns, dtype=np.float32),
        "episode_end_steps": np.array(episode_end_steps, dtype=np.int32),
        "losses": np.array(losses, dtype=np.float32),
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
            },
            run_dir / "checkpoint.pt",
        )

        np.savez(run_dir / "metrics.npz", **metrics)

    return agent, metrics