import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import (
    CUSTOM_MODEL_NAME,
    DEFAULT_DQN_SETTINGS,
    DEFAULT_TRAIN_SEEDS,
    SHARED_CORE_CONFIG,
    SHARED_CORE_ENV_ID,
    dump_json,
    ensure_run_dirs,
    get_observation_dim_and_num_actions,
    make_env,
    resolve_device,
    seed_env,
    set_seed,
)


class ReplayBuffer:
    def __init__(self, capacity, observation_shape):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, done):
        self.observations[self.position] = observation
        self.next_observations[self.position] = next_observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }


class QNetwork(nn.Module):
    def __init__(self, observation_dim, num_actions, hidden_sizes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

    def forward(self, observations):
        return self.network(observations)


class CustomDQNPolicy:
    def __init__(self, q_network, device):
        self.q_network = q_network
        self.device = device

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        q_network = QNetwork(
            checkpoint["observation_dim"],
            checkpoint["num_actions"],
            checkpoint["hidden_sizes"],
        ).to(device)
        q_network.load_state_dict(checkpoint["model_state_dict"])
        q_network.eval()
        return cls(q_network, device)

    def predict(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_network(observation).argmax(dim=1).item())


def epsilon_for_step(step, settings):
    if step >= settings["epsilon_decay_steps"]:
        return settings["epsilon_end"]
    progress = step / float(settings["epsilon_decay_steps"])
    return settings["epsilon_start"] + progress * (settings["epsilon_end"] - settings["epsilon_start"])


def save_checkpoint(path, q_network, optimizer, step, observation_dim, num_actions, hidden_sizes, seed):
    torch.save(
        {
            "seed": int(seed),
            "step": int(step),
            "observation_dim": int(observation_dim),
            "num_actions": int(num_actions),
            "hidden_sizes": [int(size) for size in hidden_sizes],
            "model_state_dict": q_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_one_seed(seed, total_timesteps, device="cpu", settings=None):
    settings = dict(DEFAULT_DQN_SETTINGS if settings is None else settings)
    device = resolve_device(device)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    paths = ensure_run_dirs(CUSTOM_MODEL_NAME, seed)

    env = make_env()
    observation, _ = seed_env(env, seed)
    observation_dim, num_actions = get_observation_dim_and_num_actions()

    q_network = QNetwork(observation_dim, num_actions, settings["hidden_sizes"]).to(device)
    target_network = QNetwork(observation_dim, num_actions, settings["hidden_sizes"]).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(q_network.parameters(), lr=settings["learning_rate"])
    replay_buffer = ReplayBuffer(settings["buffer_size"], observation.shape)

    with paths["training_log_path"].open("w", newline="", encoding="utf-8") as training_log:
        writer = csv.DictWriter(
            training_log,
            fieldnames=[
                "episode_index",
                "global_step",
                "episode_reward",
                "episode_length",
                "epsilon",
                "crashed",
                "mean_speed",
            ],
        )
        writer.writeheader()

        episode_index = 0
        episode_reward = 0.0
        episode_length = 0
        speed_sum = 0.0
        crashed = False

        for step in range(1, total_timesteps + 1):
            epsilon = epsilon_for_step(step - 1, settings)
            if rng.random() < epsilon:
                action = int(rng.integers(num_actions))
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(q_network(obs_tensor).argmax(dim=1).item())

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(observation, action, reward, next_observation, done)

            episode_reward += float(reward)
            episode_length += 1
            speed_sum += float(info.get("speed", 0.0))
            crashed = crashed or bool(info.get("crashed", False))

            if (
                step >= settings["learning_starts"]
                and step % settings["train_freq"] == 0
                and len(replay_buffer) >= settings["batch_size"]
            ):
                batch = replay_buffer.sample(settings["batch_size"], rng)
                observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
                next_observations = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=device)
                actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
                rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
                dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)

                predicted_q = q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = rewards + settings["gamma"] * (1.0 - dones) * target_network(next_observations).max(dim=1).values

                loss = F.smooth_l1_loss(predicted_q, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), settings["gradient_clip_norm"])
                optimizer.step()

            if step % settings["target_update_interval"] == 0:
                target_network.load_state_dict(q_network.state_dict())

            if step % settings["checkpoint_interval"] == 0:
                save_checkpoint(
                    paths["checkpoints_dir"] / f"step_{step:07d}.pt",
                    q_network,
                    optimizer,
                    step,
                    observation_dim,
                    num_actions,
                    settings["hidden_sizes"],
                    seed,
                )

            if done:
                episode_index += 1
                writer.writerow(
                    {
                        "episode_index": episode_index,
                        "global_step": step,
                        "episode_reward": round(episode_reward, 6),
                        "episode_length": episode_length,
                        "epsilon": round(epsilon, 6),
                        "crashed": int(crashed),
                        "mean_speed": round(speed_sum / max(episode_length, 1), 6),
                    }
                )
                training_log.flush()
                observation, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
                speed_sum = 0.0
                crashed = False
            else:
                observation = next_observation

    save_checkpoint(
        paths["final_model_path"],
        q_network,
        optimizer,
        total_timesteps,
        observation_dim,
        num_actions,
        settings["hidden_sizes"],
        seed,
    )
    env.close()

    dump_json(
        paths["metadata_path"],
        {
            "model_name": CUSTOM_MODEL_NAME,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "device": device,
            "environment_id": SHARED_CORE_ENV_ID,
            "shared_core_config": SHARED_CORE_CONFIG,
            "dqn_settings": settings,
        },
    )

    return paths


def train_many(seeds, total_timesteps, device="cpu", settings=None):
    for seed in seeds:
        train_one_seed(seed, total_timesteps, device=device, settings=settings)


def main():
    parser = argparse.ArgumentParser(description="Train the custom DQN agent.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    args = parser.parse_args()
    train_many(args.seeds, args.timesteps, device=args.device)


if __name__ == "__main__":
    main()
