import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import dump_json, resolve_device, seed_env, set_seed
from common_observation_extension import (
    DEFAULT_EXTENSION_SETTINGS,
    DEFAULT_EXTENSION_TRAIN_SEEDS,
    ensure_extension_run_dirs,
    get_observation_shape_and_num_actions,
    make_extension_env,
    preprocess_observation,
)
from observation_extension_config import build_config
from shared_core_config import SHARED_CORE_ENV_ID


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


class MLPQNetwork(nn.Module):
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


class CNNQNetwork(nn.Module):
    def __init__(self, observation_shape, num_actions, channels, kernel_sizes, strides, head_hidden):
        super().__init__()
        in_channels, _, _ = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=kernel_sizes[0], stride=strides[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[1], stride=strides[1]),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, *observation_shape), dtype=torch.float32)
            flattened_dim = int(np.prod(self.encoder(dummy).shape[1:]))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
        )

    def forward(self, observations):
        encoded = self.encoder(observations)
        return self.head(encoded)


def build_q_network(observation_mode, observation_shape, num_actions, settings):
    if observation_mode == "kinematics":
        return MLPQNetwork(observation_shape[0], num_actions, settings["mlp_hidden_sizes"]), "mlp"
    if observation_mode == "occupancy_grid":
        return (
            CNNQNetwork(
                observation_shape,
                num_actions,
                settings["cnn_channels"],
                settings["cnn_kernel_sizes"],
                settings["cnn_strides"],
                settings["cnn_head_hidden"],
            ),
            "cnn",
        )
    raise ValueError(f"Unknown observation mode: {observation_mode}")


class ExtensionDQNPolicy:
    def __init__(self, q_network, observation_mode, device):
        self.q_network = q_network
        self.observation_mode = observation_mode
        self.device = device

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        settings = checkpoint["dqn_settings"]
        observation_mode = checkpoint["observation_mode"]
        observation_shape = tuple(checkpoint["observation_shape"])
        num_actions = checkpoint["num_actions"]

        q_network, _ = build_q_network(observation_mode, observation_shape, num_actions, settings)
        q_network = q_network.to(device)
        q_network.load_state_dict(checkpoint["model_state_dict"])
        q_network.eval()
        return cls(q_network, observation_mode, device)

    def predict(self, observation):
        processed = preprocess_observation(observation, self.observation_mode)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(processed, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = int(self.q_network(obs_tensor).argmax(dim=1).item())
        return action


def epsilon_for_step(step, settings):
    if step >= settings["epsilon_decay_steps"]:
        return settings["epsilon_end"]
    progress = step / float(settings["epsilon_decay_steps"])
    return settings["epsilon_start"] + progress * (settings["epsilon_end"] - settings["epsilon_start"])


def save_checkpoint(
    path,
    q_network,
    optimizer,
    step,
    seed,
    observation_mode,
    observation_shape,
    num_actions,
    encoder_type,
    settings,
):
    torch.save(
        {
            "seed": int(seed),
            "step": int(step),
            "observation_mode": observation_mode,
            "observation_shape": list(observation_shape),
            "num_actions": int(num_actions),
            "encoder_type": encoder_type,
            "dqn_settings": settings,
            "model_state_dict": q_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_one_seed(observation_mode, seed, total_timesteps, device="cpu", settings=None):
    settings = dict(DEFAULT_EXTENSION_SETTINGS if settings is None else settings)
    device = resolve_device(device)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    paths = ensure_extension_run_dirs(observation_mode, seed)

    env = make_extension_env(observation_mode)
    raw_observation, _ = seed_env(env, seed)
    observation = preprocess_observation(raw_observation, observation_mode)

    observation_shape, num_actions = get_observation_shape_and_num_actions(observation_mode)
    q_network, encoder_type = build_q_network(observation_mode, observation_shape, num_actions, settings)
    q_network = q_network.to(device)
    target_network, _ = build_q_network(observation_mode, observation_shape, num_actions, settings)
    target_network = target_network.to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=settings["learning_rate"])
    replay_buffer = ReplayBuffer(settings["buffer_size"], observation_shape)

    print(
        (
            f"[train] mode={observation_mode} seed={seed} "
            f"timesteps={total_timesteps} encoder={encoder_type}"
        ),
        flush=True,
    )

    last_step = 0
    was_interrupted = False

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

        try:
            for step in range(1, total_timesteps + 1):
                last_step = step
                epsilon = epsilon_for_step(step - 1, settings)
                if rng.random() < epsilon:
                    action = int(rng.integers(num_actions))
                else:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(q_network(obs_tensor).argmax(dim=1).item())

                raw_next_observation, reward, terminated, truncated, info = env.step(action)
                next_observation = preprocess_observation(raw_next_observation, observation_mode)
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
                        next_max = target_network(next_observations).max(dim=1).values
                        target_q = rewards + settings["gamma"] * (1.0 - dones) * next_max

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
                        seed,
                        observation_mode,
                        observation_shape,
                        num_actions,
                        encoder_type,
                        settings,
                    )
                    print(
                        (
                            f"[checkpoint] mode={observation_mode} seed={seed} "
                            f"step={step}/{total_timesteps}"
                        ),
                        flush=True,
                    )

                if done:
                    episode_index += 1
                    mean_speed = speed_sum / max(episode_length, 1)
                    writer.writerow(
                        {
                            "episode_index": episode_index,
                            "global_step": step,
                            "episode_reward": round(episode_reward, 6),
                            "episode_length": episode_length,
                            "epsilon": round(epsilon, 6),
                            "crashed": int(crashed),
                            "mean_speed": round(mean_speed, 6),
                        }
                    )
                    training_log.flush()
                    print(
                        (
                            f"[episode] mode={observation_mode} seed={seed} ep={episode_index} "
                            f"step={step}/{total_timesteps} reward={episode_reward:.3f} "
                            f"len={episode_length} crashed={int(crashed)} "
                            f"eps={epsilon:.3f} speed={mean_speed:.2f}"
                        ),
                        flush=True,
                    )
                    raw_observation, _ = env.reset()
                    observation = preprocess_observation(raw_observation, observation_mode)
                    episode_reward = 0.0
                    episode_length = 0
                    speed_sum = 0.0
                    crashed = False
                else:
                    observation = next_observation
        except KeyboardInterrupt:
            was_interrupted = True
            print(
                (
                    f"[interrupt] mode={observation_mode} seed={seed} "
                    f"saving emergency checkpoint at step={last_step}"
                ),
                flush=True,
            )
            if last_step > 0:
                save_checkpoint(
                    paths["checkpoints_dir"] / f"interrupted_step_{last_step:07d}.pt",
                    q_network,
                    optimizer,
                    last_step,
                    seed,
                    observation_mode,
                    observation_shape,
                    num_actions,
                    encoder_type,
                    settings,
                )

    final_step = last_step if last_step > 0 else total_timesteps
    save_checkpoint(
        paths["final_model_path"],
        q_network,
        optimizer,
        final_step,
        seed,
        observation_mode,
        observation_shape,
        num_actions,
        encoder_type,
        settings,
    )
    env.close()

    dump_json(
        paths["metadata_path"],
        {
            "observation_mode": observation_mode,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "stopped_step": final_step,
            "interrupted": was_interrupted,
            "device": device,
            "environment_id": SHARED_CORE_ENV_ID,
            "environment_config": build_config(observation_mode),
            "encoder_type": encoder_type,
            "dqn_settings": settings,
        },
    )

    if was_interrupted:
        print(
            (
                f"[saved_on_interrupt] mode={observation_mode} seed={seed} "
                f"step={final_step} model={paths['final_model_path'].name}"
            ),
            flush=True,
        )

    return paths


def train_many(observation_mode, seeds, total_timesteps, device="cpu", settings=None):
    for seed in seeds:
        print(f"[train_many] start mode={observation_mode} seed={seed}", flush=True)
        train_one_seed(observation_mode, seed, total_timesteps, device=device, settings=settings)
        print(f"[train_many] done mode={observation_mode} seed={seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train custom DQN on kinematics or occupancy-grid observations.")
    parser.add_argument("--observation-mode", choices=["kinematics", "occupancy_grid"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_EXTENSION_TRAIN_SEEDS)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    args = parser.parse_args()

    train_many(args.observation_mode, args.seeds, args.timesteps, device=args.device)


if __name__ == "__main__":
    main()
