import numpy as np
import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
        )

    def forward(self, observations):
        return self.network(observations)


class CNNQNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        action_dim,
        channels=(16, 32),
        kernel_sizes=(3, 3),
        strides=(1, 1),
        head_hidden=128,
    ):
        super().__init__()
        in_channels = observation_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=kernel_sizes[0],
                stride=strides[0],
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=kernel_sizes[1],
                stride=strides[1],
            ),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, *observation_shape), dtype=torch.float32)
            encoded_shape = self.encoder(dummy).shape[1:]
            flattened_dim = int(np.prod(encoded_shape))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, action_dim),
        )

    def forward(self, observations):
        return self.head(self.encoder(observations))


def build_q_network(
    observation_shape,
    action_dim,
    observation_mode="kinematics",
    mlp_hidden_sizes=(128, 128),
    cnn_channels=(16, 32),
    cnn_kernel_sizes=(3, 3),
    cnn_strides=(1, 1),
    cnn_head_hidden=128,
):
    if observation_mode == "kinematics":
        observation_dim = int(np.prod(observation_shape))
        return MLPQNetwork(observation_dim, action_dim, hidden_sizes=mlp_hidden_sizes)

    if observation_mode == "occupancy_grid":
        return CNNQNetwork(
            observation_shape=observation_shape,
            action_dim=action_dim,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            head_hidden=cnn_head_hidden,
        )

    raise ValueError(f"Unknown observation mode: {observation_mode}")
