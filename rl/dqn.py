import random

import numpy as np
import torch

from .network import build_q_network
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class DQN:
    def __init__(
        self,
        observation_space,
        action_space,
        buffer_capacity,
        batch_size,
        learning_rate,
        gamma,
        target_update_freq,
        epsilon,
        device,
        observation_mode="kinematics",
        double_dqn=False,
        prioritized_replay=False,
        priority_alpha=0.6,
        priority_eps=1e-5,
        mlp_hidden_sizes=(128, 128),
        cnn_channels=(16, 32),
        cnn_kernel_sizes=(3, 3),
        cnn_strides=(1, 1),
        cnn_head_hidden=128,
    ):
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.observation_mode = observation_mode
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.priority_alpha = priority_alpha
        self.priority_eps = priority_eps
        self.mlp_hidden_sizes = tuple(mlp_hidden_sizes)
        self.cnn_channels = tuple(cnn_channels)
        self.cnn_kernel_sizes = tuple(cnn_kernel_sizes)
        self.cnn_strides = tuple(cnn_strides)
        self.cnn_head_hidden = cnn_head_hidden
        self.reset()

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_space.n)

        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(state)
            return q_values.argmax(dim=1).item()

    def update(self, batch, weights=None):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1).values

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = target_q_values - q_values
        per_sample_loss = td_errors.pow(2)

        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), td_errors.detach().abs().cpu().numpy()

    def reset(self):
        observation_shape = tuple(self.observation_space.shape)
        n_actions = self.action_space.n

        if self.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                self.buffer_capacity,
                alpha=self.priority_alpha,
                priority_eps=self.priority_eps,
            )
        else:
            self.buffer = ReplayBuffer(self.buffer_capacity)

        self.net = build_q_network(
            observation_shape=observation_shape,
            action_dim=n_actions,
            observation_mode=self.observation_mode,
            mlp_hidden_sizes=self.mlp_hidden_sizes,
            cnn_channels=self.cnn_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            cnn_strides=self.cnn_strides,
            cnn_head_hidden=self.cnn_head_hidden,
        ).to(self.device)
        self.target_net = build_q_network(
            observation_shape=observation_shape,
            action_dim=n_actions,
            observation_mode=self.observation_mode,
            mlp_hidden_sizes=self.mlp_hidden_sizes,
            cnn_channels=self.cnn_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            cnn_strides=self.cnn_strides,
            cnn_head_hidden=self.cnn_head_hidden,
        ).to(self.device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
