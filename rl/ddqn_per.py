from .network import Network
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import random
import numpy as np


class DDQN_PER:
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
        double_dqn=True,
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
        self.double_dqn = double_dqn

        self.reset()

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_space.n)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.net(state)
                return q_values.argmax().item()

    def update(self, batch_data):
        batch, indices, weights = batch_data

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1).values

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = target_q_values - q_values

        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        return loss.item()
    
    def reset(self):
        obs_size = int(np.prod(self.observation_space.shape))
        n_actions = self.action_space.n

        self.buffer = PrioritizedReplayBuffer(self.buffer_capacity)

        self.net = Network(obs_size, n_actions).to(self.device)
        self.target_net = Network(obs_size, n_actions).to(self.device)

        self.sync_target()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())