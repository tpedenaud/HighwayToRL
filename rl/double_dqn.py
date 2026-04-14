"""
Double DQN Agent - mitigates overestimation bias in value estimates.

Van Hasselt et al. (2015): "Deep Reinforcement Learning with Double Q-learning"

Key difference from standard DQN:
- Standard DQN:  y = r + γ * max_a' Q_target(s', a')
- Double-DQN:   y = r + γ * Q_target(s', argmax_a' Q_current(s', a'))

The current network selects actions, target network evaluates them.
This decoupling reduces overestimation bias and improves stability.
"""

from .network import Network
from .replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import random
import numpy as np


class DoubleDQN:
    """Double-DQN agent with reduced overestimation bias."""
    
    def __init__(self, observation_space, action_space, buffer_capacity, batch_size, 
                 learning_rate, gamma, target_update_freq, epsilon, device):
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.reset()

    def act(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_space.n)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.net(state)
                return q_values.argmax().item()
            
    def update(self, batch):
        """
        Double-DQN update rule.
        
        The key difference: use current network to SELECT actions,
        use target network to EVALUATE them.
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # ===== DOUBLE-DQN DIFFERENCE =====
            # Use CURRENT network to select best actions (this reduces overestimation)
            next_actions = self.net(next_states).argmax(dim=1)
            # Use TARGET network to evaluate those actions
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # ===================================
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def reset(self):
        """Initialize networks and optimizer."""
        obs_size = int(np.prod(self.observation_space.shape))
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.net = Network(obs_size, n_actions).to(self.device)
        self.target_net = Network(obs_size, n_actions).to(self.device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def sync_target(self):
        """Synchronize target network with current network."""
        self.target_net.load_state_dict(self.net.state_dict())
