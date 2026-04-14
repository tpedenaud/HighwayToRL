import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=None):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[index] for index in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        return None

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, priority_eps=1e-5):
        super().__init__(capacity)
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = 1.0

        current_size = len(self.buffer)
        scaled_priorities = self.priorities[:current_size] ** self.alpha
        total_priority = scaled_priorities.sum()

        if total_priority <= 0:
            probabilities = np.full(current_size, 1.0 / current_size, dtype=np.float32)
        else:
            probabilities = scaled_priorities / total_priority

        indices = np.random.choice(
            current_size,
            size=batch_size,
            replace=False,
            p=probabilities,
        )
        batch = [self.buffer[index] for index in indices]

        weights = (current_size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            adjusted_priority = max(float(priority), self.priority_eps)
            self.priorities[index] = adjusted_priority
            self.max_priority = max(self.max_priority, adjusted_priority)
