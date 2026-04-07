import torch.nn as nn

class Network(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=128):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )   

    def forward(self, x):
        return self.net(x)