import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import device

class QNetwork(nn.Module):
    def __init__(self, seed, state_size, action_size,
                 fc_units=256):
        self.seed = torch.manual_seed(seed)
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, action_size)
        )

    def forward(self, state):
        return self.network(state)

