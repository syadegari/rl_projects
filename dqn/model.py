import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import device

class QNetwork(nn.Module):
    def __init__(self, seed, state_size, action_size,
                 fc_units=256):
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
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


class DuelingQNetwork(nn.Module):
    def __init__(self, seed, state_size, action_size,
                 fc_units=256):
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        super(DuelingQNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
        )
        self.v_branch = nn.Sequential(
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 1)
        )
        self.a_branch = nn.Sequential(
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, action_size)
        )

    def forward(self, x):
        x = self.main(x)
        v_out = self.v_branch(x)
        a_out = self.a_branch(x)
        #
        return v_out + a_out - a_out.mean(axis=1, keepdim=True)
