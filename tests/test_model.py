import unittest
from dqn.model import QNetwork, DuelingQNetwork

import torch

class TestQNetwork(unittest.TestCase):
    def test_QNetwork_forward(self):
        state_size = 8
        action_size = 4
        seed = 101
        m = QNetwork(seed, state_size, action_size)
        out = m.forward(torch.rand(1, state_size))
        assert out.shape[1] == action_size

    def test_DuelingQNetwork_forward(self):
        state_size = 8
        action_size = 4
        seed = 101
        m = DuelingQNetwork(seed, state_size, action_size)
        out = m.forward(torch.rand(1, state_size))
        assert out.shape[1] == action_size


if __name__ == '__main__':
    unittest.main()
