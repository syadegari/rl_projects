import unittest
from dqn.model import QNetwork

import torch

class TestQNetwork(unittest.TestCase):
    def test_forward(self):
        state_size = 8
        action_size = 4
        seed = 101
        m = QNetwork(state_size, action_size, seed)
        assert m.forward(torch.rand(state_size)).shape[0] == action_size

if __name__ == '__main__':
    unittest.main()
