import unittest
import torch
import numpy as np

from reinforce.reinforce import get_return



class TestReward(unittest.TestCase):
    def test_zero_reward(self):
        N = 5
        rewards = np.zeros(N)
        R = get_return(rewards, gamma=1.0)
        self.assertEqual(0.0, R)

    def test_constant_reward(self):
        N = 5
        rewards = np.ones(N)
        gamma = 0.99
        R = get_return(rewards, gamma)
        hand_calculation = np.array([gamma ** i for i in range(N)]).sum()
        self.assertEqual( R, hand_calculation)

if __name__ == '__main__':
    unittest.main()
