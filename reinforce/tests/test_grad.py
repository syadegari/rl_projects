import unittest
import torch

from reinforce.reinforce import get_min_max_grads, get_grad_amplitude
from reinforce.reinforce import Policy

class TestGradFunctions(unittest.TestCase):
    
    def setUp(self):
        self.model = Policy(4, 2, [16])
        
    def test_get_min_max_grads(self):
        _, log_prob = self.model.act(torch.randn(1, 4))
        # maybe more than one .act call to have a list of log probs
        loss = log_prob.sum()
        loss.backward()
        min_grad, max_grad = get_min_max_grads(self.model)
        self.assertIsInstance(min_grad, float)
        self.assertIsInstance(max_grad, float)
        self.assertLessEqual(min_grad, max_grad)

    def test_get_grad_amplitude(self):
        _, log_prob = self.model.act(torch.randn(1, 4))
        # maybe more than one .act call to have a list of log probs
        loss = log_prob.sum()
        loss.backward()
        grad_amp = get_grad_amplitude(self.model)
        self.assertIsInstance(grad_amp, float)
        self.assertGreaterEqual(grad_amp, 0.0)

if __name__ == "__main__":
    unittest.main()
