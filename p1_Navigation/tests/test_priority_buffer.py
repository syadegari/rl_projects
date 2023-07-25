import unittest
from dqn.replay_buffer import PrioritizedReplayBuffer

import torch
import numpy as np

class TestPriorityReplayBuffer(unittest.TestCase):
    def test_alpha_zero_is_uniform_sampling(self):
        '''
        The premise of this test is to see if the buffer with parameter alpha=0.0
        will be reduced to uniform replay buffer. This is crucial since we are
        removing the replay buffer and instead use the prio-buffer when we need to
        use a replay buffer. The test verifies that the weights are equal to 1.0
        when alpha=0.0, therefore reducing the prio buffer to a simple, uniform
        replay buffer. 
        '''
        n_batch = 64
        buffer_size = 200
        buffer = PrioritizedReplayBuffer(buffer_size=buffer_size,
                                         batch_size=n_batch, 
                                         seed=101, 
                                         n_total_steps=1000, 
                                         alpha=0.0, 
                                         beta_0=.4)
        nS, nA = 8, 4
        for _ in range(400):
            buffer.add(
                state=np.random.rand(nS),
                action=np.random.randint(0, nA),
                reward=np.random.rand(),
                next_state=np.random.rand(nS),
                done=np.random.randint(0, 2)
            )
        batch_indices = np.random.choice(np.arange(buffer_size), n_batch, replace=False)
        batch_priorities = np.abs(np.random.random(n_batch))
        buffer.update_priorities(
            batch_indices=batch_indices,
            batch_priorities=batch_priorities
        )
        sampled_values = buffer.sample(i_step=255)
        self.assertTrue(torch.all(sampled_values.weights.cpu().detach() == torch.ones_like(sampled_values.weights)))


if __name__ == '__main__':
    unittest.main()
