import random
import torch
import numpy as np
from collections import deque, namedtuple
from dataclasses import dataclass

from .params import device

@dataclass
class SampledValues:
    states       : None
    actions      : None
    rewards      : None
    next_states  : None
    dones        : None
    indices      : None
    weights      : None

class PrioritizedReplayBuffer:
    '''
    Propportional experience replay buffer. 
    - One can recover the vanilla experience replay (without priority) 
    by setting alpha=0.0. 
    '''
    def __init__(self, buffer_size, batch_size, seed,
                n_total_steps, alpha=0.6, beta_0=0.4, eps=1e-5):
        self.alpha = alpha
        self.beta_0 = beta_0
        self.eps = eps
        self.n_total_steps = n_total_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(seed)
        # using deque for buffer (like we did with ReplayBuffer) will be 
        # tricky in this situation since we have to keep track of two
        # list-like objects, both self.buffer and self.priorities.
        # If we insert a new element into the buffer, we need to 
        # assign max-priority to it in self.priorities.
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.experience = namedtuple("Experience",
                                    field_names=["state",
                                                "action",
                                                "reward",
                                                "next_state",
                                                "done"])

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        assert state.ndim >= 1
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self) < self.buffer_size:
            self.buffer.append(self.experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = self.experience(state, action, reward, next_state, done)

        # set the priority of new elements to max to increase 
        # the chance of being sampled at least once
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer_size

    def get_beta(self, i_step):
        '''
        Anneals beta linearly from its initial value, beta_0 to 1.
        Check section 3.4 of the paper on Prio Exp Replay
        '''
        return (1 - self.beta_0) * i_step / self.n_total_steps + self.beta_0

    def sample(self, i_step):
        N = len(self)
        beta = self.get_beta(i_step)
        # The if-else block is necessary since we are using a circular list
        # (check the last line in .add method) and have to keep track of 
        # pos, even when the buffer is full.
        # Otherwise this could have been simplified to just one line:
        #    priorities = self.priorities[:self.pos]
        if N == self.buffer_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs  = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, self.batch_size, replace=False, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = torch.from_numpy(weights).to(torch.float32).to(device)
        
        to_torch = lambda experiences, key: torch.from_numpy(
            np.vstack(
                [e.__getattribute__(key) for e in experiences if e is not None]
            )
        ).to(device)
        
        states = to_torch(samples, 'state').to(torch.float32)
        actions = to_torch(samples, 'action').to(torch.long)
        rewards = to_torch(samples, 'reward').to(torch.float32)
        next_states = to_torch(samples, 'next_state').to(torch.float32)
        dones = to_torch(samples, 'done').to(torch.uint8)
        
        return SampledValues(states=states, 
                             actions=actions,
                             rewards=rewards, 
                             next_states=next_states, 
                             dones=dones, 
                             indices=indices, 
                             weights=weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
