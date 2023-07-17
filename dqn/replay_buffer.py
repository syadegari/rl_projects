import random
import torch
import numpy as np
from collections import deque, namedtuple

from .params import device


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", 
            field_names=["state", 
                         "action",
                         "reward",
                         "next_state",
                         "done"]
        )
        self.seed = random.seed(seed)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            self.experience(state, action, reward, next_state, done)
        )

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        to_torch = lambda experiences, key: torch.from_numpy(
            np.vstack(
                [e.__getattribute__(key) for e in experiences if e is not None]
            )
        ).to(device)

        states = to_torch(experiences, 'state').to(torch.float32)
        actions = to_torch(experiences, 'action').to(torch.long)
        rewards = to_torch(experiences, 'reward').to(torch.float32)
        next_states = to_torch(experiences, 'next_state').to(torch.float32)
        dones = to_torch(experiences, 'done').to(torch.uint8)

        lengths = [len(x) for x in [states, actions, rewards, next_states, dones]]
        assert all(x == lengths[0] for x in lengths)

        return states, actions, rewards, next_states, dones
