import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

from unityagents import UnityEnvironment

import sys
import traceback
import pdb

from numpy import ndarray
from torch import Tensor, tensor
from typing import Any, Optional, Union, Tuple, List


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 0  
    total_steps: int = 0
    lr: float = 0.0
    #
    gamma: float = 0.0
    update_every: int = 1
    soft_update: float = 1e-3
    #
    buffer_size: int = 0
    buffer_alpha: float = 0.6
    buffer_beta: float = 0.4
    buffer_eps: float = 1e-5
    #
    t_max: int = 0
    score_threshold: float = 0
    score_window: int = 100
    eps_init: float = 0
    eps_final: float = 0
    eps_decay: float = 0
    #
    device: str = 'cpu'

    def __post_init__(self):
        self.device = get_device(self.device)
        assert self.eps_init >= self.eps_final, "Initial epsilon should be bigger than final epsilon"

@dataclass
class Experience:
    state: np.ndarray = None
    action: np.ndarray = None
    reward: float = None
    next_state: np.ndarray = None
    done: bool = False


@dataclass
class SampledValues:
    states: Tensor = tensor(0.0)
    actions: Tensor = tensor(0.0)
    rewards: Tensor = tensor(0.0)
    next_states: Tensor = tensor(0.0)
    dones: Tensor = tensor(0.0)
    weights: Tensor = tensor(0.0)
    indices: ndarray = np.array(0.0)


def get_device(device_name: Optional[str]) -> str:
    if device_name is None or device_name == 'cpu':
        return 'cpu'
    if device_name == 'gpu' or device_name == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    raise ValueError("Not a valid argument")


class QNetwork(nn.Module):
    def __init__(self, n_state: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PriotorizedExperienceReplay:
    def __init__(self,
                 buffer_size: int,
                 batch_size: int,
                 seed: int,
                 total_steps: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 eps: float = 1e-5,
                 device: Optional[str]=None) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.total_steps = total_steps
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
        self.buffer: list[Experience] = []
        self.pos = 0
        self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
       
        self.device  = device

    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done) -> None:
        max_priority = self.priorities.max() if self.buffer else 1.0
        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        # priority is always max when inserting a new experience
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer_size


    def get_beta(self, step: int) -> float:
        beta = self.beta + (1.0 - self.beta) * step / self.total_steps 
        return min(1.0, beta)

    def sample(self, step: int) -> SampledValues:
        N = len(self.buffer)
        beta = self.get_beta(step)

        if N < self.buffer_size:
            priorities = self.priorities[:N]
        else:
            priorities = self.priorities

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, self.batch_size, replace=False, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        return SampledValues(
            states=self.to_torch(self.stack(samples, 'state'), torch.float32),
            actions=self.to_torch(self.stack(samples, 'action'), torch.long),
            rewards=self.to_torch(self.stack(samples, 'reward'), torch.float32),
            next_states=self.to_torch(self.stack(samples, 'next_state'), torch.float32),
            dones=self.to_torch(self.stack(samples, 'done'), torch.uint8),
            weights=self.to_torch(weights, torch.float32),
            indices=indices
        )

    def to_torch(self, x: ndarray, dtype: torch.dtype) -> Tensor:
        return torch.from_numpy(x).to(self.device).to(dtype)

    def stack(self, samples: List[Experience], attribute:str) -> ndarray:
        return np.vstack([s.__getattribute__(attribute) for s in samples if s is not None])

    def update_priorities(self, batch_indices: ndarray, batch_priorities: Union[ndarray, Tensor]) -> None:
        if not isinstance(batch_priorities, ndarray):
            batch_priorities = batch_priorities.cpu().numpy()
        self.priorities[batch_indices] = batch_priorities + self.eps


class DQNAgent:
    def __init__(self, config: Config) -> None:
        self.q_local = QNetwork(config.num_state, config.num_action)
        self.q_target = QNetwork(config.num_state, config.num_action)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=config.lr)
        
        self.num_action = config.num_action
        self.tau = config.soft_update
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        self.step_counter: int = 0
        self.update_every = config.update_every

        self.buffer = PriotorizedExperienceReplay(config.buffer_size, config.batch_size, config.seed, config.total_steps, config.buffer_alpha, config.buffer_beta, config.buffer_eps, config.device)

        self.device = config.device

    def act(self, state: ndarray, eps: float) -> int:
        '''TODO: Should specify the shape of state. If only one state should be made into batch 1'''
        if np.random.rand() > eps:
            with torch.no_grad():
                self.q_local.eval()
                state = torch.from_numpy(state).reshape(1, -1).to(self.device)
                action_vals = self.q_local.forward(state).cpu().numpy()
                self.q_local.train()
                return action_vals.argmax().item()
        else:
            return np.random.choice(self.num_action)

    def step(self, state, action, reward, next_state, done, episode: int) -> None:
        self.buffer.add(state, action, reward, next_state, done)
        self.step_counter = (self.step_counter + 1) % self.update_every
        if self.step_counter == 0 and len(self.buffer) > self.batch_size:
            sampled_values = self.buffer.sample(episode)
            self.learn(sampled_values)

    def learn(self, sampled_values: SampledValues) -> None:
        with torch.no_grad():
            qtarget_next = self.q_target(sampled_values.next_states).max(dim=1)[0].unsqueeze(1)
        q_target = sampled_values.rewards + (self.gamma * qtarget_next * (1 - sampled_values.dones))
        q_expected = self.q_local(sampled_values.states).gather(dim=1, index=sampled_values.actions.reshape(-1, 1))

        loss = (sampled_values.weights * (q_target - q_expected) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_errors = (q_expected - q_target).abs().numpy().squeeze()

        self.buffer.update_priorities(sampled_values.indices, td_errors)
        
    def soft_update(self) -> None:
        '''
        Soft update from local network to target network
        '''
        assert self.tau < 0.1, "tau parameter for soft update should be close to zero"
        for param_local, param_target in zip(self.q_local.parameters(), self.q_target.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param_local.data * self.tau)


def env_reset(env) -> ndarray:
    brain_name = env.brain_names[0]
    env_info =  env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    return state


def env_step(env, action) -> Tuple[ndarray, float, bool]:
    brain_name = env.brain_names[0]
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    return next_state, reward, done


def dqn(n_episode: int, env, agent: DQNAgent, t_max, eps_init, eps_final, eps_decay, score_window, score_threshold) -> List[float]:
    scores = []
    for episode in range(1, n_episode + 1):
        state = env_reset(env)
        score = 0
        eps = eps_init

        for _ in range(t_max):
            action = agent.act(state, eps)
            next_state, reward, done = env_step(env, action)
            agent.step(state, action, reward, next_state, done, episode)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        eps = max(eps_final, eps * eps_decay)

        if np.mean(scores[:-score_window:]) > score_threshold:
            print(f"Sovled in {episode} episodes")
            return scores
        else:
            print(f"episode = {episode}: {scores[-1]}")


def create_plot(scores: List[float]) -> None:
    ...


def main() -> None:
    env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64", no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    num_state = brain.vector_observation_space_size
    num_action = brain.vector_action_space_size
    #
    config = Config(
        seed=101, batch_size=32, total_steps=100, lr=1e-3,
        #
        gamma=0.99, update_every=10, discount_factor=0.9, soft_update=0.999,
        #
        buffer_size=10_000, buffer_alpha=0.6, buffer_beta=0.4, buffer_eps=1e-5,
        #
        # dueling_network=True,
        # noisy_network=True,
        # double_dqn=True,
        # priotorized_exp_replay_buffer=True,
        #
        t_max=100, score_threshold=13.0, score_window=100, eps_init=1.0, eps_final=0.01, eps_decay=0.99,
        #
        num_state=num_state, num_action=num_action,
        #
        device='cpu'

    ) 
    try:
        agent = DQNAgent(config)
        scores = dqn(100, env, agent, )
        create_plot(scores)
    except RuntimeError as error:
        print(f"Caught an error: {error}") 
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        pdb.post_mortem(exc_traceback)
    finally:
        env.close()


if __name__ == "__main__":
    main()
