from tqdm import tqdm
import logging
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
    #
    double_dqn: bool = False
    dueling_network: bool = False
    #
    seed: int = 0
    batch_size: int = 0  
    n_episodes: int = 0
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
    buffer_beta_anneal_steps: int = 1_000_000
    #
    t_max: int = 0
    score_threshold: float = 0
    score_window: int = 100
    eps_init: float = 0
    eps_final: float = 0
    eps_decay: float = 0
    #
    device: str = 'cpu'
    num_state: int = 0
    num_action: int = 0

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
    def __init__(self, n_state: int, n_actions: int, nonlinear_fn=nn.ReLU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, 64),
            nonlinear_fn(),
            nn.Linear(64, 128),
            nonlinear_fn(),
            nn.Linear(128, 128),
            nonlinear_fn(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, n_state: int, n_actions: int, nonlinear_fn=nn.ReLU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, 64),
            nonlinear_fn(),
            nn.Linear(64, 128),
            nonlinear_fn(),
            nn.Linear(128, 128),
            nonlinear_fn(),
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class PrioritizedExperienceReplay:
    def __init__(self,
                 buffer_size: int,
                 batch_size: int,
                 seed: int,
                 beta_anneal_steps: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 eps: float = 1e-5,
                 device: Optional[str]=None) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.beta_anneal_steps = beta_anneal_steps
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
        beta = self.beta + (1.0 - self.beta) * step / self.beta_anneal_steps 
        return min(1.0, beta)

    def sample(self, step: int) -> SampledValues:
        N = len(self.buffer)
        beta = self.get_beta(step)

        # This if/else is not needed because we always take all of the buffer regardless of the size
        # if N < self.buffer_size:
        #     priorities = self.priorities[:N]
        # else:
        #     priorities = self.priorities
        priorities = self.priorities[:N]

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
            dones=self.to_torch(self.stack(samples, 'done'), torch.float32),
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
        if config.dueling_network:
            self.q_local = DuelingQNetwork(config.num_state, config.num_action)
            self.q_target = DuelingQNetwork(config.num_state, config.num_action)
        else:
            self.q_local = QNetwork(config.num_state, config.num_action)
            self.q_target = QNetwork(config.num_state, config.num_action)
        # Initiate target to be the same as local and later soft update the 
        # w_target <- tau * w_local
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.q_local.to(config.device)
        self.q_target.to(config.device)

        self.optimizer = optim.Adam(self.q_local.parameters(), lr=config.lr)
        
        self.num_action = config.num_action
        self.tau = config.soft_update
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        self.step_counter: int = 0
        self.learning_step: int = 0
        self.update_every = config.update_every

        self.double_dqn = config.double_dqn

        self.buffer = PrioritizedExperienceReplay(config.buffer_size, config.batch_size, config.seed, config.buffer_beta_anneal_steps, config.buffer_alpha, config.buffer_beta, config.buffer_eps, config.device)

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

    def step(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)
        self.step_counter = (self.step_counter + 1) % self.update_every
        if self.step_counter == 0 and len(self.buffer) > self.batch_size:
            self.learning_step += 1
            sampled_values = self.buffer.sample(self.learning_step)
            self.learn(sampled_values)

    def learn(self, sampled_values: SampledValues) -> None:
        with torch.no_grad():
            if self.double_dqn:
                actions = self.q_local(sampled_values.next_states).argmax(dim=1, keepdim=True)
                qtarget_next = self.q_target(sampled_values.next_states).gather(dim=1, index=actions)
            else:
                qtarget_next = self.q_target(sampled_values.next_states).max(dim=1, keepdim=True)[0]

        q_target = sampled_values.rewards + (self.gamma * qtarget_next * (1 - sampled_values.dones))
        q_expected = self.q_local(sampled_values.states).gather(dim=1, index=sampled_values.actions.reshape(-1, 1))

        loss = (sampled_values.weights * (q_target - q_expected) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_errors = (q_expected - q_target).abs().detach().cpu().numpy().squeeze()

        self.buffer.update_priorities(sampled_values.indices, td_errors)
        self.soft_update()
        
    def soft_update(self) -> None:
        '''
        Soft update from local network to target network
        '''
        assert self.tau < 0.1, f"tau parameter for soft update should be close to zero, but got tau = {self.tau}"
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


def save_experiment(episode: int, agent: DQNAgent, config: Config, scores: List[float], suffix_name: str):
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.q_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'config': config.__dict__,
        'scores': scores
    }
    torch.save(checkpoint, f'experiment_checkpoint_{suffix_name}.pth')


def dqn(n_episode: int, env, agent: DQNAgent, t_max, eps_init, eps_final, eps_decay, score_window, score_threshold, config: Config, progress_bar=False) -> List[float]:
    solved: bool = False
    scores = []
    eps = eps_init

    episodes = range(1, n_episode+1)
    if progress_bar:
        episodes = tqdm(episodes, desc="Episode", ncols=140)

    for episode in episodes:
        state = env_reset(env)
        score = 0

        for _ in range(t_max):
            action = agent.act(state.astype(np.float32), eps)
            next_state, reward, done = env_step(env, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        eps = max(eps_final, eps * eps_decay)

        mean_score = np.mean(scores[-score_window:])
        if mean_score > score_threshold and (not solved):
            save_experiment(episode, agent, config, scores, "solved")
            if progress_bar:
                episodes.set_description(f"Solved in {episode}")
            else:
                print(f"Solved in {episode} episodes")
            solved = True
            # return scores
        if progress_bar:
            episodes.set_postfix(
                {'score': f'{scores[-1]}', 'mean_score': f'{mean_score:.2f}', 'eps': f'{eps:.4f}'}
            )
        else:
            print(f"episode = {episode}: {scores[-1]}, mean score = {mean_score:.2f}, eps = {eps:.4f}")

    save_experiment(episode, agent, config, scores, "last")
    return scores


def run_experiment1(config: Config, env) -> List[float]:
    '''
    Use this for multiple runs from a notebook. User should
    instantiate the environment in the notebook and make sure 
    to close after running the experiments. 
    '''
    # suppress the messages from the agent at the beginning
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # num_state = brain.vector_observation_space_size
    # num_action = brain.vector_action_space_size
    config.num_state = brain.vector_observation_space_size
    config.num_action = brain.vector_action_space_size
    agent = DQNAgent(config)
    scores = dqn(config.n_episodes, env, agent, eps_init=config.eps_init, eps_final=config.eps_final, eps_decay=config.eps_decay, t_max=config.t_max, score_window=config.score_window, score_threshold=config.score_threshold, config=config, progress_bar=True)
    
    return scores


def run_experiment(config: Config, worker_id: int = 0) -> List[float]:
    '''Use this for a single run from a notebook'''
    # suppress the messages from the agent at the beginning
    logging.getLogger('unityagents').setLevel(logging.WARNING)
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=True, worker_id=worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # num_state = brain.vector_observation_space_size
    # num_action = brain.vector_action_space_size
    config.num_state = brain.vector_observation_space_size
    config.num_action = brain.vector_action_space_size
    try:
        agent = DQNAgent(config)
        scores = dqn(config.n_episodes, env, agent, eps_init=config.eps_init, eps_final=config.eps_final, eps_decay=config.eps_decay, t_max=config.t_max, score_window=config.score_window, score_threshold=config.score_threshold, config=config, progress_bar=True)
    finally:
        env.close()
    
    return scores


def main() -> None:
    '''This is for testing the algorithm form the command line'''
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=True)
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
        buffer_size=10_000, buffer_alpha=0.6, buffer_beta=0.4, buffer_eps=1e-5, buffer_beta_anneal_steps=10_000,
        #
        # dueling_network=True,
        # noisy_network=True,
        dueling_network=True,
        double_dqn=True,
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
        scores = dqn(config.n_episodes, env, agent, eps_init=config.eps_init, eps_final=config.eps_final, eps_decay=config.eps_decay, t_max=config.t_max, score_window=config.score_window, score_threshold=config.score_threshold)
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
