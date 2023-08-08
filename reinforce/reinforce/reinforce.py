from collections import deque
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.distributions import Categorical
import gym

from .params import device

pairwise = lambda l: ((i, j) for i, j in zip(l[:-1], l[1:]))

class Policy(nn.Module):
    def __init__(self, nS: int, nA: int, n_fcs: list):
        super(Policy, self).__init__()
        #
        net = []
        for i, j in pairwise([nS] + n_fcs + [nA]):
            net.append(nn.Linear(i, j))
            net.append(nn.ReLU())
        net = net[:-1] # drop the last relu
        self.network = nn.Sequential(*net)
        self.nA = nA
        self.nS = nS

    def forward(self, state):
        if isinstance(state, (np.ndarray,)):
            t = torch.from_numpy(state).to(torch.float32).to(device).unsqueeze(0)
        else:
            t = state.to(device).unsqueeze(0)
        return F.softmax(self.network(t), dim=1)

    def act(self, state):
        probs = self.forward(state)
        m = Categorical(probs=probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob

@dataclass
class GradInfo:
    min_grad: list = field(default_factory=list)
    max_grad: list = field(default_factory=list)
    grad_amplitude : list = field(default_factory=list)

    def add_info(self, net: nn.Module):
        min_grad, max_grad = get_min_max_grads(net)
        self.min_grad.append(min_grad)
        self.max_grad.append(max_grad)
        self.grad_amplitude.append(get_grad_amplitude(net))

def optimize(policy_loss: torch.Tensor, optimizer:Optimizer):
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

def get_return(rewards:np.ndarray, gamma:float) -> float:
    '''r_1 + r_2 * gamma + r_3 * gamma^2 + ...'''
    return (((gamma * np.ones_like(rewards)) ** np.arange(len(rewards))) * rewards).sum()

def get_policy_loss(log_probs:torch.Tensor, R:float) -> torch.Tensor:
    return (-R * torch.cat(log_probs)).sum()

def get_min_max_grads(net: nn.Module) -> Tuple[float, float]:
    grad_min = None
    grad_max = None

    for param in net.parameters():
        if param.requires_grad and param.grad is not None:
            if grad_min is None and grad_max is None:
                grad_min = param.grad.data.abs().min().item()
                grad_max = param.grad.data.abs().max().item()
            else:
                grad_min = min(grad_min, param.grad.data.abs().min().item())
                grad_max = max(grad_max, param.grad.data.abs().max().item())
    return grad_min, grad_max

def get_grad_amplitude(net: nn.Module) -> Optional[float]:
    grad_all = None

    for param in net.parameters():
        if param.requires_grad and param.grad is not None:
            if grad_all is None:
                grad_all = param.grad.view(-1).clone()
            else:
                grad_all = torch.cat((grad_all, param.grad.view(-1)))

    if grad_all is None:
        return None
    else:
        return grad_all.norm().item()


def log_results(policy: nn.Module, scores_window: deque, i_episode:int, print_every:int) -> None:
    if i_episode % print_every == 0:
        mean_score = np.mean(scores_window)
        min_grad, max_grad = get_min_max_grads(policy)
        min_max_grad_info = f'min/max grad: {min_grad:.2f}/{max_grad:.2f}'
        grad_amp_info = f'grad amp.: {get_grad_amplitude(policy):.2f}'
        print(f'Episode {i_episode}\tAverage Score: {mean_score:.2f}  {min_max_grad_info}  {grad_amp_info}')

def is_solved(scores_window: deque, policy: nn.Module, score_threshold: float) -> bool:
    if np.mean(scores_window) >= score_threshold:
        return True
    return False

def collect_trajectories(env: gym.Env, seed: int, policy: Policy, max_t: int) -> Tuple[List[torch.Tensor], np.ndarray]:
#     state, _ = env.reset(seed=seed)
    state, _ = env.reset()
    log_probs = []
    rewards = []
    for t in range(max_t):
        action, log_prob = policy.act(state)
        next_state, reward, done, _, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        if done:
            break
        state = next_state
    return np.array(rewards), log_probs

def seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def reinforce(env, policy, optimizer, seed, n_episodes, max_t, gamma, score_threshold, print_every):
    scores_window = deque(maxlen=100)
    scores = []
    grad_info = GradInfo()
    for i_episode in range(1, n_episodes + 1):
        rewards, log_probs = collect_trajectories(env, seed, policy, max_t)
        scores.append(sum(rewards))
        scores_window.append(sum(rewards))
        R = get_return(rewards, gamma)
        policy_loss = get_policy_loss(log_probs, R)
        optimize(policy_loss, optimizer)
        grad_info.add_info(policy)
        log_results(policy, scores_window, i_episode, print_every)
        if is_solved(scores_window, policy, score_threshold):
            break
    return scores, scores_window, grad_info

def wrapper(params: dict, called_via_cmdline: bool=False):
    seed(params['seed'])
    env = gym.make(params['env_name'])
    nS, nA = env.observation_space.shape[0], env.action_space.n
    n_fcs = params['policy_fc_units']
    policy = Policy(nS, nA, n_fcs)
    optimizer = optim.Adam(policy.parameters(), lr=params['lr'])

    scores, scores_window, grad_info = reinforce(env, policy, optimizer,
              seed=params['seed'],
              n_episodes=params['n_episodes'],
              max_t=params['max_t'],
              gamma=params['gamma'],
              score_threshold=params['score_threshold'],
              print_every=params['print_every'])

    if called_via_cmdline:
        with open(f'scores_{params["experiment_name"]}.dat', 'w') as f:
            f.writelines(f'{str(score)}\n' for score in scores)
    else:
        return scores, scores_window, grad_info
