import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .params import device
from .replay_buffer import PrioritizedReplayBuffer
from .model import QNetwork, DuelingQNetwork


def weighted_mse_loss(x, y, weights):
    # assert x.shape[0] == y.shape[0] == weights.shape, f'{x.shape, y.shape, weights.shape}'
    return (weights * (x - y) ** 2).mean()


class DQNAgent:
    '''DQN Agent with replay buffer and soft update for target network'''
    def __init__(self, state_size, action_size, params):
        self.state_size = state_size
        self.action_size = action_size

        if params['model'] == 'QNetwork':
            self.qnetwork_local = QNetwork(params['seed'], state_size, action_size).to(device)
            self.qnetwork_target = QNetwork(params['seed'], state_size, action_size).to(device)
        elif params['model'] == 'DuelingQNetwork':
            self.qnetwork_local = DuelingQNetwork(params['seed'], state_size, action_size).to(device)
            self.qnetwork_target = DuelingQNetwork(params['seed'], state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), params['lr'])

        self.buffer = PrioritizedReplayBuffer(
            params['buffer_size'],
            params['batch_size'],
            params['seed'],
            params['n_episodes'],
            params['alpha'],
            params['beta_0']
        )
        self.t_step = 0

        self.model = params['model']
        self.seed = params['seed']
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.update_every = params['update_every']
        self.batch_size = params['batch_size']
        self.tau = params['tau']

        print(self.seed, self.gamma, self.lr, self.update_every, self.batch_size, self.tau)

    def step(self, state, action, reward, next_state, done, i_episode):
        '''
        Saves experiences in replay-buffer and learns if 
        thresholds are satisfied. 
        '''
        self.buffer.add(state, action, reward, next_state, done)
        #
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                sampled_values = self.buffer.sample(i_episode)
                self.learn(sampled_values, self.gamma)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        if np.random.rand() > eps:
            return np.argmax(action_values.cpu().numpy())
        else:
            return np.random.randint(self.action_size)

    def get_qtarget(self, next_states, rewards, dones, gamma):
        qtargets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        return rewards + (gamma * qtargets_next * (1 - dones))

    def learn(self, sampled_values, gamma):
        states = sampled_values.states
        actions = sampled_values.actions
        rewards = sampled_values.rewards
        next_states = sampled_values.next_states
        dones = sampled_values.dones
        weights = sampled_values.weights
        indices = sampled_values.indices

        q_targets = self.get_qtarget(next_states, rewards, dones, gamma)
        q_expected = self.qnetwork_local(states).gather(1, actions)
        self._learn(q_expected, q_targets, indices, weights)

    def _learn(self, q_expected, q_targets, indices, weights):
        loss = weighted_mse_loss(q_expected, q_targets, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        # update priorities
        with torch.no_grad():
            td_errors = (q_expected - q_targets).abs().numpy().squeeze()
            # add buffer's eps (not the same as greedy-epsilon) to
            # ensure all experiences have a nonzero chance of
            # getting sampled
            td_errors = td_errors + self.buffer.eps
        self.buffer.update_priorities(indices, td_errors)

    def soft_update(self):
        tau = self.tau # tau << 1
        # target follows the local 
        # params_target = (1 - tau) * params_target + tau * params_local
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                             self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    
class DDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, params):
        super(DDQNAgent, self).__init__(state_size, action_size, params)

    def get_qtarget(self, next_states, rewards, dones, gamma):
        # dqn: qtargets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # ddqn
        with torch.no_grad():
            qtargets_next = torch.gather(self.qnetwork_target(next_states),
                                         dim=1,
                                         index=self.qnetwork_local(next_states).argmax(dim=1, keepdim=True))
        return rewards + (gamma * qtargets_next * (1 - dones))
