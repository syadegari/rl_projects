import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .params import device
from .replay_buffer import ReplayBuffer
from .model import QNetwork


class DQNAgent:
    '''DQN Agent with replay buffer and soft update for target network'''
    def __init__(self, state_size, action_size, params):
        self.state_size = state_size
        self.action_size = action_size

        self.qnetwork_local = QNetwork(params['seed'], state_size, action_size)
        self.qnetwork_target = QNetwork(params['seed'], state_size, action_size)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), params['lr'])
        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), params['lr'])

        self.buffer = ReplayBuffer(params['buffer_size'], params['batch_size'], params['seed'])
        self.t_step = 0

        self.seed = params['seed']
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.update_every = params['update_every']
        self.batch_size = params['batch_size']
        self.tau = params['tau']

        print(self.seed, self.gamma, self.lr, self.update_every, self.batch_size, self.tau)

    def step(self, state, action, reward, next_state, done):
        '''
        Saves experiences in replay-buffer and learns if 
        thresholds are satisfied. 
        '''
        self.buffer.add(state, action, reward, next_state, done)
        #
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                experiences = self.buffer.sample()
                self.learn(experiences, self.gamma)

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

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        qtargets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        q_targets = rewards + (gamma * qtargets_next * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)
        self._learn(q_expected, q_targets)

    def _learn(self, q_expected, q_targets):
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        

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

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # dqn: qtargets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # ddqn
        with torch.no_grad():
            qtargets_next = torch.gather(self.qnetwork_target(next_states),
                                         dim=1,
                                         index=self.qnetwork_local(next_states).argmax(dim=1, keepdim=True))
        q_targets = rewards + (gamma * qtargets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)
        self._learn(q_expected, q_targets)

    
        
