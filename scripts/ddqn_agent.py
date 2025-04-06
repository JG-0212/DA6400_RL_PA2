import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from collections import deque, namedtuple

from scripts.helpers import epsilon_greedy, softmax, compute_decay


class DDQNetwork(nn.Module):

    def __init__(self, state_size, action_size, network_type, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.network_type = network_type

        hidden_size_1 = 64
        hidden_size_2 = 32
        self.shared_network = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=hidden_size_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1)
        )

        self.state_value_head = nn.Sequential(
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size_2, out_features=1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size_2, out_features=action_size)
        )

    def forward(self, observation):

        x = self.shared_network(observation)
        state_value = self.state_value_head(x)
        advantages = self.advantage_head(x)

        if self.network_type == 1:
            q_values = (state_value
                        + (advantages - torch.mean(advantages,
                                                   dim=1, keepdim=True)))
        elif self.network_type == 2:
            q_values = (state_value
                        + (advantages - torch.max(advantages,
                                                  dim=1, keepdim=True)[0]))
        return q_values


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed, device='cpu'):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):

        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(
            np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDQNAgent:

    def __init__(self, state_space, action_space, network_type, seed, device='cpu'):

        self.GAMMA = 0.99            # discount factor

        '''Hyperparameters'''
        self.BUFFER_SIZE = None  # replay buffer size
        self.BATCH_SIZE = None        # minibatch size
        self.LR = None               # learning rate
        # how often to update the network (When Q target is present)
        self.UPDATE_EVERY = None

        self.device = device
        self.network_type = network_type

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = np.random.seed(seed)

        '''Hyperparameters for Agent'''
        self.eps_start = None
        self.eps_end = None
        self.decay_type = None
        self.eps_decay = None

        self.time_step = 0

        self.reset(seed)

    def reset(self, seed=0):
        if self.eps_start is not None:
            self.eps = self.eps_start
            self.time_step = 0

            '''DDQ Network'''
            self.qnetwork_local = DDQNetwork(self.state_size,
                                             self.action_size,
                                             self.network_type,
                                             seed).to(self.device)
            self.qnetwork_target = DDQNetwork(self.state_size,
                                              self.action_size,
                                              self.network_type,
                                              seed).to(self.device)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.LR
            )

            self.replay_memory = ReplayBuffer(self.BUFFER_SIZE,
                                              self.BATCH_SIZE,
                                              seed, self.device)

    def update_hyperparameters(self, **kwargs):
        '''The function updates hyper parameters overriding the
        default values
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self):
        if self.decay_type == 'linear':
            self.eps = max(self.eps_end, self.eps - self.eps_decay)
        elif self.decay_type == 'exponential':
            self.eps = max(self.eps_end, self.eps*self.eps_decay)

    def step(self, state, action, reward, next_state, done):

        self.replay_memory.add(state, action, reward, next_state, done)

        if len(self.replay_memory) >= self.BATCH_SIZE:
            experiences = self.replay_memory.sample()
            self.learn(experiences)

        self.time_step = (self.time_step + 1) % self.UPDATE_EVERY
        if self.time_step == 0:
            self.qnetwork_target.load_state_dict(
                self.qnetwork_local.state_dict())

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(
                state
            ).cpu().data.numpy().squeeze(0)
        self.qnetwork_local.train()
        action = epsilon_greedy(action_values, self.action_size, self.eps)
        return action, action_values

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(
            next_states
        ).detach().max(1)[0].unsqueeze(1)

        q_targets = rewards + (self.GAMMA * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
