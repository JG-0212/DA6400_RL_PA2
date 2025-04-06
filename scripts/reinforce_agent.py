import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical
from collections import deque, namedtuple

from scripts.helpers import epsilon_greedy, softmax, compute_decay


class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        self.network = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, observation):

        probs = self.network(observation)
        return probs


class ValueNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        self.network = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, observation):

        state_value = self.network(observation)
        return state_value


class ReinforceMCwithoutBaselineAgent:

    def __init__(self, state_space, action_space, seed, device='cpu'):

        self.GAMMA = 0.99            # discount factor

        '''Hyperparameters'''
        self.LR_POLICY = 5e-4               # learning rate for policy

        self.device = device

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = np.random.seed(seed)

        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )

        self.episode_history = []

        self.reset(seed)

    def reset(self, seed=0):
        '''Network'''
        self.policy_network = PolicyNetwork(self.state_size,
                                            self.action_size,
                                            seed).to(self.device)

        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.LR_POLICY
        )

    def update_hyperparameters(self, **kwargs):
        '''The function updates hyper parameters overriding the
        default values
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self, episode_history):

        discounted_returns = []
        G = 0
        for step in reversed(self.episode_history):
            G = step.reward + self.GAMMA * G
            discounted_returns.append(G)

        discounted_returns.reverse()

        for step, G in zip(self.episode_history, discounted_returns):
            state, action, reward = step.state, step.action, step.reward
            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)

            action_probs = self.policy_network(state)
            action_dist = Categorical(action_probs)
            log_prob = action_dist.log_prob(torch.tensor(
                action, dtype=torch.long, device=self.device))

            loss = -log_prob*G

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.episode_history.clear()

    def step(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.episode_history.append(e)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)

        self.policy_network.eval()
        with torch.no_grad():
            action_probs = self.policy_network(state)
        self.policy_network.train()

        m = Categorical(action_probs)

        action = m.sample().item()
        return action, None


class ReinforceMCwithBaselineAgent:

    def __init__(self, state_space, action_space, seed, device='cpu'):

        self.GAMMA = 0.99            # discount factor

        '''Hyperparameters'''
        self.LR_POLICY = 5e-4               # learning rate for policy
        self.LR_VALUE = 5e-3              # learning rate for value

        self.device = device

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = np.random.seed(seed)

        self.reset(seed)

    def reset(self, seed=0):
        self.eps = self.eps_start

        '''Network'''
        self.policy_network = PolicyNetwork(self.state_size,
                                            self.action_size,
                                            seed).to(self.device)
        self.value_network = ValueNetwork(self.state_size,
                                          self.action_size,
                                          seed).to(self.device)
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.LR_POLICY
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), lr=self.LR_VALUE
        )

    def update_hyperparameters(self, **kwargs):
        '''The function updates hyper parameters overriding the
        default values
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self, episode_history):

        G = [0]
        for i in range(len(episode_history)-1, -1, -1):
            G.append(self.GAMMA*G[-1]+episode_history[i][-1])

        G.reverse()
        G.pop()

        for i in range(len(episode_history)):
            mc_return = G[i]
            state, action, reward = episode_history[i]

            state_value = self.value_network(state)
            mc_error = mc_return - state_value
            action_probs = self.policy_network(state)[action]
            action_log_prob = torch.log(action_probs)

            policy_loss = -mc_error*((self.GAMMA)**i)*action_log_prob
            value_loss = F.mse_loss(mc_return, state_value)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.network_optimizer.zero_grad()
            value_loss.backward()

            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)

            for param in self.value_network.parameters():
                param.grad.data.clamp_(-1, 1)

            self.policy_optimizer.step()
            self.value_optimizer.step()

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)

        self.policy_network.eval()
        with torch.no_grad():
            action_probs = self.policy_network(state)
        self.policy_network.train()

        m = Categorical(action_probs)

        action = m.sample()
        return action, None
