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

    def __init__(self, state_size, action_size, seed, hidden_layer_sizes=[64, 64, 64]):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        layers = []
        input_size = state_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))
        layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, observation):

        probs = self.network(observation)
        return probs


class ValueNetwork(nn.Module):

    def __init__(self, state_size, seed, hidden_layer_sizes=[64, 64, 64]):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        layers = []
        input_size = state_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

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
        self.episode_history.clear()

        '''Network'''
        self.policy_network = PolicyNetwork(
            self.state_size,
            self.action_size,
            seed,
            hidden_layer_sizes=[64, 32]
        ).to(self.device)

        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.LR_POLICY
        )

    def update_hyperparameters(self, **kwargs):
        '''The function updates hyper parameters overriding the
        default values
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self):

        discounted_returns = []
        G = 0
        for step in reversed(self.episode_history):
            G = step.reward + self.GAMMA * G
            discounted_returns.append(G)

        discounted_returns.reverse()

        states = torch.tensor(
            np.array([e.state for e in self.episode_history]),
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            np.array([e.action for e in self.episode_history]),
            dtype=torch.long,
            device=self.device
        )
        returns = torch.tensor(
            np.array(discounted_returns),
            dtype=torch.float32,
            device=self.device
        )

        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        discounts = self.GAMMA**torch.arange(0, len(self.episode_history),
                                             dtype=torch.float32,
                                             device=self.device)

        policy_loss = -(discounts * returns * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        for param in self.policy_network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

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
        self.UPDATE_EVERY = 20

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
        self.episode_history.clear()
        self.time_step = 0

        '''Network'''
        self.policy_network = PolicyNetwork(
            self.state_size,
            self.action_size,
            seed,
            hidden_layer_sizes=[64, 64, 64]
        ).to(self.device)

        self.value_network_local = ValueNetwork(
            self.state_size,
            seed,
            hidden_layer_sizes=[64, 64]
        ).to(self.device)

        self.value_network_target = ValueNetwork(
            self.state_size,
            seed,
            hidden_layer_sizes=[64, 64]
        ).to(self.device)

        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.LR_POLICY
        )
        self.value_optimizer = optim.Adam(
            self.value_network_local.parameters(), lr=self.LR_VALUE
        )

    def update_hyperparameters(self, **kwargs):
        '''The function updates hyper parameters overriding the
        default values
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self):
        discounted_returns = []
        G = 0
        for step in reversed(self.episode_history):
            G = step.reward + self.GAMMA * G
            discounted_returns.append(G)
        discounted_returns.reverse()

        states = torch.tensor(
            np.array([e.state for e in self.episode_history]),
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            np.array([e.action for e in self.episode_history]),
            dtype=torch.long,
            device=self.device
        )
        rewards = torch.tensor(
            np.array([e.reward for e in self.episode_history]),
            dtype=torch.float32,
            device=self.device
        )
        next_states = torch.tensor(
            np.array([e.next_state for e in self.episode_history]),
            dtype=torch.float32,
            device=self.device
        )
        dones = torch.tensor(
            np.array([e.done for e in self.episode_history]),
            dtype=torch.float32,
            device=self.device
        )
        returns = torch.tensor(
            np.array(discounted_returns),
            dtype=torch.float32,
            device=self.device
        )

        state_values = self.value_network_local(states).squeeze()
        next_state_values = self.value_network_target(
            next_states
        ).detach().squeeze()

        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        discounts = self.GAMMA**torch.arange(0, len(self.episode_history),
                                             dtype=torch.float32,
                                             device=self.device)

        td_targets = rewards + self.GAMMA*next_state_values*(1-dones)
        value_loss = F.mse_loss(state_values, td_targets)
        advantages = returns - state_values.detach()
        policy_loss = -(discounts * advantages * log_probs).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        for param in self.value_network_local.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        for param in self.policy_network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        self.episode_history.clear()

        self.time_step = (self.time_step + 1)%self.UPDATE_EVERY
        if self.time_step == 0:
            self.value_network_target.load_state_dict(
                self.value_network_local.state_dict()
            )

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
