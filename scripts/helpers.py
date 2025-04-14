import numpy as np


def epsilon_greedy(action_values, action_size, eps):
    '''The function returns an action choice based on the
    epsilon-greedy policy derived from the action values
    '''
    if np.random.uniform(0, 1) <= eps:
        action = np.random.choice(np.arange(action_size))
        return action
    else:
        return np.argmax(action_values)


def softmax(action_values, action_size, tau):
    '''The function returns an action choice based on the
    softmax policy derived from the action values with 
    temperature control
    '''
    softmax_Q = np.exp((action_values - np.max(action_values))/tau)
    softmax_Q /= np.sum(softmax_Q)
    softmax_action = np.random.choice(a=np.arange(action_size),
                                      p=softmax_Q)
    return softmax_action


def moving_average(arr, n=100):
    """The function returns a rolling average of  scores over a window
    of size n
    """
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


def compute_decay(param_start, param_end, frac_episodes_to_decay, num_episodes, decay_type):
    """The function identifies the decay parameter to decay a parameter from 
    start to end in a fixed number of episodes.
    """
    if decay_type == 'linear':
        param_decay = ((param_start-param_end) /
                       (frac_episodes_to_decay*num_episodes))
    elif decay_type == 'exponential':
        param_decay = 10 ** (np.log10(param_end/param_start) /
                             (frac_episodes_to_decay*num_episodes))

    return param_decay
