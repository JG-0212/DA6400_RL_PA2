from tqdm import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt

from scripts.ddqn_agent import DDQNAgent
from scripts.reinforce_agent import ReinforceMCwithBaselineAgent, ReinforceMCwithoutBaselineAgent


class Trainer:

    def training(self, env, agent, n_episodes=10000, process_training_info=lambda *args, **kwargs: (False, {})):
        """
        To train an agent in the given environment.

        Args:
            - env: The environment for training.
            - agent: An agent with `.step()`, `.act()`, and `.update_agent_parameters()`.
            - n_episodes (int, optional): Number of training episodes. Defaults to 10000.
            - process_training_info (function, optional): Runs after each episode.
                - First return value must be a `bool` for early stopping.  
                - Second return value must be a `dict` to update the progress bar's postfix.

        Returns:
            - dict: Summary of the training process.
        """

        begin_time = datetime.datetime.now()

        history_scores = []
        history_total_rewards = []
        history_termination = []
        history_truncation = []

        progress_bar = tqdm(range(1, n_episodes+1), desc="Training")

        for i_episode in progress_bar:
            state, _ = env.reset()
            score = 0
            total_reward = 0
            terminated, truncated = False, False
            episode_history = []

            while not (terminated or truncated):
                action, action_vals = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_history.append((state, action, reward))
                done = (terminated or truncated)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += self.compute_score(reward)
                total_reward += reward

            agent.update_agent_parameters(episode_history)

            history_scores.append(score)
            history_total_rewards.append(score)
            history_termination.append(terminated)
            history_truncation.append(truncated)

            early_stop, info = process_training_info(
                agent,
                history_scores,
                history_termination,
                history_truncation,
                episode_history
            )

            if info:
                progress_bar.set_postfix(info)

            if early_stop:
                break

        end_time = datetime.datetime.now()
        return {
            "computation_time": end_time - begin_time,
            "scores": np.array(history_scores),
            "total_rewards": np.array(history_total_rewards)
        }

    def compute_score(self, reward):
        return reward


class trainingInspector:

    def __init__(self, max_return):
        """To inspect an agent's performance during training

        The function self.process_training_info runs after every episode and 
        a can be used to send a signal for early stopping and update the
        progress bar during training

        Args:
            - max_return (float): The maximum return of an epsiode.
        """

        self.max_mean_score = None
        self.regret = 0
        self.max_return = max_return

    def process_training_info(self, agent, scores, termination, truncation, episode_history):
        """Processes training info. Returns a signal for succesful solving of the
        environment and moving average of scores in the past 100 episodes.
        """
        mean_scores = np.array(scores[max(0, len(scores)-100):]).mean()
        if mean_scores >= self.max_return:
            return False, {"Mean Score": mean_scores}
        return False, {"Mean Score": mean_scores}


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


def test_agent(env, agent, trainer, hyperparameter_list, num_experiments=5):
    """To test agents and compute metrics
    Args:
        - env: The environment for training.
        - agent: An agent with `.step()`, `.act()`, and `.update_agent_parameters()`.
        - trainer: Training function to satisfy a given environment's needs
        - hyperparameter_list: Set of hyperparameters for which we wish to evaluate
        - num_experiments: Number of realisations to determine expected performance

    Returns:
        - a dictionary of metrics that comprises of mean and standard deviance of scores
        and the rolling average all averaged over num_experiments
    """

    test_results = []

    for test_num, test_hyperparameters in enumerate(hyperparameter_list):

        result_history = {
            "experiment": [],
            "scores": [],
            "moving_average_scores": []
        }

        for experiment in range(1, num_experiments+1):
            hyperparameters = test_hyperparameters.copy()
            hyperparameters.pop("num_episodes", None)
            hyperparameters.pop("max_return", None)
            agent.update_hyperparameters(**hyperparameters)

            max_return = test_hyperparameters["max_return"]
            num_episodes = test_hyperparameters["num_episodes"]

            if isinstance(agent, DDQNAgent):
                label = f"DDQN Type {agent.network_type}"
            elif isinstance(agent, ReinforceMCwithoutBaselineAgent):
                label = "REINFORCE MC (without baseline)"
            elif isinstance(agent, ReinforceMCwithBaselineAgent):
                label = "REINFORCE MC (with baseline)"

            ti = trainingInspector(max_return)

            results = trainer.training(
                env, agent,
                n_episodes=num_episodes,
                process_training_info=ti.process_training_info)

            result_history["scores"].append(results["scores"])
            result_history["moving_average_scores"].append(
                moving_average(results["scores"]))

        result_history["scores"] = np.array(result_history["scores"])
        result_history["moving_average_scores"] = np.array(
            result_history["moving_average_scores"])

        metrics = {
            "label": label + f" hyperparams {test_num + 1}",
            "episodes": range(1, num_episodes+1),
            "rolling_episodes": range(1, result_history["moving_average_scores"].shape[1] + 1),
            "means": result_history["scores"].mean(axis=0),
            "std_dev": result_history["scores"].std(axis=0),
            "rolling_means": result_history["moving_average_scores"].mean(axis=0),
            "rolling_std_dev": result_history["moving_average_scores"].std(axis=0)
        }

        test_results.append(metrics)

    return test_results


def plot_test_results(test_results, experiments):
    """
    Utility function to plot results
    """
    plt.subplots(1, 2, figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.title("Scores vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    for i in experiments:
        label = test_results[i]["label"]
        episodes = test_results[i]["episodes"]
        means = test_results[i]["means"]
        std_dev = test_results[i]["std_dev"]

        plt.plot(episodes, means, linewidth=0.2, label=label)
        plt.fill_between(episodes, means-std_dev, means+std_dev, alpha=0.6)

    plt.legend()

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.title("Rolling means of scores vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Rolling means of scores")
    for i in experiments:
        label = test_results[i]["label"]
        rolling_episodes = test_results[i]["rolling_episodes"]
        rolling_means = test_results[i]["rolling_means"]
        rolling_std_dev = test_results[i]["rolling_std_dev"]

        plt.plot(rolling_episodes, rolling_means, linewidth=1, label=label)
        plt.fill_between(rolling_episodes, rolling_means -
                         rolling_std_dev, rolling_means+rolling_std_dev, alpha=0.4)

    plt.legend()
    plt.tight_layout()
