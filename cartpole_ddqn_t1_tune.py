import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from scripts.ddqn_agent import DDQNAgent
from scripts.reinforce_agent import ReinforceMCwithoutBaselineAgent, ReinforceMCwithBaselineAgent
from scripts.training import Trainer, trainingInspector, test_agent, plot_test_results, compute_decay


def episode_trigger(x):
    if x % 200 == 0:
        return True
    return False


def process_hyperparameters_ddqn(hyperparameters):
    hyperparameters.update({
        "eps_decay": compute_decay(
            hyperparameters["eps_start"],
            hyperparameters["eps_end"],
            hyperparameters["frac_episodes_to_decay"],
            hyperparameters["num_episodes"],
            hyperparameters["decay_type"]
        )
    })

    hyperparameters.pop("frac_episodes_to_decay", None)
    return hyperparameters


def main():
    """Function setup to configure a sweep run, record videos of policy in action and
    log results in wandb
    """
    run = wandb.init()
    config = wandb.config

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    num_episodes = 1000
    max_return = 500
    ddqn_type1_hyperparameter_list = [

        process_hyperparameters_ddqn(_) for _ in [
            {
                "num_episodes": num_episodes,
                "max_return": max_return,
                "BUFFER_SIZE": int(5e6),
                "BATCH_SIZE": int(config.batch_size),
                "UPDATE_EVERY": int(config.update_every),
                "LR": float(config.learning_rate),
                "eps_start": 1,
                "eps_end": 0.005,
                "decay_type": "exponential",
                "frac_episodes_to_decay": float(config.frac_episodes_to_decay)
            }

        ]]
    
    run.name = repr(ddqn_type1_hyperparameter_list[0]).strip("{}")

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="backups/cartpole-ddqn-type1-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    ddqn_type1_agent = DDQNAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        network_type=1,
        device=device,
        seed=0
    )
    trainer = Trainer()
    ddqn_type1_results = test_agent(
        env, ddqn_type1_agent, trainer, ddqn_type1_hyperparameter_list, num_experiments=1)

    env.close()

    for score in ddqn_type1_results[0]["means"]:
        wandb.log(
            {
                "score": score
            }
        )
    for moving_avg in ddqn_type1_results[0]["rolling_means"]:
        wandb.log(
            {
                "mean_score": moving_avg
            }
        )

    wandb.log({
        "max_mean_score": np.max(ddqn_type1_results[0]["rolling_means"]),
        "regret": num_episodes*max_return - np.sum(ddqn_type1_results[0]["means"])
    })


if __name__ == '__main__':
    main()
