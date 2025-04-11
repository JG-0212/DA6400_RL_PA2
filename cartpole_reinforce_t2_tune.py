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


def main():
    """Function setup to configure a sweep run, record videos of policy in action and
    log results in wandb
    """
    run = wandb.init()
    config = wandb.config

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    num_episodes = 2000
    max_return = 500
    reinforce_type2_hyperparameter_list = [
        {
            "num_episodes": num_episodes,
            "max_return": max_return,
            "LR_POLICY": float(config.learning_rate_policy),
            "LR_VALUE": float(config.learning_rate_policy)*float(config.multiplier),
            "UPDATE_EVERY": int(config.update_every)
        }
    ]

    run.name = repr(reinforce_type2_hyperparameter_list[0]).strip("{}")

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="backups/cartpole-reinforce-type2-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    reinforce_type2_agent = ReinforceMCwithBaselineAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        seed=0
    )

    trainer = Trainer()
    reinforce_type2_results = test_agent(
        env, reinforce_type2_agent, trainer, reinforce_type2_hyperparameter_list, num_experiments=3)

    env.close()

    for score in reinforce_type2_results[0]["means"]:
        wandb.log(
            {
                "score": score
            }
        )
    for moving_avg in reinforce_type2_results[0]["rolling_means"]:
        wandb.log(
            {
                "mean_score": moving_avg
            }
        )

    wandb.log({
        "max_mean_score": np.max(reinforce_type2_results[0]["rolling_means"]),
        "regret": num_episodes*max_return - np.sum(reinforce_type2_results[0]["means"])
    })


if __name__ == '__main__':
    main()
