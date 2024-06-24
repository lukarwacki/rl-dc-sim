"""
Functions common across all algorithms.

"""

# Imports
import os
from stable_baselines3.common.evaluation import evaluate_policy

# Custom imports
from simulation.helpers.energyplus_util import overwrite_weather_file, overwrite_it_file
from simulation.helpers.gym_monitor import LoggingCallback, WriteSettingsCallback, TensorBoardCallback

def train_agent(env, args, algo):
    """
    Train the provided agent on the provided environment.

    Args:
        env: The environment to train the agent on.
        args: Additional arguments for training.

    Returns:
        The trained agent.
    """
    # Define agent
    agent = algo(env, args)

    # Define callbacks
    callbacks = [WriteSettingsCallback(env, args)]
    if args.log_actions_states:
        callbacks.append(LoggingCallback(env.log_dir, args, env.ep_model.state_names))
    if args.verbose_tb:
        callbacks.append(TensorBoardCallback())
    
    # Train agent
    ep_length = 52_848
    num_episodes = args.num_episodes
    num_timesteps = int(num_episodes * ep_length)
    agent.learn(total_timesteps=num_timesteps, tb_log_name=env.log_name, callback=callbacks)
    
    # Save the model
    agent.save(os.path.join(env.log_dir, 'agent'))
    
    return agent


def evaluate_agent(agent, log_dir):
    """
    Evaluate the agent's performance on the environment.

    Args:
        agent (object): The RL agent to be evaluated.
        log_dir (str): The directory where the evaluation results will be logged.

    Returns:
        tuple: A tuple containing the mean reward and standard deviation of the reward
               for the evaluated episodes.
    """
    n_evals = 1     # Test for just one episode since the environment is the same anyways due to seeding
    env = agent.get_env()

    # Overwrite weather and it files in logging
    overwrite_weather_file(log_dir,'Weather_amsterdam_2022.epw')
    overwrite_it_file(log_dir, 'IT_Load_4.csv')

    # Evaluate the policy
    mean_rew, std_rew = evaluate_policy(model=agent, env=env, n_eval_episodes=n_evals)

    # Print the results
    print(f"The mean reward of the {n_evals} tested episodes is: {mean_rew}")
    print(f"The std of the reward of {n_evals} tested episodes is: {std_rew}")

    return mean_rew, std_rew
