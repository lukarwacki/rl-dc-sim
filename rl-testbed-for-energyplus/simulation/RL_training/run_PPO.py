# Imports
import sys
import os
import time
import datetime
import shutil
from mpi4py import MPI
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO 
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Custom imports
from simulation.helpers.gym_make_environment import make_env
from simulation.helpers.energyplus_util import energyplus_arg_parser, energyplus_logbase_dir, weather_files_dir, it_load_files_dir, copy_file
from simulation.gym_energyplus.envs import EnergyPlusEnv
from simulation.helpers.gym_monitor import Monitor, LoggingCallback, WriteSettingsCallback, TensorBoardCallback
from postprocessing.tools.episode_visualization import run_gui

# Function definitions
def train_agent(env, args):
    """
    Train the PPO agent on the provided environment.

    Args:
        env: The environment to train the agent on.
        args: Additional arguments for training.

    Returns:
        The trained agent.
    """
    # Define agent
    agent = create_PPO(env, args)

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

def create_PPO(env, args):
    """
    Create a PPO agent with the provided environment and hyperparameters.

    Parameters:
        env (gym.Env): The environment to train the agent on.
        args (argparse.Namespace): The command-line arguments containing hyperparameters.

    Returns:
        PPO: The created PPO agent.
    """
    # Check if schedules are used 
    if args.learning_rate_schedule == 'constant':
        learning_rate = args.learning_rate
    elif args.learning_rate_schedule == 'linear':
        learning_rate = lambda f: f * args.learning_rate
    else:
        raise ValueError(f"Invalid learning rate schedule: {args.learning_rate_schedule}, should be either 'constant' or 'linear'")
    
    if args.clip_range_schedule == 'constant':
        clip_range = args.clip_range
    elif args.clip_range_schedule == 'linear':
        clip_range = lambda f: f * args.clip_range
    else:
        raise ValueError(f"Invalid clip range schedule: {args.clip_range_schedule}, should be either 'constant' or 'linear'")

    # Define the agent
    agent = PPO(
        # General
        env=env,
        verbose=1,
        tensorboard_log=env.tb_log_dir,
        seed=args.seed,
        stats_window_size=args.stats_window_size,
        # Policy related
        policy=args.policy,
        learning_rate=learning_rate,
        policy_kwargs=args.policy_kwargs,
        n_epochs=args.n_epochs,
        # RL related
        gamma=args.gamma,
        clip_range=clip_range,
        clip_range_vf=args.clip_range_vf,
        target_kl=args.KL_target,
        n_steps=args.n_steps,
        batch_size=args.mini_batch_size,
        gae_lambda=args.GAE_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef
    )
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

    # Add weather file to the logging directory to override the standard weather files
    weather_dir = weather_files_dir()
    weather_filename = 'Weather_amsterdam_2022.epw'
    weather_location = os.path.join(weather_dir, weather_filename)
    copy_file(log_dir, weather_location)

    # Add IT load file to the logging directory to override the standard IT load files
    it_load_dir = it_load_files_dir()
    it_load_filename = 'IT_Load_4.csv'
    it_load_location = os.path.join(it_load_dir, it_load_filename)
    copy_file(log_dir, it_load_location)
    copied_it_load_filename = os.path.join(log_dir, it_load_filename)
    shutil.move(copied_it_load_filename, os.path.join(log_dir, 'IT_load.csv'))

    # Evaluate the policy
    mean_rew, std_rew = evaluate_policy(model=agent, env=env, n_eval_episodes=n_evals)

    # Print the results
    print(f"The mean reward of the {n_evals} tested episodes is: {mean_rew}")
    print(f"The std of the reward of {n_evals} tested episodes is: {std_rew}")

    return mean_rew, std_rew

def main():
    """
    This function handles the training of the PPO agent on the EnergyPlus environment. 
    First arguments are parsed, and can be modified to the desired settings. Then, an 
    enviroment is created, the agent is trained, and finally the agent is evaluated.
    """
    # Parse arguments
    args = energyplus_arg_parser().parse_args()

    # Modify settings
    args.num_episodes = 5                           # Number of episodes to train the agent
    args.log_actions_states = False                 # Log the normalized actions and states during training (only for debugging purposes)
    args.evaluate_agent = True                      # Do an evaluation run on a new weather & IT load file after training
    args.seed = 1                                   # Seed for reproducibility

    # Modify hyperparams (now set to tuned hyperparameters)
    args.learning_rate = 0.0007909848317679657      # Learning rate of the NNs in the PPO
    args.learning_rate_schedule = 'linear'          # Learning rate schedule over the training iterations
    args.n_steps = 512                              # Number of steps to collect samples for each training iteration
    args.mini_batch_size = 48                       # Size of the mini-batch for each training iteration of the NNs
    args.n_epochs = 25                              # Number of epochs to train the NNs for each training iteration
    args.gamma = 0.8525366263101639                 # Discount factor
    args.GAE_lambda = 0.9946797892895495            # Factor for the Generalized Advantage Estimation
    args.clip_range = 0.4487117411242193            # Clipping range for the PPO
    args.clip_range_schedule = 'linear'             # Clipping range schedule over the training iterations
    args.clip_range_vf = None                       # Clipping range for the value function (StableBaselines specific PPO factor)
    args.KL_target = 0.1                            # Target for the Kullback-Leibler divergence approximation 
    args.policy_kwargs = dict(net_arch=dict(pi=[28],vf=[18]))   # Architecture of the NNs in the PPO
    args.ent_coef = 0.006501481643366731            # Entropy coefficient for the PPO
    args.vf_coef = 0.7478320995825463               # Value function coefficient for the PPO

    # Reward settings
    args.use_reward_file = True                     # True if reward settings are defined here, False if reward settings are hardcoded in the environment (old version)
    args.reward_P_type = 'P_HVAC'                   # Type of reward for the power consumption, can be: 'P_HVAC', 'PUE', None
    args.reward_T_type = 'Gaussian'                 # Type of reward for the leaving CRAH temperature, can be: 'Gaussian', None
    args.penalty_T_type = 'ReLU'                    # Type of penalty for the leaving server temperature, can be: 'ReLU', 'ReLU2', 'Softplus', None
    args.penalty_a_fluctuation_type = 'Trapezoidal' # Type of penalty for the action fluctuation, can be: 'Trapezoidal', 'Linear', 'Quadratic', None
    args.lambda_T_r = .1                            # Weight of the reward for the leaving CRAH temperature
    args.lambda_T_p = .8                            # Weight of the penalty for the leaving server temperature
    args.fluct_T_sp = 0.1                           # Weight of the penalty for the chilled water setpoint fluctuation
    args.fluct_m_chill = 0.0                        # Weight of the penalty for the chilled water mass flow fluctuation
    args.fluct_m_air = 0.0                          # Weight of the penalty for the air mass flow fluctuation
    args.T_SP = 24.0                                # Setpoint for the leaving CRAH temperature     
    args.T_SP_bandwidth = 3.                        # Allowable temperature deviation for the leaving CRAH temperature setpoint
    args.T_constr = 30.                             # Maximum allowable leaving server air temperature
    args.action_bandwidth = 1.                      # Maximum allowable action fluctuation (for the trapezoidal penalty only)

    # Create a wrapped environment
    env = make_env(args)
    
    # Train the agent
    trained_agent = train_agent(env, args)
    
    # Evaluate the agent
    if args.evaluate_agent:
        rew_eval, _ = evaluate_agent(trained_agent, env.log_dir)

    # Close environment
    env.close()

    return trained_agent

if __name__ == '__main__':
    main()
    sys.exit()  # Exit the program