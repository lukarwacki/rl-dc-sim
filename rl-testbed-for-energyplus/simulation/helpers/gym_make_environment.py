import sys
import os
import datetime
from mpi4py import MPI
import numpy as np
import json

from stable_baselines3 import PPO 
from stable_baselines3.common import logger
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

import gym
import gym.wrappers
from gym.envs.registration import register

# Custom imports
from simulation.gym_energyplus.envs import EnergyPlusEnv
import simulation.helpers.gym_wrappers as custom_wrappers
from simulation.helpers.gym_monitor import Monitor
from simulation.helpers.energyplus_util import (
    energyplus_arg_parser,
    energyplus_logbase_dir,
    tensorboard_logbase_dir,
    optuna_logbase_dir,
)


def make_env(args, seed_index=0, is_hyper_param_optimization=False, trial_num=0):
    # Create a new log name
    log_name = make_log_name(reward=args.reward_P_type ,episodes=args.num_episodes, is_hyper_param_optimization=is_hyper_param_optimization, trial_num=trial_num)

    # Generate the full path for the log directory
    if is_hyper_param_optimization:
        log_dir = os.path.join(optuna_logbase_dir(), log_name)
    else:
        log_dir = os.path.join(energyplus_logbase_dir(), log_name)
    
    tb_log_dir = tensorboard_logbase_dir()
    
    if not os.path.exists(log_dir + '/output'): 
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir

    # Select model and weather file
    model = os.getenv("ENERGYPLUS_MODEL")
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        sys.exit(1)
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        sys.exit(1)
    
    # Configure the logger for the process
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print("train: init logger with dir={}".format(log_dir)) #XXX
        log = logger.configure(log_dir)
    else:
        logger.configure(format_strings=[])
        logger.Logger.set_level()
    
    # Put the reward settings in the logging directory
    reward_file = write_reward_settings_json(args, log_dir)

    # Create and wrap the environment
    if args.use_reward_file is True:
        env = gym.make(args.env, framework=args.framework, reward_file=reward_file)
    else:
        env = gym.make(args.env, framework=args.framework)    

    if args.clip_action:
        env = gym.wrappers.ClipAction(env)
    
    if args.normalize_observation:
        env = custom_wrappers.NormalizeObservationMinMax(env, high=env.ep_model.obs_high, low=env.ep_model.obs_low)
        if args.transform_observation:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    
    # if args.normalize_reward:
    #     env = custom_wrappers.NormalizeRewardMinMax(env, gamma=args.gamma, low=env.ep_model.rew_low, high=env.ep_model.rew_high)
    #     # if args.transform_reward:
    #     #     env = gym.wrappers.TransformReward(env, lambda rew: np.clip(rew, -10, 10))
    
    env = Monitor(env, log.get_dir(), allow_early_resets=True)

    # Seeding
    seed = args.seed + seed_index
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Adding tb_log
    env.tb_log_dir=tb_log_dir
    env.log_name=log_name

    return env

def make_log_name(reward=None,episodes=None, is_hyper_param_optimization=False, trial_num=0):
    # Generate a timestamp for the log name
    log_name_general = datetime.datetime.now().strftime("SB3_%y_%m_%d_%Hh%M")

    if is_hyper_param_optimization:
        log_name = log_name_general + "_optuna_" + str(trial_num)
    else:
        # Ask for the type of run (random/training) with input validation
        while True:
            test_type = input("Enter the type of run you're doing [random/training]: ").strip().lower()
            if test_type in ["random", "training"]:
                break
            else:
                print("Invalid input. Please enter either 'random' or 'training'.")

        # Process based on the type of run
        if test_type == "random":
            test_description = input("Enter a description of your test: ")
            msg = f"random_{test_description}"
        else:  # For 'training' type
            if reward is None:
                reward = input("Describe the type of reward you're using: ")
            if episodes is None:
                episodes = input("For how many episodes will the training run: ")
            test_description = input("Enter a description of your test: ")
            msg = f"rew_{reward}_ep_{episodes}_{test_description}"

        # Combine all information to form the log name
        log_name = f"{log_name_general}_{msg}"

    return log_name

def write_reward_settings_json(args, log_dir):
    reward_file = os.path.join(log_dir, 'reward_settings.json')

    # Reward settings
    reward_settings = {
        # Reward and penalty functions
        'reward_P_type': args.reward_P_type,
        'reward_T_type': args.reward_T_type,
        'penalty_T_type': args.penalty_T_type,
        'penalty_a_fluctuation_type': args.penalty_a_fluctuation_type,
        # Reward and penalty weights
        'lambda_T_r': args.lambda_T_r,
        'lambda_T_p': args.lambda_T_p,
        'fluct_T_sp': args.fluct_T_sp,
        'fluct_m_chill': args.fluct_m_chill,
        'fluct_m_air': args.fluct_m_air,
        'T_SP': args.T_SP,
        'T_SP_bandwidth': args.T_SP_bandwidth,
        'T_constr': args.T_constr,
        'beta': args.beta,
        'softplus_shift': args.softplus_shift,
        'action_bandwidth': args.action_bandwidth 
    }

    # Write the reward settings to a JSON file
    with open(reward_file, 'w') as f:
        json.dump(reward_settings, f, indent=4)

    return reward_file