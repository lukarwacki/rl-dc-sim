"""
Helpers for script run_energyplus.py.
"""
import os
import glob
import shutil
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
# following import necessary to register EnergyPlus-v0 env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for run_energyplus.py.
    """
    parser = arg_parser()

    # General arguments
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--seed', '-s', help='RNG seed', type=int, default=0)
    parser.add_argument('--save-interval', type=int, default=int(0))
    parser.add_argument('--model-pickle', help='model pickle', type=str, default='')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    parser.add_argument('--evaluate_agent', help='Evaluate the agent after training', type=bool, default=False)
    parser.add_argument('--framework', help='framework to use', type=str, default='openai')

    # Training settings
    parser.add_argument('--num_episodes', help='The number of episodes', type=int, default=1)
    parser.add_argument('--tb_log_name', help='Directory where tensorboard is logging', type=str, default='tb_log')
    parser.add_argument('--log_actions_states', help='Should all actions and states be logged', type=bool, default=False)
    parser.add_argument('--verbose_tb', help='Should TensorBoard have verbose graphs or not', type=bool, default=True)
    parser.add_argument('--stats_window_size', help='Amount of episode to show in tb', type=int, default=1)

    # Normalization settings
    parser.add_argument('--clip_action', help='If actions should be clipped to the action space or not', type=bool, default=True)
    parser.add_argument('--normalize_observation', help='If observations should be normalized or not', type=bool, default=True)
    parser.add_argument('--transform_observation', help='If an observation should be clipped after nomalization or not', type=bool,default=True)
    parser.add_argument('--normalize_reward', help='If a reward should be normalized or not', type=bool,default=True)
    parser.add_argument('--transform_reward', help='If a reward should be clipped after normalization', type=bool, default=True)

    # Agent settings
    parser.add_argument('--policy', help='The policy the agent uses', type=str, default='MlpPolicy')
    parser.add_argument('--policy_kwargs', help='Other keyword arguments to be passed to the policy', type=dict, default=None)
    
    parser.add_argument('--learning_rate_schedule', help='The learning rate schedule', type=str, default='constant')
    parser.add_argument('--learning_rate', help="The learning rate of the policy and value function", type=float, default=3e-4)
    parser.add_argument('--gamma', help='The reward discount factor during learning', type=float, default=0.99)
    
    parser.add_argument('--clip_range_schedule', help='The schedule for the clipping range', type=str, default='constant')
    parser.add_argument('--clip_range', help='The PPO clipping parameter', type=float, default=0.2)
    parser.add_argument('--clip_range_vf', help='Clipping parameter for the value function,', type=Optional[float], default=None)
    parser.add_argument('--KL_target', help='Limit the KL divergence between updates', type=Optional[float], default=None)

    parser.add_argument('--n_steps', help='Amount of steps in the replay buffer', type=int, default=2048)
    parser.add_argument('--mini_batch_size', help='Size of the minibatches', type=int, default=64)
    parser.add_argument('--n_epochs', help='Number of epoch when optimizing the surrogate loss', type=int, default=10)
    
    parser.add_argument('--GAE_lambda', help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator', type=float, default=0.95)
    parser.add_argument('--ent_coef', help='The entropy coefficient for the loss calculation', type=float, default=0.0)
    parser.add_argument('--vf_coef', help='The value function coefficient for the loss calculation', type=float, default=0.5)
    
    # Reward settings
    parser.add_argument('--use_reward_file', help='If a reward file should be used', type=bool, default=False)
    parser.add_argument('--reward_P_type', help='The type of reward for the power consumption', type=str, default='P_HVAC')
    parser.add_argument('--reward_T_type', help='The type of reward for the temperature', type=str, default='Gaussian')
    parser.add_argument('--penalty_T_type', help='The type of penalty for the temperature', type=str, default='ReLU')
    parser.add_argument('--penalty_a_fluctuation_type', help='The type of penalty for the action fluctuation', type=str, default='Trapezoidal')
    parser.add_argument('--lambda_T_r', help='The weight of the temperature reward', type=float, default=1.0)
    parser.add_argument('--lambda_T_p', help='The weight of the temperature penalty', type=float, default=1.0)
    parser.add_argument('--fluct_T_sp', help='The weight of the temperature setpoint fluctuation', type=float, default=0.1)
    parser.add_argument('--fluct_m_chill', help='The weight of the chiller mass flow fluctuation', type=float, default=0.0)
    parser.add_argument('--fluct_m_air', help='The weight of the air mass flow fluctuation', type=float, default=0.0)
    parser.add_argument('--T_SP', help='The temperature setpoint', type=float, default=24.0)
    parser.add_argument('--T_SP_bandwidth', help='The temperature setpoint bandwidth', type=float, default=3.)
    parser.add_argument('--T_constr', help='The temperature constraint', type=float, default=30.)
    # parser.add_argument('--T_constr_op', help='The operating temperature constraint', type=float, default=27.)
    # parser.add_argument('--T_constr_serv', help='The service agreement temperature constraint', type=float, default=30.)
    parser.add_argument('--beta', help='The beta parameter for the penalty function', type=float, default=1.0)
    parser.add_argument('--softplus_shift', help='The shift for the softplus function', type=float, default=3.0)
    parser.add_argument('--action_bandwidth', help='The bandwidth for the action fluctuation penalty', type=float, default=.5)

    return parser


def energyplus_locate_log_dir(index=0):
    pat_openai = energyplus_logbase_dir() + f'/openai-????-??-??-??-??-??-??????*/progress.csv'
    pat_ray = energyplus_logbase_dir() + f'/ray-????-??-??-??-??-??-??????*/*/progress.csv'
    pat_stablebaselines3 = energyplus_logbase_dir() + f'Stable-Baselines3-????-??-??-??:??:??*/*/progress.csv'
    files = [
        (f, os.path.getmtime(f))
        for pat in [pat_openai, pat_ray, pat_stablebaselines3]
        for f in glob.glob(pat)
    ]
    newest = sorted(files, key=lambda files: files[1])[-(1 + index)][0]
    dir = os.path.dirname(newest)
    # in ray, progress.csv is in a subdir, so we need to get
    # one step upper.
    if "/ray-" in dir:
        dir = os.path.dirname(dir)
    print('energyplus_locate_log_dir: {}'.format(dir))
    return dir


def energyplus_logbase_dir():
    logbase_dir = os.getenv('ENERGYPLUS_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir

def tensorboard_logbase_dir():
    logbase_dir = os.getenv('TENSORBOARD_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir

def optuna_logbase_dir():
    logbase_dir = os.getenv('OPTUNA_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir

def weather_files_dir():
    weather_dir = os.getenv('WEATHER_DIR')
    if weather_dir is None:
        raise ValueError("The weather directory can not be found")
    return weather_dir

def it_load_files_dir():
    it_load_dir = os.getenv('P_ITE_DIR')
    if it_load_dir is None:
        raise ValueError("The IT load directory can not be found")
    return it_load_dir

def copy_file(log_dir, source_file):
    # Check if the source file exists
    if not os.path.exists(source_file):
        print(f"Source file '{source_file}' does not exist.")
        return
    
    # Check if the log_dir exists
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist.")
        return
    
    # Get the base filename (without the path)
    file_name = os.path.basename(source_file)
    
    # Destination path
    destination_path = os.path.join(log_dir, file_name)
    
    # Check if the file already exists in the log directory
    if os.path.exists(destination_path):
        print(f"'{file_name}' already exists in the log directory.")
    else:
        shutil.copy(source_file, destination_path)
        print(f"Copied '{file_name}' to '{log_dir}'.")


def overwrite_weather_file(log_dir, fn = "Weather_amsterdam_2022.epw"):
    # Add weather file to the logging directory to override the standard weather files
    weather_location = os.path.join(weather_files_dir(), fn)
    copy_file(log_dir, weather_location)

def overwrite_it_file(log_dir, fn = "IT_Load_4.csv"):
    # Add IT load file to the logging directory to override the standard IT load files
    it_load_dir = it_load_files_dir()
    it_load_location = os.path.join(it_load_dir, fn)
    copy_file(log_dir, it_load_location)
    copied_it_load_filename = os.path.join(log_dir, fn)
    shutil.move(copied_it_load_filename, os.path.join(log_dir, 'IT_load.csv'))
