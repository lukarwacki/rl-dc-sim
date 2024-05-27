import os
import sys
import time
import random
import numpy as np
import pandas as pd
import gym

import optuna
from optuna import Trial
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history

from simulation.RL_training.run_PPO import train_agent, evaluate_agent

from simulation.helpers.gym_make_environment import make_env
from simulation.helpers.energyplus_util import energyplus_arg_parser, energyplus_logbase_dir, weather_files_dir, it_load_files_dir, optuna_logbase_dir

def train_model(trial, args):
    # Reward settings
    args.use_reward_file = True
    args.reward_P_type = 'P_HVAC'
    args.reward_T_type = 'Gaussian'
    args.penalty_T_type = 'ReLU'
    args.penalty_a_fluctuation_type = 'Trapezoidal'
    args.lambda_T_r = .1
    args.lambda_T_p = .8
    args.fluct_T_sp = 0.1
    args.fluct_m_chill = 0.0
    args.fluct_m_air = 0.0
    args.T_SP = 24.0
    args.T_SP_bandwidth = 3.
    args.T_constr = 30.
    args.action_bandwidth = 1.

    # Create a wrapped environment
    env = make_env(args, is_hyper_param_optimization=True, trial_num=trial.number)
    
    # Train the agent
    trained_agent = train_agent(env, args)
    
    # Evaluate the agent
    rew_eval, _ = evaluate_agent(trained_agent, env.log_dir)

    # Close environment
    env.close()

    return rew_eval    

def objective_extensive_search(trial):
    # Parse arguments
    args = energyplus_arg_parser().parse_args()

    # Modify settings
    args.num_episodes = 0
    args.log_actions_states = False
    args.verbose_tb = True
    args.evaluate_agent = True

    # Hyperparameters to tune
    # args.learning_rate = trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True)
    args.learning_rate = 0.0007909848317679657
    args.learning_rate_schedule = 'linear'
    # args.gamma = trial.suggest_float("gamma", 0.85, 0.95)
    args.gamma = 0.85
    # args.GAE_lambda = trial.suggest_float("GAE_lambda", 0.94, 1)
    args.GAE_lambda = 0.9946797892895495
    # args.n_steps = trial.suggest_int("n_steps", 128, 1024, step=128)
    args.n_steps = 512
    # args.mini_batch_size = trial.suggest_int("mini_batch_size", 16, args.n_steps/2, step=16) # Must be <= n_steps
    args.mini_batch_size = 64
    # args.n_epochs = trial.suggest_int("n_epochs", 10, 25)
    args.n_epochs = 25
    # args.clip_range = trial.suggest_float("clip_range", 0.2, 0.5)
    args.clip_range = 0.4487117411242193
    args.clip_range_schedule = 'linear'
    args.KL_target = 0.1
    # args.ent_coef = trial.suggest_float("ent_coef", 0.002, 0.01)
    args.ent_coef = 0.006501481643366731
    # args.vf_coef = trial.suggest_float("vf_coef", 0.4, 0.9)
    args.vf_coef = 0.7478320995825463

    # Network architecture
    n_layers_pi = trial.suggest_int("n_layers_pi", 1, 4)
    net_arch_pi = []
    for i in range(n_layers_pi):
        n_neurons = trial.suggest_int(f"n_neurons_pi_{i}", 1, 64)
        net_arch_pi.append(n_neurons)

    n_layers_vf = trial.suggest_int("n_layers_vf", 1, 4)
    net_arch_vf = []
    for i in range(n_layers_vf):
        n_neurons = trial.suggest_int(f"n_neurons_vf_{i}", 1, 64)
        net_arch_vf.append(n_neurons)

    args.policy_kwargs = dict(net_arch=dict(pi=net_arch_pi, vf=net_arch_vf))

    # Train the model
    rew_eval = train_model(trial, args)

    return rew_eval

def objective_simple_search(trial):
    # Parse arguments
    args = energyplus_arg_parser().parse_args()

    # Modify settings
    args.num_episodes = 10
    args.log_actions_states = False
    args.verbose_tb = True
    args.evaluate_agent = True

    # Hyperparameters to tune
    lr_exp = trial.suggest_int("lr_exp", -5, -2)
    args.learning_rate = 10**lr_exp
    args.learning_rate_schedule = trial.suggest_categorical("learning_rate_schedule", ["constant", "linear"])
    args.gamma = trial.suggest_float("gamma", 0.9, 0.999, step=0.005)
    args.GAE_lambda = trial.suggest_float("GAE_lambda", 0.9, 1, step=0.01)
    args.n_steps = trial.suggest_int("n_steps", 512, 4096, step=512)
    args.mini_batch_size = trial.suggest_int("mini_batch_size", 128, args.n_steps, step=128) # Must be <= n_steps
    args.n_epochs = trial.suggest_int("n_epochs", 5, 30, step=5)
    args.clip_range = trial.suggest_float("clip_range", 0.1, 0.4, step=0.1)
    args.clip_range_schedule = trial.suggest_categorical("clip_range_schedule", ["constant", "linear"])
    args.KL_target = trial.suggest_float("KL_target", 0.1, 0.3, step=0.1)
    args.ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01, step=0.002)
    args.vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0, step=0.1)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_neurons = trial.suggest_int("n_neurons", 16, 128, step=16)
    args.policy_kwargs = dict(net_arch=dict(pi=[n_neurons]*n_layers, vf=[n_neurons]*n_layers))

    # Train the model
    rew_eval = train_model(trial, args)

    return rew_eval

if __name__=="__main__":
    # Search type
    # while True:
    #     search_type = input('Enter search type (simple/extensive): ')
    #     if search_type in ['simple', 'extensive']:
    #         break
    #     else:
    #         print("Invalid input. Please enter either 'simple' or 'extensive'.")

    # Define the unique name of the study
    # study_basename = input("Enter a short description as basename for your study: ").strip().lower()

    # study_name = f"{study_basename}_{time.strftime('%Y%m%d_%H%M%S')}"
    # study_name = "simple_search_20240410_212815"
    # study_name = "extensive_search_20240411_220951"
    search_type = 'extensive'
    study_name = "reward_check_2024_04_26"
    
    # Define the directory where the study will be saved
    optuna_log = optuna_logbase_dir()
    if not os.path.exists(optuna_log):
        os.makedirs(optuna_log)

    study_storage = f"sqlite:///{optuna_log}/{study_name}.db"

    # Create study
    study = optuna.create_study(direction="maximize", sampler=TPESampler(), study_name=study_name, storage=study_storage, load_if_exists=True)

    # Optimize the study
    if search_type == 'simple':
        study.optimize(objective_simple_search, n_trials=30, callbacks=[MaxTrialsCallback(30, states=(TrialState.COMPLETE,))])
    else:
        # Run for 100 trials every time this script is run
        study.optimize(objective_extensive_search, n_trials=200, callbacks=[MaxTrialsCallback(150, states=(TrialState.COMPLETE,))])

    print(f"Study name: {study_name}")
    print(f"Best trial:")
    print(study.best_trial)

    print("Done.")
    sys.exit(0)