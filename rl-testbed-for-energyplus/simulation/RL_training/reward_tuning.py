import os
import sys
import time
import random
import json

import numpy as np
import pandas as pd

import gym

import optuna
from optuna import Trial
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history

from simulation.helpers.energyplus_util import energyplus_arg_parser, energyplus_logbase_dir, weather_files_dir, it_load_files_dir, optuna_logbase_dir
from postprocessing.tools.energyplus_analyzer import EnergyPlusAnalysis
from simulation.helpers.gym_make_environment import make_env
from simulation.helpers.constraint_metrics import temperature_constraint_analysis, action_analysis, setpoint_analysis

from simulation.RL_training.run_PPO import train_agent, evaluate_agent

def setup_model_args(trial):
    # Parse standard arguments
    args = energyplus_arg_parser().parse_args()

    # Set optimal hyper parameters
    args.num_episodes = 10
    args.log_actions_states = False
    args.verbose_tb = True
    args.evaluate_agent = True

    # Set hyperparameters
    args.learning_rate = 0.0007909848317679657
    args.learning_rate_schedule = 'linear'
    args.gamma = 0.8525366263101639
    args.GAE_lambda = 0.9946797892895495
    args.n_steps = 512
    args.mini_batch_size = 48
    args.n_epochs = 25
    args.clip_range = 0.4487117411242193
    args.clip_range_schedule = 'linear'
    args.KL_target = 0.1
    args.ent_coef = 0.006501481643366731
    args.vf_coef = 0.7478320995825463

    arch_pi = [28]
    arch_vf = [18]
    args.policy_kwargs = dict(net_arch=dict(pi=arch_pi, vf=arch_vf))

    # SET REWARD PARAMETERS
    args.use_reward_file = True

    # Power reward
    args.reward_P_type = 'P_HVAC'
    
    # T_setpoint reward
    args.T_SP = 24.0    # Temperature setpoint value
    
    args.lambda_T_r = trial.suggest_float('lambda_T_r', 0, 1.5)
    args.reward_T_type = 'Gaussian'
    args.T_SP_bandwidth = trial.suggest_float('T_SP_bandwidth', 0, 5)
    
    # T_constraint reward
    args.T_constr = 30.0   # Temperature constraint value

    args.lambda_T_p = trial.suggest_float('lambda_T_p', 0, 3)
    args.penalty_T_type = trial.suggest_categorical('penalty_T_type', ['ReLU', 'Softplus', 'ReLU2'])
    if args.penalty_T_type == 'Softplus':
        args.beta = trial.suggest_float('beta_softplus', 0, 10)
        args.softplus_shift = 0.0


    # Action reward
    args.penalty_a_fluctuation_type = trial.suggest_categorical('penalty_a_fluctuation_type', ['Trapezoidal', 'Linear', 'Quadratic'])
    args.fluct_T_sp = trial.suggest_float('fluct_T_sp', 0, 0.5)
    args.fluct_m_chill = 0.0
    args.fluct_m_air = 0.0
    if args.penalty_a_fluctuation_type == 'Trapezoidal':
        args.action_bandwidth = trial.suggest_float('action_bandwidth', 0, 7)
    
    return args

def calc_performance_metrics(env, args):
    # Load the df of the evaluation and compute constraint metrics
    episode_idx = args.num_episodes+1
    data_dir = optuna_logbase_dir()
    run_dir = env.log_name
    analysis = EnergyPlusAnalysis(data_dir=data_dir, run_dir=run_dir, episode_idx=episode_idx)
    df = analysis.df

    # Compute the metrics
    column_T_ChL1 = "CHILLED WATER LOOP SUPPLY OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)"
    column_m_dot_chill = "CHILLED WATER LOOP CHILLER OUTLET NODE:System Node Mass Flow Rate [kg/s](TimeStep)"
    column_m_dot_air = "AIR LOOP CC OUTLET NODE:System Node Mass Flow Rate [kg/s](TimeStep)"
    column_T_AL1 = "COMPUTERROOM ZN CRAC SUPPLY OUTLET NODE:System Node Temperature [C](TimeStep)"
    column_T_AL2 = 'COMPUTERROOM ZN RETURN AIR NODE:System Node Temperature [C](TimeStep)'
    column_P_HVAC = "Total HVAC Power"

    L1_action_ChL1, L2_action_ChL1, Linf_action_ChL1 = action_analysis(df, column_T_ChL1)
    L1_T_SP_Al1, L2_T_SP_AL1, Linf_T_SP_AL1 = setpoint_analysis(df, column_T_AL1)
    L1_T_constr_AL2, L2_T_constr_AL2, Linf_T_constr_AL2 = temperature_constraint_analysis(df, column_T_AL2)
    P_HVAC_mean = df[column_P_HVAC].mean()

    # Write metrics to a json file
    metrics = {
        "P_HVAC": P_HVAC_mean,
        "L1_action_ChL1": L1_action_ChL1,
        "L2_action_ChL1": L2_action_ChL1,
        "Linf_action_ChL1": Linf_action_ChL1,
        "L1_T_SP_AL1": L1_T_SP_Al1,
        "L2_T_SP_AL1": L2_T_SP_AL1,
        "Linf_T_SP_AL1": Linf_T_SP_AL1,
        "L1_T_constr_AL2": L1_T_constr_AL2,
        "L2_T_constr_AL2": L2_T_constr_AL2,
        "Linf_T_constr_AL2": Linf_T_constr_AL2
    }
    
    metrics_file = os.path.join(env.log_dir, 'performance_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    return P_HVAC_mean, L2_action_ChL1, L2_T_SP_AL1, L2_T_constr_AL2

def objective(trial):
    args = setup_model_args(trial)

    # Create a wrapped environment
    env = make_env(args, is_hyper_param_optimization=True, trial_num=trial.number)

    # Train the agent
    trained_agent = train_agent(env, args)

    # Perform an evaluation run on the trained agent
    rew_eval, _ = evaluate_agent(trained_agent, env.log_dir)

    # Compute the performance metrics
    P_HVAC_mean, L2_action_ChL1, L2_T_SP_AL1, L2_T_constr_AL2 = calc_performance_metrics(env, args)

    # Close environment
    env.close()

    # Return the metrics
    return P_HVAC_mean, L2_action_ChL1, L2_T_SP_AL1, L2_T_constr_AL2

def main():
    # Define the study name
    study_name = "reward_tuning_2024_04_29"
    
    # Define the amount of trials
    n_trials = 100

    # Define the directory where the study will be saved
    optuna_log = optuna_logbase_dir()
    if not os.path.exists(optuna_log):
        os.makedirs(optuna_log)
    
    study_storage = f"sqlite:///{optuna_log}/{study_name}.db"

    partial_fixed_sampler = optuna.samplers.PartialFixedSampler({"penalty_a_fluctuation_type": "Trapezoidal"}, base_sampler=TPESampler())

    # Create the study
    study = optuna.create_study(
        study_name=study_name, 
        storage=study_storage, 
        load_if_exists=True,
        directions=['minimize', 'minimize', 'minimize', 'minimize'],
    )

    # Optimize the study
    study.optimize(
        objective, 
        n_trials=n_trials, 
        callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],
        sampler=partial_fixed_sampler
    )

if __name__ == '__main__':
    main()
    sys.exit(0)