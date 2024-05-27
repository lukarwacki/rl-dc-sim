import sys

from mpi4py import MPI
from simulation.helpers.energyplus_util import (
    energyplus_arg_parser,
    energyplus_logbase_dir,
    weather_files_dir,
    it_load_files_dir,
    copy_file,
)
from stable_baselines3 import PPO 
import os
import datetime
from stable_baselines3.common import logger
from simulation.helpers.gym_monitor import Monitor, LoggingCallback
from simulation.helpers.gym_make_environment import make_env
import gym
from gym.envs.registration import register
from simulation.gym_energyplus.envs import EnergyPlusEnv
import json
import numpy as np
import time
import shutil

def constant_agent(T_chill_vec, T_air_vec, sim_type='constant_T', m_chill_vec=None, m_air_vec=None):
    # Determine the temperature for which to do a baseline study
    action_list = []
    if sim_type == 'constant_T':
        for T_air in T_air_vec:
            for T_chill in T_chill_vec:
                if T_air > T_chill:
                    action_list.append([T_chill, T_air])
                    print(f"T_chill: {T_chill}, T_air: {T_air}")
    
    # Determine the mass flow rates for which to do a baseline study
    elif sim_type == 'constant_m_T':
        # Check if mass flow rates are provided
        assert m_chill_vec is not None and m_air_vec is not None, "Please provide mass flow rates for both chillers and air handling units"
        for m_air in m_air_vec:
            for m_chill in m_chill_vec:
                for T_air in T_air_vec:
                    for T_chill in T_chill_vec:
                        if T_air > T_chill:
                            action_list.append([T_chill, m_chill, T_air, m_air]) 
                            print(f"m_chill: {m_chill}, T_chill: {T_chill}, m_air: {m_air}, T_air: {T_air}") 
    else:
        raise ValueError(f"Invalid simulation type {sim_type}. Please choose between 'constant_T' and 'constant_m_T'")
    num_episodes = len(action_list)

    # Load arguments and make modifications
    args = energyplus_arg_parser().parse_args()
    args.framework = None
    args.num_episodes = num_episodes

    # Create a wrapped environment
    env = make_env(args)
    
    # Get the action and observation space
    action_space = env.action_space
    observation_space = env.observation_space
    print(f"Action space: {action_space}")
    print(f"State space: {observation_space}")

    # Append random action for last simulation run
    action_list.append(env.action_space.sample())
    action_list = np.array(action_list)

    # Initialize the logging of agents
    obs_vec = []
    rew_vec = []
    act_vec = []

    # Copy the weather and IT_file override into the log_dir
    log_dir = env.log_dir

    weather_dir = weather_files_dir()
    weather_filename = 'Weather_amsterdam_2022.epw'
    weather_location = os.path.join(weather_dir,weather_filename)
    copy_file(log_dir, weather_location)

    it_load_dir = it_load_files_dir()
    it_load_filename = 'IT_Load_4.csv'
    it_load_location = os.path.join(it_load_dir,it_load_filename)
    copy_file(log_dir, it_load_location)
    copied_it_load_filename = os.path.join(log_dir,it_load_filename)
    shutil.move(copied_it_load_filename, os.path.join(log_dir, 'IT_load.csv'))

    for action in action_list:
        # Check if action is inside action space
        if not action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be inside action space: {action_space}. You are probably using the wrong .idf file.")

        # Run the random agent
        obs = env.reset()
        done = False

        while not done:    
            # Perform selected action for one time step
            obs, rew, done, info = env.step(action)

            # Add variables to vectors
            act_vec.append(action)
            rew_vec.append(rew)
            obs_vec.append(obs)

    act_vec = np.ravel(act_vec)
    obs_vec = np.vstack(obs_vec[1::2])
    
    # Close the environment
    env.close()
    
    return act_vec, obs_vec, rew_vec


if __name__ == '__main__':
    # sim_type = 'constant_m_T'
    sim_type = 'constant_T'
    
    # m_chill_vec = np.arange(3,6,1)
    # T_chill_vec = np.arange(17,25,1)

    # m_air_vec = np.arange(3,6,1)
    # T_air_vec = np.arange(17,25,1)
    
    m_chill_vec = np.array([5])
    T_chill_vec = np.array([22])

    m_air_vec = np.array([5])
    T_air_vec = np.array([24])

    # Run the simulation
    constant_agent(
        T_chill_vec=T_chill_vec, 
        T_air_vec=T_air_vec, 
        sim_type=sim_type, 
        m_chill_vec=m_chill_vec, 
        m_air_vec=m_air_vec
    )
    