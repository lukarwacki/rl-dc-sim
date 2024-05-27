import os
import time
import math

import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from tqdm import tqdm

from gym import spaces
from simulation.gym_energyplus.envs.energyplus_model import EnergyPlusModel


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')

class EnergyPlusModelRL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass(EnergyPlusModel):
    def __init__(self, 
                 model_file,    # Path to the EnergyPlus model file
                 log_dir,       # Directory where log files will be stored
                 verbose=False):# Flag indicating whether to print verbose output
        """
        Initializes an instance of the EnergyPlusModel1ZoneDCSmallHighITE class.

        Parameters:
        - model_file (str): Path to the EnergyPlus model file.
        - log_dir (str): Directory where log files will be stored.
        - verbose (bool, optional): Flag indicating whether to print verbose output.
        
        Initializes instance variables, sets default values, and handles compatibility for different EnergyPlus versions.
        """
        # Call the constructor of the parent class (EnergyPlusModel)
        super(EnergyPlusModelRL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass, self).__init__(model_file, log_dir, verbose)

        # Set a lower limit for the reward value
        self.reward_low_limit = -10000.
        self.setup_reward_normalization()
        self.setup_observation_normalization()

        # List to store power consumption information as text
        self.text_power_consumption = []

        # Determine the suffix for the facility power output variable based on EnergyPlus version
        if self.energyplus_version < (9, 4, 0):
            self.facility_power_output_var_suffix = "Electric Demand Power"
        else:
            self.facility_power_output_var_suffix = "Electricity Demand Rate"
        
        # List of electric powers per episode (not used)
        self.electric_powers = []

        
        # Choose the reward and penalty functions
        self.reward_P_type = 'P_HVAC'
        self.reward_T_type = None
        self.penalty_T_type = None
        self.penalty_a_fluctuation_type = None

        # Temperature constraint parameters
        self.T_constr = 24.
        self.T_bandwidth = 3.

        # Penalty & reward weights
        self.weight_penalty_T = 0.5
        self.weight_reward_T = 1.

        fluct_T_chill_sp = 0.0
        fluct_m_chill = 0.0
        fluct_T_air_sp = 0.0
        fluct_m_air = 0.0

        self.weight_action_fluctuation = np.array([fluct_T_chill_sp, fluct_m_chill, fluct_T_air_sp, fluct_m_air])
        
        # Softplus parameters
        self.beta = 1.
        self.softplus_shift = 3.

        # Gaussian parameters
        self.sigma = self.T_bandwidth / 3

        # Action bandwidth
        self.action_bandwidth = 1

        

    # -----------------------------------------------------------------------
    # Observation, state & reward
    # -----------------------------------------------------------------------
    def setup_spaces(self):
        # Chiller setpoint temperature limits
        T_chill_sp_lower_limit = 17.0
        T_chill_sp_upper_limit = 24.0

        m_chill_lower_limit = 0.0
        m_chill_upper_limit = 12.5

        m_air_lower_limit = 2.0
        m_air_upper_limit = 10.0

        T_air_sp_lower_limit = 17.0
        T_air_sp_upper_limit = 24.0

        # self.action_space_limits = np.array([[T_chill_sp_lower_limit, T_chill_sp_upper_limit],[m_chill_lower_limit, m_chill_upper_limit], [m_air_lower_limit, m_air_upper_limit]])

        # Set normalized action space
        self.action_space = spaces.Box(
            low=np.array([T_chill_sp_lower_limit, m_chill_lower_limit, T_air_sp_lower_limit, m_air_lower_limit]),
            high=np.array([T_chill_sp_upper_limit, m_chill_upper_limit, T_air_sp_upper_limit, m_air_upper_limit]),
            dtype=np.float64
        )

        # Observation space limits
        observation_lower_limit = -50.0
        observation_upper_limit = 2.0e5

        # Observation space shape
        observation_space_shape = (33,)
        self.observation_space = spaces.Box(
            low=observation_lower_limit,
            high=observation_upper_limit,
            shape=observation_space_shape,
            dtype=np.float64
        )
    
    def setup_observation_normalization(self):
        # Normalization ranges
        T_range = [-10,25]
        m_range = [0,12.5]
        P_hvac_range = [0,3e4]
        P_ite_range = [0,7e4]
        r_range = [0,1]
        Q_range = [0,7e4]
        v_range = [0,4]

        # State ranges dictionary
        ranges = dict(
            T=T_range,
            m=m_range,
            P_hvac=P_hvac_range,
            P_ite=P_ite_range,
            r=r_range,
            Q=Q_range,
            v=v_range,
        )

        # Setup states
        states = []

        states_CoL = ['m','T','T','T','T','P_hvac','r','P_hvac']
        states.extend(states_CoL)

        states_ChL = ['m','T','T','T','T','P_hvac','r','P_hvac']
        states.extend(states_ChL)

        states_AL = ['m','T','T','T','T','T','T','P_hvac','P_hvac']
        states.extend(states_AL)

        states_zn = ['Q','P_ite','P_ite','P_ite','Q','P_hvac', 'T','v']
        states.extend(states_zn)

        # Low and high values
        self.obs_low = [ranges[key][0] for key in states]
        self.obs_high = [ranges[key][1] for key in states]

    def setup_reward_normalization(self):
        self.rew_low = 0.
        self.rew_high = 3e4 

    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = np.zeros(shape=self.observation_space.shape)

    def compute_reward(self, raw_state=None):
        # Calculate power reward
        if self.reward_P_type == 'PUE':
            reward_power = self._reward_PUE(raw_state)
        elif self.reward_P_type == 'P_HVAC':
            reward_power = self._reward_P_HVAC(raw_state)
        else:
            reward_power = 0.
        
        # Calculate temperature reward
        if self.reward_T_type == 'Gaussian':
            reward_temp = self._reward_T_gauss(raw_state)
        else:
            reward_temp = 0.
        
        # Calculate temperature penalty
        if self.penalty_T_type == 'Softplus':
            penalty_temp = self._penalty_T_softplus(raw_state)
        elif self.penalty_T_type == 'ReLU':
            penalty_temp = self._penalty_T_relu(raw_state)
        elif self.penalty_T_type == 'Quadratic':
            penalty_temp = self._penalty_T_quad(raw_state)
        else:
            penalty_temp = 0.

        # Calculate action fluctuation penalty
        if self.penalty_a_fluctuation_type == 'Linear':
            penalty_a_diff = self._penalty_a_linear(raw_state)
        elif self.penalty_a_fluctuation_type == 'Trapezoidal':
            penalty_a_diff = self._penalty_a_trapezoidal(raw_state)
        elif self.penalty_a_fluctuation_type == 'Quadratic':
            penalty_a_diff = self._penalty_a_quadratic(raw_state)
        else:
            penalty_a_diff = np.zeros(shape=self.action_space.shape)

        # Calculate the reward
        reward = reward_power + self.weight_reward_T * reward_temp - self.weight_penalty_T * penalty_temp - np.dot(self.weight_action_fluctuation, penalty_a_diff)
        return reward
    
    # Calculate complete reward functions
    def _reward_PUE(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        P_ITE = self._calc_P_ITE()
        P_HVAC = self._calc_P_HVAC()
        PUE = (P_ITE+P_HVAC)/P_ITE
        rew = -PUE
        if math.isnan(rew):
            rew = 1
        return rew

    # Calculate rewards
    def _reward_P_HVAC(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        P_HVAC = self._calc_P_HVAC()
        return 1 - P_HVAC / self.rew_high 
    
    def _reward_T_gauss(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        T_diff = raw_state[19]-self.T_constr
        rew = np.exp(-0.5 * (T_diff / self.sigma)**2)
        return rew

    # Calculate temperature penalties
    def _penalty_T_softplus(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        T_diff = raw_state[19]-self.T_constr
        penalty = np.log(1 + np.exp(self.beta*(T_diff-self.softplus_shift)))/self.beta
        return penalty

    def _penalty_T_relu(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        T_diff = raw_state[19]-self.T_constr
        penalty = np.max([0, T_diff])
        return penalty
    
    def _penalty_T_quad(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        T_diff = raw_state[19]-self.T_constr
        if T_diff <= 0:
            penalty = 0
        else:
            penalty = T_diff**2
        return penalty
    
    # Calculate action penalties
    def _penalty_a_linear(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        a_diff = self.action - self.action_prev
        penalty = abs(a_diff)
        return penalty
    
    def _penalty_a_trapezoidal(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        a_diff = self.action - self.action_prev
        penalty = np.maximum(0, np.abs(a_diff) - self.action_bandwidth/2)
        return penalty
    
    def _penalty_a_quadratic(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        a_diff = self.action - self.action_prev
        penalty = a_diff ** 2
        return penalty


    # Calculate powers
    def _calc_P_HVAC(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        return raw_state[5] + raw_state[7] + raw_state[13] + raw_state[15] + raw_state[23] + raw_state[24]
    
    def _calc_P_ITE(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        return raw_state[26] + raw_state[27] + raw_state[28]
    
    def format_state(self, raw_state):
        # Perform mapping from raw state to the gym compatible state
        return np.array(raw_state)
    
    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    # These function aren't used, EnergyPlusAnalyzer is used instead
    def read_episode(self, ep): pass

    def plot_episode(self, ep): pass

    def dump_timesteps(self, log_dir='', csv_file='', **kwargs): pass

    def dump_episodes(self, log_dir='', csv_file='', **kwargs): pass

def test_EPlusModel_RL_DC():
    # Variables
    model_file = os.getenv('ENERGYPLUS_MODEL')
    log_dir = 'log_test'

    # Test functions
    ep_model = EnergyPlusModelRL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass(model_file=model_file, log_dir=log_dir)
    ep_model.set_raw_state(raw_state=None)
    ep_model.compute_reward

    return ep_model

if __name__ == "__main__":
    test_ep_model = test_EPlusModel_RL_DC()
    print(test_ep_model.action_space)
    print(test_ep_model.observation_space)
    print(test_ep_model.compute_reward())