import os
import time
import math

import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from tqdm import tqdm

from gym import spaces
from gym_energyplus.envs.energyplus_model import EnergyPlusModel

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')

class EnergyPlusModelRL_DC_T_chill_SP_reduced_state(EnergyPlusModel):
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
        super(EnergyPlusModelRL_DC_T_chill_SP_reduced_state, self).__init__(model_file, log_dir, verbose)

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

        
        # Map reward types to functions
        self.reward_type = 'P_HVAC_softplus'
        self.alpha = 1e4
        self.beta = 3
        self.softplus_shift = 3/self.beta
        self.reward_functions = {
            'PUE': self._reward_PUE,
            'P_HVAC': self._reward_P_HVAC,
            'P_HVAC_relu': lambda: self._reward_P_HVAC_relu(22., self.alpha),
            'P_HVAC_softplus': lambda: self._reward_P_HVAC_softplus(22., self.alpha, self.beta, self.softplus_shift)
        }
        

    # -----------------------------------------------------------------------
    # Observation, state & reward
    # -----------------------------------------------------------------------
    def setup_spaces(self):
        # Chiller setpoint temperature limits
        T_chill_sp_lower_limit = 17.0
        T_chill_sp_upper_limit = 22.0
        self.action_space_limits = np.array([[T_chill_sp_lower_limit, T_chill_sp_upper_limit]])

        # Set normalized action space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
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
        self.rew_low = -3e4
        self.rew_high = -0.

    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = np.zeros(shape=self.observation_space.shape)

    def compute_reward(self, raw_state=None):
        # Get the reward function based on reward type
        reward_function = self.reward_functions.get(self.reward_type)
        
        if reward_function is None:
            raise ValueError(f"Incorrect reward type, chosen reward type: {self.reward_type}, but should be element of {self.reward_type_list}")
        
        # Call the reward function and return the reward
        return reward_function()
    
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
    
    def _reward_P_HVAC(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        return -self._calc_P_HVAC()

    def _reward_P_HVAC_relu(self, T_constr: float, alpha: float, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        
        P_HVAC = self._calc_P_HVAC()
        T_diff = raw_state[19]-T_constr
        penalty = np.max([0, T_diff])

        return - (P_HVAC + alpha*penalty)
    
    def _reward_P_HVAC_softplus(self, T_constr: float, alpha: float, beta: float, c: float, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        
        P_HVAC = self._calc_P_HVAC()
        T_diff = raw_state[19]-T_constr
        penalty = np.log(1 + np.exp(beta*T_diff))/beta

        return - (P_HVAC + alpha*penalty)

    def _calc_P_HVAC(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        return raw_state[6] + raw_state[7] + raw_state[13] + raw_state[15] + raw_state[23] + raw_state[24]
    
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
    ep_model = EnergyPlusModelRL_DC_T_chill_SP_reduced_state(model_file=model_file, log_dir=log_dir)
    ep_model.set_raw_state(raw_state=None)
    ep_model.compute_reward

    return ep_model

if __name__ == "__main__":
    test_ep_model = test_EPlusModel_RL_DC()
    print(test_ep_model.action_space)
    print(test_ep_model.observation_space)
    print(test_ep_model.compute_reward())
