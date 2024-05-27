import os
import time
import math
import numpy as np
from scipy.special import expit
import pandas as pd
import datetime as dt
from gym import spaces
from gym_energyplus.envs.energyplus_model import EnergyPlusModel
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')

class EnergyPlusModelRL_DC_T_chill(EnergyPlusModel):
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
        super(EnergyPlusModelRL_DC_T_chill, self).__init__(model_file, log_dir, verbose)

        # Set a lower limit for the reward value
        self.reward_low_limit = -10000.
        
        # List to store power consumption information as text
        self.text_power_consumption = []

        # Determine the suffix for the facility power output variable based on EnergyPlus version
        if self.energyplus_version < (9, 4, 0):
            self.facility_power_output_var_suffix = "Electric Demand Power"
        else:
            self.facility_power_output_var_suffix = "Electricity Demand Rate"
        
        # List of electric powers per episode (not used)
        self.electric_powers = []

    # -----------------------------------------------------------------------
    # Observation, state & reward
    # -----------------------------------------------------------------------
    def setup_spaces(self):
        # Chiller setpoint temperature limits
        T_chill_sp_lower_limit = 18.0
        T_chill_sp_upper_limit = 20.0
        self.real_action_space = [T_chill_sp_lower_limit, T_chill_sp_upper_limit]

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
        observation_space_shape = (55,)
        self.observation_space = spaces.Box(
            low=observation_lower_limit,
            high=observation_upper_limit,
            shape=observation_space_shape,
            dtype=np.float64
        )
    
    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = np.zeros(shape=self.observation_space.shape)

    def compute_reward(self, raw_state=None):
        rew = self._test_reward_PUE()
        return rew
    
    def _test_reward_PUE(self, raw_state=None):
        if raw_state is None:
            raw_state = self.raw_state
        P_ITE = raw_state[48] + raw_state[49] + raw_state[50]
        P_HVAC = raw_state[8] + raw_state[12] + raw_state[21] + raw_state[23] + raw_state[41] + raw_state[42]
        PUE = (P_ITE+P_HVAC)/P_ITE
        rew = -PUE
        if math.isnan(rew):
            rew = 1
        return rew
    
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
    ep_model = EnergyPlusModelRL_DC_T_chill(model_file=model_file, log_dir=log_dir)
    ep_model.set_raw_state(raw_state=None)
    ep_model.compute_reward

    return ep_model

if __name__ == "__main__":
    test_ep_model = test_EPlusModel_RL_DC()
    print(test_ep_model.action_space)
    print(test_ep_model.observation_space)
    print(test_ep_model.compute_reward())
