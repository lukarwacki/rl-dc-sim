import json
import math
import numpy as np
import pandas as pd


class RewardCalculator():
    """
    Class to calculate the reward for the EnergyPlus environment

    This class is used by the energyplus_model_XXXXX.py files to calculate the reward for the environment.
    
    This class is initialized with a JSON file containing the reward parameters, as defined by the argumentparser. 
    These parameters are used to calculate the reward for the environment.
    
    The calculation of the reward is done by the 'calculate_reward' method, which takes the raw state, previous action, 
    current action, and the maximum HVAC power as an input. This function can be modified to include a different reward if desired.

    All other methods are helper functions for the 'calculate_reward' method, and should not be called directly.

    NOTE: If the reward is modified, also modify the following functions accordingly:
        - 'write_reward_settings_json' function in simulation/helpers/gym_make_environment.py
        - 'WriteSettingsCallback' class in simulation/helpers/gym_monitor.py
    """

    def __init__(self, reward_file):
        self.reward_file = reward_file

        try:
            with open(self.reward_file, 'r') as file:
                self.reward_data = json.load(file)
        
        except FileNotFoundError:
            print(f"File not found: {self.reward_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {self.reward_file}")
        
        # Set the reward and penalty functions
        self.reward_P_type = self.reward_data['reward_P_type']
        self.reward_T_type = self.reward_data['reward_T_type']
        self.penalty_T_type = self.reward_data['penalty_T_type']
        self.penalty_a_fluctuation_type = self.reward_data['penalty_a_fluctuation_type']

        # Set the reward weights
        self.lambda_T_r = self.reward_data['lambda_T_r']
        self.lambda_T_p = self.reward_data['lambda_T_p']

        fluct_T_sp = self.reward_data['fluct_T_sp']
        fluct_m_chill = self.reward_data['fluct_m_chill']
        fluct_m_air = self.reward_data['fluct_m_air']
        self.lambda_a_p = np.array([fluct_T_sp, fluct_m_chill, fluct_m_air])

        # Set the temperature constraint parameters
        self.T_SP = self.reward_data['T_SP']
        self.T_SP_bandwidth = self.reward_data['T_SP_bandwidth']
        self.T_constr = self.reward_data['T_constr']

        # Set the softplus parameters
        self.beta = self.reward_data['beta']
        self.softplus_shift = self.reward_data['softplus_shift']

        # Set the Gaussian parameters
        self.sigma = self.T_SP_bandwidth / 3

        # Set the action bandwidth
        self.action_bandwidth = self.reward_data['action_bandwidth']
    

    def calculate_reward(self, raw_state, action_prev, action, P_HVAC_max):
         # Calculate power reward
        if self.reward_P_type == 'PUE':
            reward_power = self._reward_PUE(raw_state)
        elif self.reward_P_type == 'P_HVAC':
            reward_power = self._reward_P_HVAC(P_HVAC_max, raw_state)
        elif self.reward_P_type == None:
            reward_power = 0.
        else:
            raise ValueError(f"Unknown reward type: {self.reward_P_type}")
        
        # Calculate temperature reward
        if self.reward_T_type == 'Gaussian':
            reward_temp = self._reward_T_gauss(raw_state)
        elif self.reward_T_type == None:
            reward_temp = 0.
        else:
            raise ValueError(f"Unknown reward type: {self.reward_T_type}")
        
        # Calculate temperature penalty
        if self.penalty_T_type == 'Softplus':
            penalty_temp = self._penalty_T_softplus(T_constr=self.T_constr, beta=self.beta, softplus_shift=self.softplus_shift, raw_state=raw_state)
        elif self.penalty_T_type == 'ReLU':
            penalty_temp = self._penalty_T_relu(T_constr=self.T_constr, raw_state=raw_state)
        elif self.penalty_T_type == 'ReLU2':
            penalty_temp = self._penalty_T_quad(T_constr=self.T_constr, raw_state=raw_state)
        elif self.penalty_T_type == None:
            penalty_temp = 0.
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_T_type}")

        # Calculate action fluctuation penalty
        if self.penalty_a_fluctuation_type == 'Linear':
            penalty_a_diff = self._penalty_a_linear(action, action_prev, raw_state)
        elif self.penalty_a_fluctuation_type == 'Trapezoidal':
            penalty_a_diff = self._penalty_a_trapezoidal(action, action_prev, raw_state)
        elif self.penalty_a_fluctuation_type == 'Quadratic':
            penalty_a_diff = self._penalty_a_quadratic(action, action_prev, raw_state)
        elif self.penalty_a_fluctuation_type == None:
            penalty_a_diff = np.zeros(shape=action.shape)
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_a_fluctuation_type}")

        # Calculate the reward
        reward = reward_power + self.lambda_T_r * reward_temp - self.lambda_T_p * penalty_temp - np.dot(self.lambda_a_p, penalty_a_diff)
        return reward
    
    # ------------------ Power rewards ------------------
    def _reward_PUE(self, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        P_ITE = self._calc_P_ITE(raw_state)
        P_HVAC = self._calc_P_HVAC(raw_state)
        PUE = (P_ITE+P_HVAC)/P_ITE
        rew = -PUE
        if math.isnan(rew):
            rew = 1
        return rew

    def _reward_P_HVAC(self, P_HVAC_max, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        P_HVAC = self._calc_P_HVAC(raw_state)
        return 1 - P_HVAC / P_HVAC_max 
    
    # --------------- Setpoint temperature --------------
    def _reward_T_gauss(self, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        T_diff = raw_state[19]-self.T_SP
        rew = np.exp(-0.5 * (T_diff / self.sigma)**2)
        return rew

    # --------------- Temperature penalties --------------
    def _penalty_T_softplus(self, T_constr, beta, softplus_shift, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        T_diff = raw_state[22] - T_constr
        penalty = np.log(1 + np.exp(beta*(T_diff-softplus_shift)))/beta
        return penalty

    def _penalty_T_relu(self, T_constr, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        T_diff = raw_state[22] - T_constr
        penalty = np.max([0, T_diff])
        return penalty
    
    def _penalty_T_quad(self, T_constr, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        T_diff = raw_state[22] - T_constr
        if T_diff <= 0:
            penalty = 0
        else:
            penalty = T_diff**2
        return penalty
    
    # --------------- Action penalties --------------
    def _penalty_a_linear(self, action, action_prev, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        a_diff = action - action_prev
        penalty = abs(a_diff)
        return penalty
    
    def _penalty_a_trapezoidal(self, action, action_prev, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        a_diff = action - action_prev
        penalty = np.maximum(0, np.abs(a_diff) - self.action_bandwidth/2)
        return penalty
    
    def _penalty_a_quadratic(self, action, action_prev, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        a_diff = action - action_prev
        penalty = a_diff ** 2
        return penalty


    # Calculate powers
    def _calc_P_HVAC(self, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        return raw_state[5] + raw_state[7] + raw_state[13] + raw_state[15] + raw_state[23] + raw_state[24]
    
    def _calc_P_ITE(self, raw_state=None):
        if raw_state is None:
            raise ValueError("No state provided")
        return raw_state[26] + raw_state[27] + raw_state[28]
    