import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from postprocessing.tools.energyplus_analyzer import EnergyPlusAnalysis
from simulation.helpers.energyplus_util import energyplus_logbase_dir

def temperature_constraint_analysis(df, column_to_analyze):
    # Temperature constraint analysis
    T_AL2_constr = 30.0
    T_AL2 = df[column_to_analyze]
    delta_T = T_AL2 - T_AL2_constr

    T_violation = np.maximum(delta_T, 0)
    T_violation_quadratic = T_violation**2
    
    L1_T_violation = calc_L1(T_violation)
    L2_T_violation = calc_L2(T_violation)
    Linf_T_violation = calc_Linf(T_violation)
    
    return L1_T_violation, L2_T_violation, Linf_T_violation

def action_analysis(df, column_to_analyze):
    # Extract the data
    actions = df[column_to_analyze]
    
    action_diff = actions.diff()

    L2_action = calc_L2(action_diff)
    L1_action = calc_L1(action_diff)
    Linf_action = calc_Linf(action_diff)

    return L1_action, L2_action, Linf_action

def setpoint_analysis(df, column_to_analyze):
    T_air = df[column_to_analyze]
    T_air_SP = 24.

    d_T_air = T_air - T_air_SP

    L1_T_air = calc_L1(d_T_air)
    L2_T_air = calc_L2(d_T_air)
    Linf_T_air = calc_Linf(d_T_air)

    return L1_T_air, L2_T_air, Linf_T_air

def calc_L2(vec):
    return np.sqrt(np.sum(vec**2))

def calc_L1(vec):
    return np.sum(np.abs(vec))

def calc_Linf(vec):
    return np.max(np.abs(vec))