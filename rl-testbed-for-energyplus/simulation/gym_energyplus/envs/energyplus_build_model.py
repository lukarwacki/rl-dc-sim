# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from re import match
import os

# from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer import \
#     EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer
# from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp import \
#     EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp
# from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan import \
#     EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan
# from gym_energyplus.envs.energyplus_model_DC_1Zone_Small_HighITE import \
#     EnergyPlusModel1ZoneDCSmallHighITE
# from gym_energyplus.envs.energyplus_model_RL_DC_T_chill_SP import \
#     EnergyPlusModelRL_DC_T_chill
# from gym_energyplus.envs.energyplus_model_RL_DC_T_chill_SP_reduced_state import \
#     EnergyPlusModelRL_DC_T_chill_SP_reduced_state
from simulation.gym_energyplus.envs.energyplus_model_RL_DC_ChillerTemp_ChillerMass_AirMass import \
    EnergyPlusModelRL_DC_ChillerTemp_ChillerMass_AirMass
from simulation.gym_energyplus.envs.energyplus_model_RL_DC_ChillerTemp_AirTemp import \
    EnergyPlusModelRL_DC_ChillerTemp_AirTemp
from simulation.gym_energyplus.envs.energyplus_model_RL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass import \
    EnergyPlusModelRL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass

def build_ep_model(model_file, log_dir, verbose=False, reward_file=None):
    model_basename = os.path.splitext(os.path.basename(model_file))[0]
    print(f"Model basename is: {model_basename}")
    # if match('2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.*', model_basename):
    #     model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose
    #     )
    # elif match('2ZoneDataCenterHVAC_wEconomizer_Temp.*', model_basename):
    #     model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose
    #     )
    # elif match('2ZoneDataCenterHVAC_wEconomizer.*', model_basename):
    #     model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose
    #     )
    # elif match('DC_1Zone_Small_HighITE_v1.*', model_basename):
    #     model = EnergyPlusModel1ZoneDCSmallHighITE(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose
    #     )
    # elif match('RL_DC_T_chill_SP_reduced_state.*', model_basename):
    #     model = EnergyPlusModelRL_DC_T_chill_SP_reduced_state(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose            
    #     )
    # elif match('RL_DC_T_chill_SP_v1.*', model_basename):
    #     model = EnergyPlusModelRL_DC_T_chill(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose
    #     )
    if match('RL_DC_ChillerTemp_ChillerMass_AirMass.*', model_basename):
        model = EnergyPlusModelRL_DC_ChillerTemp_ChillerMass_AirMass(
            model_file=model_file,
            log_dir=log_dir,
            verbose=verbose,
            reward_file=reward_file,
        )
    elif match('RL_DC_ChillerTemp_AirTemp.*', model_basename):
        model = EnergyPlusModelRL_DC_ChillerTemp_AirTemp(
            model_file=model_file,
            log_dir=log_dir,
            verbose=verbose
        )
    elif match('RL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass.*', model_basename):
        model = EnergyPlusModelRL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass(
            model_file=model_file,
            log_dir=log_dir,
            verbose=verbose
        )
    else:
        raise ValueError('Unsupported EnergyPlus model')
    return model

# if __name__ == "__main__":
#     # Variables
#     model_file = os.getenv('ENERGYPLUS_MODEL')
#     log_dir = 'log_test'
#     print("Model file = {}".format(model_file))
#     print("Logging directory = {}".format(log_dir))

#     model, model_basename = build_ep_model(model_file=model_file, log_dir=log_dir)
#     print("Model basename = {}".format(model_basename))

