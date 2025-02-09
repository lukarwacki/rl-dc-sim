# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os
import time
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
from matplotlib.widgets import Slider, Button, RadioButtons

class EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(EnergyPlusModel):

    def __init__(self,
                 model_file,
                 log_dir,
                 verbose=False):
        super(EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan, self).__init__(model_file, log_dir, verbose)
        self.reward_low_limit = -10000.
        self.axepisode = None
        self.num_axes = 5
        self.text_power_consumption = []
        if self.energyplus_version < (9, 4, 0):
            self.facility_power_output_var_suffix = "Electric Demand Power"
        else:
            self.facility_power_output_var_suffix = "Electricity Demand Rate"

        self.electric_powers = [
            #'WESTDATACENTER_EQUIP:ITE CPU Electric Power [W](Hourly)', # low
            #'WESTDATACENTER_EQUIP:ITE Fan Electric Power [W](Hourly)', # low
            #'WESTDATACENTER_EQUIP:ITE UPS Electric Power [W](Hourly)', # low

            #'WESTDATACENTER_EQUIP:ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WESTDATACENTER_EQUIP:ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WEST ZONE:Zone ITE CPU Electric Power [W](Hourly)', # low
            #'WEST ZONE:Zone ITE Fan Electric Power [W](Hourly)', # low
            #'WEST ZONE:Zone ITE UPS Electric Power [W](Hourly)', # low
            #'WEST ZONE:Zone ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WEST ZONE:Zone ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WEST DATA CENTER IEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (works only on very cold day)
            #'WEST DATA CENTER DEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (never works)
            #'EMS:Power Utilization Effectiveness [](TimeStep)',

            #f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)', # very high
            #f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix} [W](Hourly)', # very high
            #f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix} [W](Hourly)', # low

            #'EASTDATACENTER_EQUIP:ITE CPU Electric Power [W](Hourly)', # low
            #'EASTDATACENTER_EQUIP:ITE Fan Electric Power [W](Hourly)', # low
            #'EASTDATACENTER_EQUIP:ITE UPS Electric Power [W](Hourly)', # low

            #'EASTDATACENTER_EQUIP:ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EASTDATACENTER_EQUIP:ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EAST ZONE:Zone ITE CPU Electric Power [W](Hourly)', # low
            #'EAST ZONE:Zone ITE Fan Electric Power [W](Hourly)', # low
            #'EAST ZONE:Zone ITE UPS Electric Power [W](Hourly)', # low
            #'EAST ZONE:Zone ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EAST ZONE:Zone ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EAST DATA CENTER IEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (works only on very cold day)
            #'EAST DATA CENTER DEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (never works)
            
            #'EMS:Power Utilization Effectiveness [](TimeStep)',
        ]
        
    def setup_spaces(self):
        # Bound action temperature
        lo = 10.0
        hi = 40.0
        flow_hi = 7.0
        flow_lo = flow_hi * 0.25
        self.action_space = spaces.Box(low =   np.array([ lo, lo, flow_lo, flow_lo]),
                                       high =  np.array([ hi, hi, flow_hi, flow_hi]),
                                       dtype = np.float32)
        self.observation_space = spaces.Box(low   = np.array([-20.0, -20.0, -20.0,          0.0,          0.0,          0.0]),
                                            high  = np.array([ 50.0, 200.0, 200.0, 1000000000.0, 1000000000.0, 1000000000.0]),
                                            dtype = np.float32)
        
    def set_raw_state(self, raw_state):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def compute_reward(self):
        rew, _ = self._compute_reward()
        return rew

    def _compute_reward(self, raw_state = None):
        return self.compute_reward_center23_5_gaussian1_0_trapezoid0_1_pue0_0(raw_state)
        #return self.compute_reward_center23_5_gaussian1_0_trapezoid1_0_pue0_0(raw_state)
        #return self.compute_reward_gaussian1_0_trapezoid1_0_pue0_0(raw_state)
        #return self.compute_reward_gaussian1_0_trapezoid0_1_pue0_0_pow0(raw_state)
        #return self.compute_reward_gaussian1_0_trapezoid1_0_pue0_0(raw_state)
        
    def compute_reward_center23_5_gaussian1_0_trapezoid0_1_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 23.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 0.1,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)
        
    def compute_reward_center23_5_gaussian1_0_trapezoid1_0_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 23.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 1.0,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)
        
    def compute_reward_gaussian1_0_trapezoid1_0_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 1.0,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)

    def compute_reward_gaussian1_0_trapezoid0_1_pue0_0_pow0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 0.1,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 0.0,
            raw_state = raw_state)

    def compute_reward_gaussian1_0_trapezoid0_1_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 0.1,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)

    def compute_reward_gaussian_pue0_0(self, raw_state = None): # gaussian, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 1.0,
            temperature_trapezoid_weight = 0.,
            fluctuation_weight = 0.1,
            PUE_weight = 1.0,
            Whole_Building_Power_weight = 0.,
            raw_state = raw_state)

    def compute_reward_gaussian_whole_power(self, raw_state = None): # gaussian, whole power
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 1.0,
            temperature_trapezoid_weight = 0.,
            fluctuation_weight = 0.1,
            PUE_weight = 0.0, # PUE not used
            Whole_Building_Power_weight = 1 / 100000.,
            raw_state = raw_state)

    def compute_reward_common(self, 
                              temperature_center = 22.5,
                              temperature_tolerance = 0.5,
                              temperature_gaussian_weight = 0.,
                              temperature_gaussian_sharpness = 1.,
                              temperature_trapezoid_weight = 0.,
                              fluctuation_weight = 0.,
                              PUE_weight = 0.,
                              Whole_Building_Power_weight = 0.,
                              raw_state = None):
        if raw_state is not None:
            st = raw_state
        else:
            st = self.raw_state

        Tenv = st[0]
        Tz1 = st[1]
        Tz2 = st[2]
        PUE = st[3]
        Whole_Building_Power = st[4]
        IT_Equip_Power = st[5]
        Whole_HVAC_Power = st[6]

        rew_PUE = -(PUE - 1.0) * PUE_weight
        # Temp. gaussian
        rew_temp_gaussian1 = np.exp(-(Tz1 - temperature_center) * (Tz1 - temperature_center) * temperature_gaussian_sharpness) * temperature_gaussian_weight
        rew_temp_gaussian2 = np.exp(-(Tz2 - temperature_center) * (Tz2 - temperature_center) * temperature_gaussian_sharpness) * temperature_gaussian_weight
        rew_temp_gaussian = rew_temp_gaussian1 + rew_temp_gaussian2
        # Temp. Trapezoid
        phi_low = temperature_center - temperature_tolerance
        phi_high = temperature_center + temperature_tolerance
        if Tz1 < phi_low:
            rew_temp_trapezoid1 = - temperature_trapezoid_weight * (phi_low - Tz1)
        elif Tz1 > phi_high:
            rew_temp_trapezoid1 = - temperature_trapezoid_weight * (Tz1 - phi_high)
        else:
            rew_temp_trapezoid1 = 0.
        if Tz2 < phi_low:
            rew_temp_trapezoid2 = - temperature_trapezoid_weight * (phi_low - Tz2)
        elif Tz2 > phi_high:
            rew_temp_trapezoid2 = - temperature_trapezoid_weight * (Tz2 - phi_high)
        else:
            rew_temp_trapezoid2 = 0.
        rew_temp_trapezoid = rew_temp_trapezoid1 + rew_temp_trapezoid2
        
        rew_fluct = 0.
        if raw_state is None:
            for cur, prev in zip(self.action, self.action_prev):
                rew_fluct -= abs(cur - prev) * fluctuation_weight
        rew_Whole_Building_Power = - Whole_Building_Power * Whole_Building_Power_weight
        rew = rew_temp_gaussian + rew_temp_trapezoid + rew_fluct + rew_PUE + rew_Whole_Building_Power
        if os.path.exists("/tmp/verbose"):
            print('compute_reward: rew={:7.3f} (temp_gaussian1={:7.3f}, temp_gaussian2={:7.3f}, temp_trapezoid1={:7.3f}, temp_trapezoid2={:7.3f}, fluct={:7.3f}, PUE={:7.3f}, Power={:7.3f})'.format(rew, rew_temp_gaussian1, rew_temp_gaussian2, rew_temp_trapezoid1, rew_temp_trapezoid2, rew_fluct, rew_PUE, rew_Whole_Building_Power))
        if os.path.exists("/tmp/verbose2"):
            print('compute_reward: Tenv={:7.3f}, Tz1={:7.3f}, Tz2={:7.3f}, PUE={:7.3f}, Whole_Powerd2={:8.1f}, ITE_Power={:8.1f}, HVAC_Power={:8.1f}, Act1={:7.3f}, Act2={:7.3f}, Act3={:7.3f}, Act4={:7.3f}'.format(Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power, self.action[0], self.action[1], self.action[2], self.action[3]))

        return rew, (rew_temp_gaussian1, rew_temp_trapezoid1, rew_temp_gaussian2, rew_temp_trapezoid2, rew_Whole_Building_Power)

    
    def format_state(self, raw_state):
        '''
        Performs mapping from raw_state (retrieved from EnergyPlus process as is) to gym compatible state
        
        state[0] = raw_state[0]: Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)
        state[1] = raw_state[1]: WEST ZONE:Zone Air Temperature [C](TimeStep)
        state[2] = raw_state[2]: EAST ZONE:Zone Air Temperature [C](TimeStep)
                    raw_state[3]: EMS:Power Utilization Effectiveness [](TimeStep)
        state[3] = raw_state[4]: Whole Building:Facility Total Electricity Demand Rate [W](Hourly)
        state[4] = raw_state[5]: Whole Building:Facility Total Building Electricity Demand Rate [W](Hourly)
        state[5] = raw_state[6]: Whole Building:Facility Total HVAC Electricity Demand Rate [W](Hourly)
        '''
        return np.array([raw_state[0], raw_state[1], raw_state[2], raw_state[4], raw_state[5], raw_state[6]])

    def read_episode(self, ep):
        """
        Reads and processes an episode's data, extracting relevant information such as
        file paths, weather keys, temperatures for different zones, power consumption,
        and computing rewards based on the environmental conditions.

        Parameters:
        - ep: Either a string representing the file path of the episode or an index
            pointing to the corresponding episode directory.

        Notes:
        - If 'ep' is an index, it checks for the existence of 'eplusout.csv' or
        'eplusout.csv.gz' in the corresponding episode directory. If neither file is
        found, an error message is printed, and the program quits.
        - The function prints the file path of the episode being read.
        - The CSV file is read into a Pandas DataFrame, with missing values filled using
        forward-fill and backward-fill methods.
        - The 'Date/Time' column in the DataFrame is converted to a 24-hour format.
        - The weather key is extracted from the episode's weather file path, and
        temperatures for outdoor air, the West Zone, and the East Zone are extracted
        from the DataFrame.
        - PUE (Power Utilization Effectiveness) values are extracted from the DataFrame.
        - Various temperatures and setpoint temperatures for different zones are assigned.
        - Electric power consumption data is extracted from the DataFrame.
        - Rewards are computed for each timestep based on environmental conditions, and
        these rewards are stored in different lists.
        - Cooling and heating setpoints are initialized for ZoneControl:Thermostat.
        - x_pos and x_labels are generated for plotting.
        """
        # Check if 'ep' is a string (file path) or an index
        if type(ep) is str:
            file_path = ep
        else:
            # If 'ep' is an index, get the corresponding episode directory
            ep_dir = self.episode_dirs[ep]

            # Check if eplusout.csv or .csv.gz exist
            for file in ['eplusout.csv', 'eplusout.csv.gz']:
                file_path = ep_dir + '/' + file
                if os.path.exists(file_path):
                    break;
            else:
                # If neither file is found, print an error message and quit
                print('No CSV or CSV.gz found under {}'.format(ep_dir))
                quit()

        # Print the file path of the episode being read
        print('read_episode: file={}'.format(file_path))

        # Read the CSV file into a Pandas DataFrame, filling missing values
        df = pd.read_csv(file_path).fillna(method='ffill').fillna(method='bfill')
        self.df = df

        # Convert 'Date/Time' column to a 24-hour format
        date = self.df['Date/Time']
        date_time = self._convert_datetime24(date)

        # Extract the weather key and temperature data for different zones
        epw_files = glob(os.path.dirname(file_path) + '/USA_??_*.epw')
        if len(epw_files) == 1:
            self.weather_key = os.path.basename(epw_files[0])[4:6]
        else:
            self.weather_key = '  '
        # Extract air temperature for different zones from the DataFrame
        self.outdoor_temp = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
        self.westzone_temp = df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
        self.eastzone_temp = df['EAST ZONE:Zone Air Temperature [C](TimeStep)']

        # Extract the PUE value from the DataFrame
        self.pue = df['EMS:Power Utilization Effectiveness [](TimeStep)']

        #self.westzone_ite_cpu_electric_power = df['WEST ZONE:Zone ITE CPU Electric Power [W](Hourly)']
        #self.westzone_ite_fan_electric_power = df['WEST ZONE:Zone ITE Fan Electric Power [W](Hourly)']
        #self.westzone_ite_ups_electric_power = df['WEST ZONE:Zone ITE UPS Electric Power [W](Hourly)']

        #WEST ZONE INLET NODE:System Node Temperature [C](TimeStep)
        #WEST ZONE INLET NODE:System Node Mass Flow Rate [kg/s](TimeStep)

        # Assign various temperatures and setpoint temperatures for West Zone
        self.westzone_return_air_temp = df['WEST ZONE RETURN AIR NODE:System Node Temperature [C](TimeStep)']
        self.westzone_mixed_air_temp = df['WEST ZONE MIXED AIR NODE:System Node Temperature [C](TimeStep)']
        self.westzone_supply_fan_outlet_temp = df['WEST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_dec_outlet_temp = df['WEST ZONE DEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_dec_outlet_setpoint_temp = df['WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_iec_outlet_temp = df['WEST ZONE IEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_iec_outlet_setpoint_temp = df['WEST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_ccoil_air_outlet_temp = df['WEST ZONE CCOIL AIR OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_ccoil_air_outlet_setpoint_temp = df['WEST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_air_loop_outlet_temp = df['WEST AIR LOOP OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_air_loop_outlet_setpoint_temp = df['WEST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']

        # Assign temperatures and setpoint temperatures for East Zone
        #XX self.eastzone_return_air_temp = df['EAST ZONE RETURN AIR NODE:System Node Temperature [C](TimeStep)']
        #XXself.eastzone_mixed_air_temp = df['EAST ZONE MIXED AIR NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_supply_fan_outlet_temp = df['EAST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_dec_outlet_temp = df['EAST ZONE DEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_dec_outlet_setpoint_temp = df['EAST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_iec_outlet_temp = df['EAST ZONE IEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_iec_outlet_setpoint_temp = df['EAST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_ccoil_air_outlet_temp = df['EAST ZONE CCOIL AIR OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_ccoil_air_outlet_setpoint_temp = df['EAST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_air_loop_outlet_temp = df['EAST AIR LOOP OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_air_loop_outlet_setpoint_temp = df['EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']

        # Extract electric power data
        self.total_building_electric_demand_power = df[f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix} [W](Hourly)']
        self.total_hvac_electric_demand_power = df[f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix} [W](Hourly)']
        self.total_electric_demand_power = df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)']

        # Initialize lists where rewards are stored 
        self.rewards = []
        self.rewards_gaussian1 = []
        self.rewards_trapezoid1 = []
        self.rewards_gaussian2 = []
        self.rewards_trapezoid2 = []
        self.rewards_power = []
        
        # Iterate over each timestep and compute rewards
        for Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power in zip(
                self.outdoor_temp,
                self.westzone_temp,
                self.eastzone_temp,
                self.pue,
                self.total_electric_demand_power,
                self.total_building_electric_demand_power,
                self.total_hvac_electric_demand_power):
            # Compute reward and extract individual elements
            rew, elem = self._compute_reward([Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power])
            
            # Append values to respective lists
            self.rewards.append(rew)
            self.rewards_gaussian1.append(elem[0])
            self.rewards_trapezoid1.append(elem[1])
            self.rewards_gaussian2.append(elem[2])
            self.rewards_trapezoid2.append(elem[3])
            self.rewards_power.append(elem[4])
        
        # Set cooling and heating setpoint for ZoneControl:Thermostat
        self.cooling_setpoint = []
        self.heating_setpoint = []
        for dt in date_time:
            self.cooling_setpoint.append(24.0)
            self.heating_setpoint.append(23.0)
        
        # Generate x_pos and x_labels for plotting
        (self.x_pos, self.x_labels) = self.generate_x_pos_x_labels(date)

    def plot_episode(self, ep):
        print('episode {}'.format(ep))

        # Read episode data and process it
        self.read_episode(ep)

        # Show statistics for different variables
        self.show_statistics('Reward', self.rewards)
        self.show_statistics('westzone_temp', self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'])
        self.show_statistics('eastzone_temp', self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'])
        self.show_statistics('Power consumption', self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'])
        self.show_statistics('pue', self.pue)

        # Show distribution of the West Zone tempearture 
        self.show_distrib('westzone_temp distribution', self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'])

        # Check if axes have been initialized, if not, create and configure them
        if self.axepisode is None: 
            self.axepisode = []
            for i in range(self.num_axes):
                # Create subplot axes for each statistic or distribution
                if i == 0:
                    ax = self.fig.add_axes([0.05, 1.00 - 0.70 / self.num_axes * (i + 1), 0.90, 0.7 / self.num_axes * 0.85])
                else:
                    ax = self.fig.add_axes([0.05, 1.00 - 0.70 / self.num_axes * (i + 1), 0.90, 0.7 / self.num_axes * 0.85], sharex=self.axepisode[0])
                ax.set_xmargin(0)
                self.axepisode.append(ax)
                ax.set_xticks(self.x_pos)
                ax.set_xticklabels(self.x_labels)
                ax.tick_params(labelbottom='off')
                ax.grid(True)

        # Initialize index variable and show_west flag
        idx = 0
        show_west = True

        if True:
            # Plot zone and outdoor temperature
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.westzone_temp, 'C0', label='Westzone temperature')
            ax.plot(self.eastzone_temp, 'C1', label='Eastzone temperature')
            ax.plot(self.outdoor_temp, 'C2', label='Outdoor temperature')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(-1.0, 40.0)

        if False:
            # Plot return air and sestpoint temperature
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            if show_west:
                ax.plot(self.westzone_return_air_temp, 'C0', label='WEST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(self.westzone_dec_outlet_setpoint_temp, 'C1', label='Westzone DEC outlet setpoint temperature')
            else:
                #ax.plot(self.eastzone_return_air_temp, 'C0', label='EAST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(self.eastzone_dec_outlet_setpoint_temp, 'C1', label='Eastzone DEC outlet setpoint temperature')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if True:
            # Plot return air and sestpoint temperature
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.df['WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)'], 'C0', label='WEST ZONE Air Mass Flow Rate')
            ax.plot(self.df['EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)'], 'C1', label='EAST ZONE Air Mass Flow Rate')
            ax.legend()
            ax.set_ylabel('Air volume rate (kg/s)')
            ax.set_ylim(0.0, 10.0)

        if True:
            # Plot west zone, return air, mixed air, supply fan
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            if show_west:
                ax.plot(self.westzone_return_air_temp, 'C0', label='WEST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(self.westzone_mixed_air_temp, 'C1', label='WEST ZONE MIXED AIR NODE:System Node Temperature')
                ax.plot(self.westzone_supply_fan_outlet_temp, 'C2', label='WEST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature')
                ax.plot(self.westzone_dec_outlet_temp, 'C3', label='Westzone DEC outlet temperature')
            else:
                #ax.plot(self.eastzone_return_air_temp, 'C0', label='EAST ZONE RETURN AIR NODE:System Node Temperature')
                #ax.plot(self.eastzone_mixed_air_temp, 'C1', label='EAST ZONE MIXED AIR NODE:System Node Temperature')
                ax.plot(self.eastzone_supply_fan_outlet_temp, 'C2', label='EAST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature')
                ax.plot(self.eastzone_dec_outlet_temp, 'C3', label='Eastzone DEC outlet temperature')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if True:
            # Plot west zone ccoil, air loop
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            if show_west:
                ax.plot(self.westzone_iec_outlet_temp, 'C0', label='Westzone IEC outlet temperature')
                ax.plot(self.westzone_ccoil_air_outlet_temp, 'C1', label='Westzone ccoil air outlet temperature')
                ax.plot(self.westzone_air_loop_outlet_temp, 'C2', label='Westzone air loop outlet temperature')
                ax.plot(self.westzone_dec_outlet_setpoint_temp, label='Westzone DEC outlet setpoint temperature', linewidth=0.5, color='gray')
            else:
                ax.plot(self.eastzone_iec_outlet_temp, 'C0', label='Eastzone IEC outlet temperature')
                ax.plot(self.eastzone_ccoil_air_outlet_temp, 'C1', label='Eastzone ccoil air outlet temperature')
                ax.plot(self.eastzone_air_loop_outlet_temp, 'C2', label='Eastzone air loop outlet temperature')
                ax.plot(self.eastzone_dec_outlet_setpoint_temp, label='Eastzone DEC outlet setpoint temperature', linewidth=0.5, color='gray')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if False:
            # Plot calculated reward
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.rewards, 'C0', label='Reward')
            ax.plot(self.rewards_gaussian1, 'C1', label='Gaussian1')
            ax.plot(self.rewards_trapezoid1, 'C2', label='Trapezoid1')
            ax.plot(self.rewards_gaussian2, 'C3', label='Gaussian2')
            ax.plot(self.rewards_trapezoid2, 'C4', label='Trapezoid2')
            ax.plot(self.rewards_power, 'C5', label='Power')
            ax.legend()
            ax.set_ylabel('Reward')
            ax.set_ylim(-2.0, 2.0)


        if False:
            # Plot PUE
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.pue, 'C0', label='PUE')
            ax.legend()
            ax.set_ylabel('PUE')
            ax.set_ylim(top=2.0, bottom=1.0)
            
        if False:
            # Plot other electric power consumptions
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            #ax.plot(self.total_electric_demand_power, 'C0', label=f'Whole Building:Facility Total {self.facility_power_output_var_suffix}')
            #ax.plot(self.total_building_electric_demand_power, 'C1', label=f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix}')
            #ax.plot(self.total_hvac_electric_demand_power, 'C2', label=f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix}')
            for i, pow in enumerate(self.electric_powers):
                ax.plot(self.df[pow], 'C{}'.format(i % 10), label=pow)
            ax.legend()
            ax.set_ylabel('Power (W)')
            ax.set_xlabel('Simulation steps')
            ax.tick_params(labelbottom='on')

        if True:
            # Plot power consumptions
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.total_electric_demand_power, 'C0', label=f'Whole Building:Facility Total {self.facility_power_output_var_suffix}')
            ax.plot(self.total_building_electric_demand_power, 'C1', label=f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix}')
            ax.plot(self.total_hvac_electric_demand_power, 'C2', label=f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix}')
            ax.legend()
            ax.set_ylabel('Power (W)')
            ax.set_xlabel('Simulation days (MM/DD)')
            ax.tick_params(labelbottom='on')

        # Show average power consumption in text
        for text in self.text_power_consumption:
            text.remove()
        self.text_power_consumption = []
        self.text_power_consumption.append(self.fig.text(0.02,  0.25, 'Whole Power:    {:6,.1f} kW'.format(
            np.average(self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)']) / 1000)))
        self.text_power_consumption.append(self.fig.text(0.02,  0.235, 'Building Power: {:6,.1f} kW'.format(
            np.average(self.df[f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix} [W](Hourly)']) / 1000)))
        self.text_power_consumption.append(self.fig.text(0.02,  0.22, 'HVAC Power:     {:6,.1f} kW'.format(
            np.average(self.df[f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix} [W](Hourly)']) / 1000)))

    #--------------------------------------------------
    # Dump timesteps
    #--------------------------------------------------
    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        def rolling_mean(data, size, que):
            out = []
            for d in data:
                que.append(d)
                if len(que) > size:
                    que.pop(0)
                out.append(sum(que) / len(que))
            return out
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))
        with open('dump_timesteps.csv', mode='w') as f:
            tot_num_rec = 0
            f.write('Sequence,Episode,Sequence in episode,Reward,tz1,tz2,power,Reward(avg1000)\n')
            que = []
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                self.read_episode(ep)
                rewards_avg = rolling_mean(self.rewards, 1000, que)
                ep_num_rec = 0
                for rew, tz1, tz2, pow, rew_avg in zip(
                        self.rewards,
                        self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'],
                        rewards_avg):
                    f.write('{},{},{},{},{},{},{},{}\n'.format(tot_num_rec, ep, ep_num_rec, rew, tz1, tz2, pow, rew_avg))
                    tot_num_rec += 1
                    ep_num_rec += 1

    #--------------------------------------------------
    # Dump episodes
    #--------------------------------------------------
    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))
        with open('dump_episodes.dat', mode='w') as f:
            tot_num_rec = 0
            f.write('#Test Ave1  Min1  Max1 STD1  Ave2  Min2  Max2 STD2   Rew     Power [22,25]1 [22,25]2  Ep\n')
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                self.read_episode(ep)
                Temp1 = self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
                Temp2 = self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)']
                Ave1, Min1, Max1, STD1 = self.get_statistics(Temp1)
                Ave2, Min2, Max2, STD2 = self.get_statistics(Temp2)
                In22_25_1 = np.sum((Temp1 >= 22.0) & (Temp1 <= 25.0)) / len(Temp1)
                In22_25_2 = np.sum((Temp2 >= 22.0) & (Temp2 <= 25.0)) / len(Temp2)
                Rew, _, _, _ = self.get_statistics(self.rewards)
                Power, _, _, _ = self.get_statistics(self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'])
                
                f.write('"{}" {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:9.2f} {:8.3%} {:8.3%} {:3d}\n'.format(self.weather_key, Ave1, Min1, Max1, STD1, Ave2,  Min2, Max2, STD2, Rew, Power, In22_25_1, In22_25_2, ep))
