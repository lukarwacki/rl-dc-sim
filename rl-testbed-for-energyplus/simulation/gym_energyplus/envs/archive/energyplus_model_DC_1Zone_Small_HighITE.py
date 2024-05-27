# Created on 2023-12-05
# Author: Pepijn Six Dijkstra

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
class EnergyPlusModel1ZoneDCSmallHighITE(EnergyPlusModel):

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
        super(EnergyPlusModel1ZoneDCSmallHighITE, self).__init__(model_file, log_dir, verbose)
        
        # Set a lower limit for the reward value
        self.reward_low_limit = -10000.
        
        # Initialize variables related to plotting (RL)
        self.axepisode = None   # Placeholder for RL episode data axis
        self.num_axes = 5       # Number of axes used in plot
        
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
        # TODO: Double-check limits for action variables
        # Chiller setpoint temperature limits
        chiller_setpoint_temperature_lower_limit = 5.0
        chiller_setpoint_temperature_upper_limit = 40.0

        # Chilled water mass flow limits
        chilled_water_mass_flow_lower_limit = 1
        chilled_water_mass_flow_upper_limit = 10

        # Fan air mass flow limits
        fan_air_mass_flow_upper_limit = 10
        fan_air_mass_flow_lower_limit = 0.1 * fan_air_mass_flow_upper_limit  # (set to 10% of upper limit)

        # Set action space
        self.action_space = spaces.Box(
            low = np.array([chiller_setpoint_temperature_lower_limit, chilled_water_mass_flow_lower_limit, fan_air_mass_flow_lower_limit]),
            high = np.array([chiller_setpoint_temperature_upper_limit, chilled_water_mass_flow_upper_limit, fan_air_mass_flow_upper_limit]),
            dtype = np.float32
        )

        # TODO: Double check limits for observation space
        # Observation space overall limits
        observation_lower_limit = -100.0
        observation_upper_limit = 10000000.0

        # Observation space shape
        observation_space_shape = (38,)

        # Set observation space
        self.observation_space = spaces.Box(
            low=observation_lower_limit,
            high=observation_upper_limit,
            shape=observation_space_shape,
            dtype=np.float32
        )

    def set_raw_state(self, raw_state):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = np.zeros(shape=self.observation_space.shape)
    
    def compute_reward(self):
        rew = 1 #TODO: Add reward computation
        return rew
    
    def format_state(self, raw_state):
        # Perform mapping from raw state to the gym compatible state
        return np.array(raw_state)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
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

        # Check if the provided episode is a string (file path) or an index in episode_dirs
        if type(ep) is str:
            # If it's a string, use it as the file path
            file_path = ep
        else:
            # If it's an index, get the corresponding episode directory
            ep_dir = self.episode_dirs[ep]
            
            # Check if eplusout.csv or .csv.gz exists in the episode directory
            for file in ['eplusout.csv', 'eplusout.csv.gz']:
                # Construct the full file path
                file_path = os.path.join(ep_dir, file)
                
                # Check if the file exists
                if os.path.exists(file_path):
                    # If found, break out of the loop
                    break
            else:
                # If neither eplusout.csv nor .csv.gz is found, print a message and quit
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
        # self.df.info(verbose=True) #TODO: Remove this when function is finished

        # Extract the weather key and temperature data for different zones
        epw_files = glob(os.path.dirname(file_path) + '/USA_??_*.epw')
        if len(epw_files) == 1:
            self.weather_key = os.path.basename(epw_files[0])[4:6]
        else:
            self.weather_key = '  '

        # Extract air temperatures
        self.outdoor_temp = df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)']
        # TODO: Extract outdoor & indoor air temperatures from the DataFrame
        # TODO: Extract the PUE value from the DataFrame
        # TODO: Extract zone & HVAC data from DataFrame
        # TODO: Extract electric power data
        
        # Initialize reward list
        self.rewards = []

        # TODO: Compute rewards per time step
        # TODO: Set cooling & heating setpoints

        # Generate x_pos and x_labels for plotting
        (self.x_pos, self.x_labels) = self.generate_x_pos_x_labels(date)
    
    def plot_episode(self, ep):
        print('episode {}'.format(ep))

        # Read episode data and process it
        self.read_episode(ep)

        # Show statistics for different variables
        # TODO: Show these statistics (is just printed so not important for now)

        # Show distribution of the West Zone tempearture 
        # TODO: Show this distribution later (if even necessary)

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
        
        # Initialize index variable
        idx = 0

        # TODO: Add plotting part of function
        if True:
            # Plot zone and outdoor temperature
            ax = self.axepisode[idx]
            idx += 1

            ax.plot()
      
    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        """
        Process simulation data for each timestep across multiple episodes and write relevant information to a CSV file.
        This function calculates rolling averages of rewards over 1000 steps and records episode-wise data, including sequence,
        episode number, sequence within the episode, rewards, temperatures in two zones, power consumption, and the rolling mean
        of rewards. The results are stored in 'dump_timesteps.csv' for further analysis.

        Parameters:
            log_dir (str): The directory containing episode log files.
            csv_file (str): The CSV file containing episode data.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # TODO: Test this function
        # Define a helper function for calculating rolling mean
        def rolling_mean(data, size, que):
            out = []
            for d in data:
                que.append(d)
                if len(que) > size:
                    que.pop(0)
                out.append(sum(que) / len(que))
            return out
        
        # Get the list of episodes
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        
        # Print the number of episodes
        print('{} episodes'.format(self.num_episodes))
        
        # Open a CSV file for writing the results
        with open('dump_timesteps.csv', mode='w') as f:
            tot_num_rec = 0
            # Write header to the CSV file
            f.write('Sequence,Episode,Sequence in episode,Reward,tz1,tz2,power,Reward(avg1000)\n')
            que = []    # Initialize a queue for rolling mean calculation

            # Loop over each episode
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                
                # Read the data for the current episode and calculate the rollingmean
                self.read_episode(ep)
                rewards_avg = rolling_mean(self.rewards, 1000, que)
                ep_num_rec = 0

                # Write data to the CSV file for each time step in the episode
                for rew, tz1, tz2, pow, rew_avg in zip(
                        self.rewards,
                        # TODO: Change these values to the correct zone temps
                        self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'],
                        rewards_avg):
                    f.write('{},{},{},{},{},{},{},{}\n'.format(tot_num_rec, ep, ep_num_rec, rew, tz1, tz2, pow, rew_avg))
                    tot_num_rec += 1
                    ep_num_rec += 1
    
    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        """
        Dump episode statistics to a file for analysis. For each episode, this function extracts and calculates various
        statistics related to temperature, rewards, and power consumption. The statistics include average, minimum, maximum,
        and standard deviation of temperatures in two different zones, as well as the percentage of time temperatures are
        between 22 and 25 degrees Celsius. Additionally, statistics for rewards and facility power output are calculated.
        The results are written to a file in a structured format for further analysis.

        Parameters:
            log_dir (str): The directory containing episode log files.
            csv_file (str): The CSV file containing episode data.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # TODO: Test this function
        # Get the list of episodes & print the amount of episodes
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))

        # Open a file for writing episode statistics
        with open('dump_episodes.dat', mode='w') as f:
            tot_num_rec = 0

            # Write header for the file
            f.write('#Test Ave1  Min1  Max1 STD1  Ave2  Min2  Max2 STD2   Rew     Power [22,25]1 [22,25]2  Ep\n')

            # Loop over each episode
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))

                # Read data of the current episode
                self.read_episode(ep)

                # Extract temperature data for each zone
                # TODO: Extract temperature of the correct zone
                Temp1 = self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
                Temp2 = self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)']

                # Calculate statistics for temperature in each zone
                Ave1, Min1, Max1, STD1 = self.get_statistics(Temp1)
                Ave2, Min2, Max2, STD2 = self.get_statistics(Temp2)

                # Calculate the percentage of time temperature is between 22 and 25 degrees Celsius in each zone TODO: Doublecheck these values
                In22_25_1 = np.sum((Temp1 >= 22.0) & (Temp1 <= 25.0)) / len(Temp1)
                In22_25_2 = np.sum((Temp2 >= 22.0) & (Temp2 <= 25.0)) / len(Temp2)

                # Calculate statistics for the rewards and power
                Rew, _, _, _ = self.get_statistics(self.rewards)
                Power, _, _, _ = self.get_statistics(self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'])
                
                # Write the statistics to the file for the current episode
                f.write('"{}" {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:9.2f} {:8.3%} {:8.3%} {:3d}\n'.format(self.weather_key, Ave1, Min1, Max1, STD1, Ave2,  Min2, Max2, STD2, Rew, Power, In22_25_1, In22_25_2, ep))

def test_EPlusModel_1Zone():
    # Variables
    model_file = os.getenv('ENERGYPLUS_MODEL')
    log_dir = 'log_test'

    # Test functions
    ep_model = EnergyPlusModel1ZoneDCSmallHighITE(model_file=model_file, log_dir=log_dir)
    ep_model.set_raw_state(raw_state=None)
    ep_model.compute_reward

    return ep_model

if __name__ == "__main__":
    test_ep_model = test_EPlusModel_1Zone()