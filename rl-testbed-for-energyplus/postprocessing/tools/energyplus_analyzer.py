import os, sys, time
import platform

import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import json
# if platform.system() == 'Linux':
#     matplotlib.use('Agg')
from datetime import datetime, timedelta
plt.style.use("seaborn-v0_8-colorblind")

class EnergyPlusAnalysis():
    """
    Class for analyzing EnergyPlus simulation data.

    Parameters:
    - data_dir (str): The directory where all data is stored.
    - run_dir (str): The directory inside the data directory where the specific training run is stored. (default: '')
    - episode_idx (int): The index of the episode to be analyzed. (default: 0)
    - plot_dir (str): The directory where plots are saved. (default: 'plots')
    - verbose (bool): Verbose output of the columns of the csv file or not. (default: False)
    - OS_model (bool): If an OpenStudio Model is used or not. (default: False)
    - custom_model (bool): If the custom .idf file is used or not. (default: True)
    """

    def __init__(self,
                data_dir,
                run_dir='',
                episode_idx=0,
                plot_dir='plots',
                verbose=False,
                OS_model=False,
                custom_model=True):
        """
        Initialize the EnergyPlusAnalysis class.

        Args:
        - data_dir (str): The directory where all data is stored.
        - run_dir (str): The directory inside the data directory where the specific training run is stored. (default: '')
        - episode_idx (int): The index of the episode to be analyzed. (default: 0)
        - plot_dir (str): The directory where plots are saved. (default: 'plots')
        - verbose (bool): Verbose output of the columns of the csv file or not. (default: False)
        - OS_model (bool): If an OpenStudio Model is used or not. (default: False)
        - custom_model (bool): If the custom .idf file is used or not. (default: True)
        """

        # Initialize class variables
        self.data_dir = data_dir
        self.run_dir = self.select_run_dir(run_dir)
        self.episode_idx = episode_idx
        self.verbose = verbose
        self.plot_dir = plot_dir

        # Determine the path to the csv file in the desired episode
        if self.data_dir.endswith("eplog") or self.data_dir.endswith("optunalog"):
            # If the data is in the eplog, choose the corresponding episode
            self.ep_dir, self.csv_path = self.select_episode_dir()
        else:
            # Else, search for the eplusout .csv file in the data directory
            self.csv_path = self.select_csv_file(self.data_dir)

        # Read the CSV file into a Pandas DataFrame, filling missing values
        self.df = None
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path).fillna(method='ffill').fillna(method='bfill')
            self.df = df

            rewards_file = os.path.join(self.ep_dir, 'rewards.csv')
            if os.path.exists(rewards_file):
                rewards = pd.read_csv(rewards_file)
                self.df['Reward'] = rewards['Reward']

            if OS_model is True:
                self._calculate_totals_OS()
            elif custom_model is True:
                self._calculate_totals_gym()

            # Ignore values before the first time "01/01" is found
            # This is to neglect the warmup days
            mask = self.df["Date/Time"].str.startswith(" 01/01")
            self.df = self.df.loc[mask.idxmax():,:]

            # Read date and time variables from the dataframe and create labels
            date_time = self.df['Date/Time']
            self.df.index = self._convert_datetime24(date_time)
            self.x_pos, self.x_labels, self.x_pos_minor, self.x_labels_minor = self._generate_x_pos_x_labels(self.df["Date/Time"])

            # Show the columns of the dataframe
            if self.verbose:
                self.df.info(verbose=True)

    # ------------------------ Expanding df ------------------------ #
    def _calculate_pue(self, load):
        return (load + self.df['Total ITE Load'])/self.df['Total ITE Load']
    
    def _calculate_totals_OS(self):
        # Add Power related entries to df
        self.df['Total HVAC Power'] = self.df[[ 'COOLING TOWER SINGLE SPEED 1:Cooling Tower Fan Electricity Rate [W](TimeStep)',
                'PUMP VARIABLE SPEED 3:Pump Electricity Rate [W](TimeStep)',
                'CHILLER ELECTRIC EIR 1:Chiller Electricity Rate [W](TimeStep)',
                'PUMP VARIABLE SPEED 2:Pump Electricity Rate [W](TimeStep)',
                'COMPUTERROOM ZN CRAC ELECTRIC STEAM HUMIDIFIER:Humidifier Electricity Rate [W](TimeStep)',
                'COMPUTERROOM ZN CRAC FAN:Fan Electricity Rate [W](TimeStep)']].sum(axis=1)
        self.df['Total ITE Load'] = self.df[['SMALLDATACENTERHIGHITE COMPUTERROOM IT EQUIPMENT 1:ITE CPU Electricity Rate [W](TimeStep)',
                'SMALLDATACENTERHIGHITE COMPUTERROOM IT EQUIPMENT 1:ITE Fan Electricity Rate [W](TimeStep)',
                'SMALLDATACENTERHIGHITE COMPUTERROOM IT EQUIPMENT 1:ITE UPS Electricity Rate [W](TimeStep)']].sum(axis=1)
        
        # Add PUE related entries to df
        self.df['PUE'] = (self.df['Total HVAC Power'] + self.df['Total ITE Load'])/self.df['Total ITE Load']
        self.df['PUE Cooling Tower'] = self._calculate_pue(self.df['COOLING TOWER SINGLE SPEED 1:Cooling Tower Fan Electricity Rate [W](TimeStep)'])
        self.df['PUE Condenser Pump'] = self._calculate_pue(self.df['PUMP VARIABLE SPEED 3:Pump Electricity Rate [W](TimeStep)'])
        self.df['PUE Chiller'] = self._calculate_pue(self.df['CHILLER ELECTRIC EIR 1:Chiller Electricity Rate [W](TimeStep)'])
        self.df['PUE Chiller Pump'] = self._calculate_pue(self.df['PUMP VARIABLE SPEED 2:Pump Electricity Rate [W](TimeStep)'])
        self.df['PUE Humidifier'] = self._calculate_pue(self.df['COMPUTERROOM ZN CRAC ELECTRIC STEAM HUMIDIFIER:Humidifier Electricity Rate [W](TimeStep)'])
        self.df['PUE CRAC Fan'] = self._calculate_pue(self.df['COMPUTERROOM ZN CRAC FAN:Fan Electricity Rate [W](TimeStep)'])

        # Add temperature related entries to df
        self.df['diff_temp_CT_water'] = self.df['NODE 48:System Node Temperature [C](TimeStep)']-self.df['NODE 47:System Node Temperature [C](TimeStep)']
        self.df['diff_temp_CT_water_outlet_outdoor'] = self.df['NODE 47:System Node Temperature [C](TimeStep)'] - self.df['Environment:Site Outdoor Air Wetbulb Temperature [C](TimeStep)']
        self.df['diff_temp_CT_setpoint_outdoor'] = self.df['NODE 47:System Node Setpoint Temperature [C](TimeStep)']- self.df['Environment:Site Outdoor Air Wetbulb Temperature [C](TimeStep)']
        self.df['diff_temp_CT_air'] = self.df['COOLING TOWER SINGLE SPEED 1:Cooling Tower Outlet Temperature [C](TimeStep)'] - self.df['COOLING TOWER SINGLE SPEED 1:Cooling Tower Inlet Temperature [C](TimeStep)']
        self.df['chiller_delta_T_condenser_loop'] = self.df['NODE 57:System Node Temperature [C](TimeStep)'] - self.df['NODE 51:System Node Temperature [C](TimeStep)']
        self.df['chiller_delta_T_chiller_loop'] = self.df['NODE 29:System Node Temperature [C](TimeStep)'] - self.df['NODE 34:System Node Temperature [C](TimeStep)']

    def _calculate_totals_gym(self):
        self.df['Total HVAC Power'] = self.df[[ 'COOLING TOWER SINGLE SPEED 1:Cooling Tower Fan Electricity Rate [W](TimeStep)',
                'CONDENSER WATER LOOP PUMP:Pump Electricity Rate [W](TimeStep)',
                'CHILLER ELECTRIC EIR 1:Chiller Electricity Rate [W](TimeStep)',
                'CHILLED WATER LOOP PUMP:Pump Electricity Rate [W](TimeStep)',
                'COMPUTERROOM ZN CRAC ELECTRIC STEAM HUMIDIFIER:Humidifier Electricity Rate [W](TimeStep)',
                'COMPUTERROOM ZN CRAC FAN:Fan Electricity Rate [W](TimeStep)']].sum(axis=1)
        self.df['Total ITE Load'] = self.df[['SMALLDATACENTERHIGHITE COMPUTERROOM IT EQUIPMENT 1:ITE CPU Electricity Rate [W](TimeStep)',
                'SMALLDATACENTERHIGHITE COMPUTERROOM IT EQUIPMENT 1:ITE Fan Electricity Rate [W](TimeStep)',
                'SMALLDATACENTERHIGHITE COMPUTERROOM IT EQUIPMENT 1:ITE UPS Electricity Rate [W](TimeStep)']].sum(axis=1)
        
        # Add PUE related entries to df
        self.df['PUE'] = (self.df['Total HVAC Power'] + self.df['Total ITE Load'])/self.df['Total ITE Load']
        self.df['PUE Cooling Tower'] = self._calculate_pue(self.df['COOLING TOWER SINGLE SPEED 1:Cooling Tower Fan Electricity Rate [W](TimeStep)'])
        self.df['PUE Condenser Pump'] = self._calculate_pue(self.df['CONDENSER WATER LOOP PUMP:Pump Electricity Rate [W](TimeStep)'])
        self.df['PUE Chiller'] = self._calculate_pue(self.df['CHILLER ELECTRIC EIR 1:Chiller Electricity Rate [W](TimeStep)'])
        self.df['PUE Chiller Pump'] = self._calculate_pue(self.df['CHILLED WATER LOOP PUMP:Pump Electricity Rate [W](TimeStep)'])
        self.df['PUE Humidifier'] = self._calculate_pue(self.df['COMPUTERROOM ZN CRAC ELECTRIC STEAM HUMIDIFIER:Humidifier Electricity Rate [W](TimeStep)'])
        self.df['PUE CRAC Fan'] = self._calculate_pue(self.df['COMPUTERROOM ZN CRAC FAN:Fan Electricity Rate [W](TimeStep)'])

        # Add temperature related entries to df
        self.df['diff_temp_CT_water'] = self.df['CONDENSER WATER LOOP PUMP OUTLET NODE:System Node Temperature [C](TimeStep)']-self.df['CONDENSER WATER LOOP CT OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.df['diff_temp_CT_water_outlet_outdoor'] = self.df['CONDENSER WATER LOOP CT OUTLET NODE:System Node Temperature [C](TimeStep)'] - self.df['Environment:Site Outdoor Air Wetbulb Temperature [C](TimeStep)']
        self.df['diff_temp_CT_setpoint_outdoor'] = self.df['CONDENSER WATER LOOP SUPPLY OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']- self.df['Environment:Site Outdoor Air Wetbulb Temperature [C](TimeStep)']
        self.df['diff_temp_CT_air'] = self.df['CONDENSER WATER LOOP CT OUTLET NODE:System Node Temperature [C](TimeStep)'] - self.df['Environment:Site Outdoor Air Wetbulb Temperature [C](TimeStep)']
        self.df['chiller_delta_T_condenser_loop'] = self.df['CONDENSER WATER LOOP CHILLER OUTLET NODE:System Node Temperature [C](TimeStep)'] - self.df['CONDENSER WATER LOOP CT OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.df['chiller_delta_T_chiller_loop'] = self.df['CHILLED WATER LOOP PUMP OUTLET NODE:System Node Temperature [C](TimeStep)'] - self.df['CHILLED WATER LOOP CHILLER OUTLET NODE:System Node Temperature [C](TimeStep)']
        
        
    # ------------------------ Selecting files ------------------------ #
    # Select the run directory
    def select_run_dir(self, run_dir):
        dirs = next(os.walk(self.data_dir))[1]
        matching_dirs = [d for d in dirs if d.startswith(run_dir)]

        if not matching_dirs:
            raise ValueError("No directory found with the specified prefix")

        if len(matching_dirs) > 1:
            raise ValueError("Multiple directories found with the specified prefix")
        
        return matching_dirs[0]
    
    
    # Select the directory of the episode to be analysed
    def select_episode_dir(self):
        # Define the directory where data of a specific run is stored
        episode_dirs = select_episode_dirs(self.data_dir, self.run_dir)
        
        # Define the episode directory
        ep_dir = episode_dirs[self.episode_idx]

        # Print the list of episode directories if verbose is True
        if self.verbose:
            print('\nThe following episodes are found in this directory:')
            print(*episode_dirs, sep='\n')
            print('\nThe chosen episode directory is:')
            print('ep_dir = ' + ep_dir)

        # Check for the existence of specific files ('eplusout.csv' or 'eplusout.csv.gz') in the chosen episode directory
        file_path = self.select_csv_file(ep_dir)
        
        if self.verbose:
            print('\nread_episode: file={}'.format(file_path))

        return ep_dir, file_path

    # Get the location of the correct csv file
    def select_csv_file(self, dir):
        for file in ['eplusout.csv', 'eplusout.csv.gz']:
            file_path = dir + '/' + file
            if os.path.exists(file_path):
                break
        else:
            # If neither file is found, print an error message and quit
            print('\nNo CSV or CSV.gz found under {}'.format(dir))
            # quit()
        return file_path

    # Parse date/time format from EnergyPlus and return datetime object with correction for 24:00 case
    def _parse_datetime(self, dstr):
        # ' MM/DD  HH:MM:SS' or 'MM/DD  HH:MM:SS'
        # Dirty hack
        if dstr[0] != ' ':
            dstr = ' ' + dstr
        year = 2006 
        month = int(dstr[1:3])
        day = int(dstr[4:6])
        hour = int(dstr[8:10])
        minute = int(dstr[11:13])
        sec = 0
        msec = 0
        if hour == 24:
            hour = 0
            dt = datetime(year, month, day, hour, minute, sec, msec) + timedelta(days=1)
        else:
            dt = datetime(year, month, day, hour, minute, sec, msec)
        return dt

    # Convert list of date/time string to list of datetime objects
    def _convert_datetime24(self,dates):
        # ' MM/DD  HH:MM:SS'
        dates_new = []
        for d in dates:
            dates_new.append(self._parse_datetime(d))
        return dates_new

    # Generate x_pos and x_labels
    def _generate_x_pos_x_labels(self,dates):
        time_delta  = self._parse_datetime(dates[1]) - self._parse_datetime(dates[0])
        x_pos = []
        x_labels = []
        x_pos_minor = []
        x_labels_minor = []
        for i, d in enumerate(dates):
            dt = self._parse_datetime(d) - time_delta
            # Add minor tick at start of each day
            if dt.hour == 0 and dt.minute == 0:
                x_pos_minor.append(i)
                x_labels_minor.append(dt.strftime("%d/%m"))
                # Add label at first day of every month
                if  dt.day == 1:
                    x_pos.append(i)
                    x_labels.append(dt.strftime('%d/%m'))
        return x_pos, x_labels, x_pos_minor, x_labels_minor

    # ------------------------ Plotting ------------------------ #
    # Generate simple plot of 1 variable in 1 episode
    def plot_episode_single_var(self, plot_var, plotname: str, ylabel: str, savefigures=False):
        """
        Generate a simple line plot for a single variable across an episode, with options to display and/or save the plot.

        Parameters:
        - plot_var (list or array): The variable to be plotted over the episode.
        - plotname (str): The title of the plot.
        - ylabel (str): The label for the y-axis.
        - savefigures (bool, optional): If True, saves the plot as both a PNG and SVG file in the specified plot directory.
                                        Defaults to False.

        Returns:
        - ax (matplotlib.axes._axes.Axes): The matplotlib axes object representing the generated plot.
        """

        # Check if the plot directory exists; if not, create it
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            print('Created directory: {} '.format(self.plot_dir))
        
        # Define axes object
        plt.figure()
        ax = plt.axes()

        # Plotting the data
        ax.plot(plot_var[self.x_pos_minor[2]:])

        # Adding labels and title
        ax.set_xticks(self.x_pos)
        ax.set_xticks(self.x_pos_minor, minor=True)
        ax.set_xticklabels(self.x_labels, rotation=45)
        ax.grid(visible=True)
        ax.set_ylabel(ylabel)
        ax.set_title(plotname)

        # Show plot if on windows
        plt.show()
        
        # Save the plots
        if savefigures:
            plot_path = self.plot_dir + '/' + plotname
            ax.savefig(plot_path + '.png')
            ax.savefig(plot_path + '.svg')
        # plt.close()
        return ax
    
    def plot_subplots_from_columns(self, column_lists, plot_title=None, subplot_titles=None, legend_labels=None, figsize=(12,8), savefigures=False):
        """
        Generate a subplot layout with multiple line plots from specified columns in self.df.

        Parameters:
        - PlotConfig (PlotConfig): The configuration of the plot

        Returns:
        - None

        Note:
        - Empty subplots are removed, and the overall plot layout is adjusted for better spacing.
        """

        num_subplots = len(column_lists)
        num_rows = int(np.ceil(np.sqrt(num_subplots)))
        num_cols = int(np.ceil(num_subplots / num_rows))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)

        # Flatten the axs array if it's more than 1D
        axs = axs.flatten() if num_subplots > 1 else [axs]

        # Loop through subplots
        for i, columns in enumerate(column_lists):
            subplot_data = self.df[columns]
            
            # Loop through lines and plot them (skipping the first 2 days because of random effects)
            for j, col in enumerate(subplot_data.columns):
                label = legend_labels[i][j] if legend_labels and legend_labels[i] else col
                axs[i].plot(self.df.index, subplot_data[col], label=label)
            
            # Use custom subplot titles if provided, otherwise use default titles
            title = subplot_titles[i] if subplot_titles else f'Subplot {i + 1}'
            axs[i].set_title(title)
            axs[i].legend(loc='upper left')

            
            axs[i].tick_params(axis='x', labelrotation = 45)
            axs[i].grid(visible=True)
            # axs[i].set_ylim(ymin=0)

        # Remove empty subplots, if any
        for j in range(num_subplots, num_rows * num_cols):
            fig.delaxes(axs[j])
        
        # Add a general plot title
        if plot_title:
            fig.suptitle(plot_title, fontsize=24)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Save the plot if desired
        if savefigures:
            # Check if the plot directory exists; if not, create it
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
                print('Created directory: {} '.format(self.plot_dir))
            
            # Save the plot
            plot_path = self.plot_dir + '/' + plot_title
            fig.savefig(plot_path + '.png')

    def create_plot_from_json(self, config_file, savefigures=False):
        try:
            with open(config_file, 'r') as file:
                # Load JSON data from the file
                config = json.load(file)
        
        except FileNotFoundError:
            print(f"File not found: {config_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {config_file}")
        
        # Define plot configuration
        self.plot_subplots_from_columns(
            plot_title=config['plot_title'],
            subplot_titles=config['subplot_titles'],
            legend_labels=config['legend_labels'],
            column_lists=config['column_lists'],
            figsize=(12,8),
            savefigures=savefigures
        )
    
    def _standard_analysis(self, configs_dir="configs_os_model", savefigures=False):
        # Create plots
        path_to_configs = os.path.join(os.path.dirname(__file__), configs_dir)

        # Plot changes in actions and disturbances
        action_disturbance_config = os.path.join(path_to_configs, 'actions_disturbances.json')
        self.create_plot_from_json(action_disturbance_config,savefigures)

        # Plot air loop
        air_loop_config = os.path.join(path_to_configs, 'air_loop.json')
        self.create_plot_from_json(air_loop_config,savefigures)

        # Plot chilled water loop
        chilled_water_loop_config = os.path.join(path_to_configs, 'chilled_water_loop.json')
        self.create_plot_from_json(chilled_water_loop_config,savefigures)

        # Plot condenser loop
        condenser_loop_config = os.path.join(path_to_configs, 'condensed_water_loop.json')
        self.create_plot_from_json(condenser_loop_config,savefigures)

        # Plot power
        power_config = os.path.join(path_to_configs, 'power.json')
        self.create_plot_from_json(power_config,savefigures)

        # Plot chiller
        chiller_config = os.path.join(path_to_configs, 'chiller.json')
        self.create_plot_from_json(chiller_config,savefigures)

        # Plot cooling tower
        cooling_tower_config = os.path.join(path_to_configs, 'cooling_tower.json')
        self.create_plot_from_json(cooling_tower_config,savefigures)
    
    def standard_analysis_OS(self, savefigures=False):
        self._standard_analysis("configs_os_model", savefigures=savefigures)

    def standard_analysis(self, savefigures=False):
        self._standard_analysis("configs_gym_model", savefigures=savefigures)

def select_episode_dirs(data_dir, run_dir):
    ep_dir = os.path.join(data_dir,run_dir, 'output')
    # Create a list of directory paths for episodes within the "output" directory
    list_dir = os.listdir(ep_dir)
    list_dir = sorted(list_dir, key=lambda x: int(x.split('-')[1]))
    
    episode_dirs = [
        os.path.join(ep_dir, ep)     # Full path to the episode directory
        for ep in list_dir                  # Iterate over entries in the "output" directory
        if "episode-" in ep                 # Include only directories with names containing "episode-"
    ]

    return episode_dirs