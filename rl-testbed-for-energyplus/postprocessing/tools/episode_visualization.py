import json
import os
import tkinter as tk
import numpy as np
import ast

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from postprocessing.tools.energyplus_analyzer import EnergyPlusAnalysis, select_episode_dirs
from simulation.helpers.energyplus_util import energyplus_logbase_dir, optuna_logbase_dir
from simulation.gym_energyplus.envs.energyplus_build_model import build_ep_model

sns.set_theme(style="darkgrid")
sns.color_palette(palette="colorblind6")

class LinePlotManager:
    def __init__(self, interval=1, run_dir=None, skip_ep_1=False, overrule_episode_skip=False, sns_context="notebook", simulation_type="normal"):
        # Create the parent window
        self.parent = tk.Tk()
        self.parent.title("Line Plot Viewer")

        # Set the interval and run directory
        self.interval = interval
        self.run_dir = run_dir
        self.skip_ep_1 = skip_ep_1
        self.overrule_episode_skip = overrule_episode_skip
        self.context = sns_context
        self.simulation_type = simulation_type

        # Set the constraint temperature
        self.T_constr = 22 # TODO: Automatically set this value from the settings file

        # Setup the data
        self.setup_data()
        
        # Load the JSON configuration
        try:
            with open(self.json_location, 'r') as file:
                # Load JSON data from the file
                config = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {self.json_location}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {self.json_location}")

        # Define plot configuration
        sns.set_context(self.context)
        self.plot_title = config['plot_title']
        self.subplot_titles = config['subplot_titles']
        self.legend_labels = config['legend_labels']
        self.column_lists = config['column_lists']
        self.figsize = (12, 8)
        self.num_subplots = len(self.column_lists)
        self.num_rows = int(np.ceil(np.sqrt(self.num_subplots)))
        self.num_cols = int(np.ceil(self.num_subplots / self.num_rows))

        # Reward functions
        self.reward_type = self.reward_type

        # Figure settings
        self.fig, self.axs = plt.subplots(self.num_rows, self.num_cols, figsize=self.figsize, sharex=True)
        self.axs = self.axs.flatten() if self.num_subplots > 1 else [self.axs]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize plotting
        self.current_plot_index = 0
        
        self.create_buttons()
        self.change_plot()
        self.add_toolbar()

        # Create entry box
        self.plot_index_entry = tk.Entry(self.parent)
        self.plot_index_entry.pack(side=tk.BOTTOM, pady=5)
        self.plot_index_entry.bind("<Return>", self.on_index_entry_submit)

        # Create label for entry box
        self.episode_number_label = tk.Label(self.parent, text=f"Episode number (range=0-{(len(self.df_list)-1)*self.interval}, stepsize={interval}): ")
        self.episode_number_label.pack(side=tk.BOTTOM, pady=5)

        # Create label for run directory
        self.run_dir_label = tk.Label(self.parent, text=f"Run Directory: {self.run_dir}")
        self.run_dir_label.pack(side=tk.BOTTOM, pady=5)

    def setup_data(self):
        # Set up the data directory
        if self.simulation_type == "normal":
            log_dir = energyplus_logbase_dir()
        elif self.simulation_type == "optuna":
            log_dir = optuna_logbase_dir()
        else:
            raise ValueError(f"Invalid simulation type {self.simulation_type}, use either 'normal' or 'optuna'.")
        
        list_dir = sorted(os.listdir(log_dir))
        if self.run_dir is None:
            self.run_dir = list_dir[-1]      # Selects newest simulation run
        data_dir = os.path.join(log_dir, self.run_dir)
        episode_dirs = select_episode_dirs(log_dir, self.run_dir)

        # Check if evaluations is set to true or not
        episodes_to_ignore = 1
        self.reward_type = None
        self.alpha = None
        self.beta = None
        self.c = None

        # Define settings path
        settings_path = os.path.join(data_dir, 'settings.txt')
        evaluation_setting_string = 'evaluate_agent: True'
        
        # Run through settings files
        if (not self.overrule_episode_skip) and os.path.exists(settings_path):
            with open(settings_path, 'r') as file:
                for line in file:
                    # Check if evaluation is set to true
                    if evaluation_setting_string in line:
                        episodes_to_ignore = 3


        # Define location of the config
        if self.context == "notebook":
            self.json_location = '/home/peppie/RL-Project-Pepijn/rl-testbed-for-energyplus/common/configs_gym_model/actions_disturbances_with_reward.json'
        elif self.context == "talk":  
            self.json_location = '/home/peppie/RL-Project-Pepijn/rl-testbed-for-energyplus/common/configs_gym_model/actions_disturbances_for_communication.json'
            # self.json_location = '/home/peppie/RL-Project-Pepijn/rl-testbed-for-energyplus/common/configs_gym_model/air_loop.json'

        # Run through episodes
        df_list = []
        if not self.skip_ep_1:
            data = EnergyPlusAnalysis(data_dir=log_dir, run_dir=self.run_dir, episode_idx=0)
            df_list.append(data.df)
            print("Added episode 0")
        
        for i, episode in enumerate(episode_dirs[1:-episodes_to_ignore]):
            select_episode = (i)%self.interval
            if (select_episode == 0) or (i == 0):
                data = EnergyPlusAnalysis(data_dir=log_dir, run_dir=self.run_dir, episode_idx=i+1)
                if data.df is not None:
                    df_list.append(data.df)
                    print(f"Added episode {i+1}")

        # Append evaluation run
        if episodes_to_ignore == 3:
            data = EnergyPlusAnalysis(data_dir=log_dir, run_dir=self.run_dir, episode_idx=len(episode_dirs)-2)
            df_list.append(data.df)

        self.df_list = df_list

    def on_index_entry_submit(self, event):
        try:
            plot_index = int(self.plot_index_entry.get())
            if plot_index < 0 or plot_index >= len(self.df_list):
                raise ValueError("Invalid plot index")
            self.current_plot_index = plot_index
            self.change_plot()
        except ValueError as e:
            print("Error:", e)

    def create_buttons(self):
        button_frame = tk.Frame(self.parent)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        prev_button = tk.Button(button_frame, text='Previous', command=self.show_previous_plot)
        prev_button.pack(side=tk.LEFT, padx=10)

        next_button = tk.Button(button_frame, text='Next', command=self.show_next_plot)
        next_button.pack(side=tk.LEFT, padx=10)

    def show_previous_plot(self):
        self.current_plot_index -= 1
        if self.current_plot_index < 0:
            self.current_plot_index = len(self.df_list) - 1
        self.change_plot()

    def show_next_plot(self):
        self.current_plot_index += 1
        if self.current_plot_index >= len(self.df_list):
            self.current_plot_index = 0
        self.change_plot()

    def create_plot_actions_disturbances(self, df):
        for ax in self.axs:
            ax.clear()

        for i, columns in enumerate(self.column_lists):
            subplot_data = df[columns]
            
            for j, col in enumerate(subplot_data.columns):
                label = self.legend_labels[i][j] if self.legend_labels and self.legend_labels[i] else col
                self.axs[i].plot(df.index, subplot_data[col], label=label)
            
            title = self.subplot_titles[i] if self.subplot_titles else f'Subplot {i + 1}'
            self.axs[i].set_title(title)
            self.axs[i].legend(loc='upper left')
            self.axs[i].tick_params(axis='x', labelrotation = 45)
            self.axs[i].grid(visible=True)

        if self.current_plot_index == 0:
            episode_nr = 0
        else:
            episode_nr = self.current_plot_index*self.interval

        if self.plot_title:
            title = f"{self.plot_title}, Episode = {episode_nr}"
        else:
            title = f"Episode = {episode_nr}"
        self.fig.suptitle(title, fontsize=24)
        
        plt.tight_layout()
        self.canvas.draw()

    def change_plot(self):
        self.create_plot_actions_disturbances(self.df_list[self.current_plot_index])

    def add_toolbar(self):
        toolbar = NavigationToolbar2Tk(self.canvas, self.parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def run_gui(interval:int = 1, run_dir:str=None, skip_ep_1=False, overrule_episode_skip=False, sns_context='talk', simulation_type="normal"):
    # Create LinePlotManager instance
    line_plot_manager = LinePlotManager(interval, run_dir, skip_ep_1, overrule_episode_skip, sns_context=sns_context, simulation_type=simulation_type)

    # Run the Tkinter event loop
    line_plot_manager.parent.mainloop()


if __name__=='__main__':
    # Define the run directory
    # run_dir = "SB3_24_04_21_09h18_optuna_108"
    run_dir = None
    overrule_episode_skip=False
    sns_context = "talk"
    sim_type = "normal"
    # sim_type = "optuna"

    # Run the GUI
    run_gui(run_dir=run_dir, overrule_episode_skip=overrule_episode_skip,sns_context=sns_context, simulation_type=sim_type)