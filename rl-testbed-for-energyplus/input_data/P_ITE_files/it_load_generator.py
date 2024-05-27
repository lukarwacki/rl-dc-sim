import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import seaborn as sns

sns.set_theme()

# Generate time stamps for the year
def IT_load_generator(initial_base_load, random_walk_factor, max_day_night_load_amplitude, noise_factor, plot=False):
    # Define the total number of hours in a year
    total_hours = 24*366    # Account for leap year
    day_hours = 24

    # Generate time stamps for the year1
    start_date = datetime.datetime(2024, 1, 1)
    time_stamps = [start_date + datetime.timedelta(hours=i) for i in range(total_hours)]

    # Generate load data
    load_data = []
    day_night_load_amplitude = 0
    walk = initial_base_load
    for hour in range(total_hours):
        # Increase base load gradually over the year
        step = np.random.normal(0, random_walk_factor) * initial_base_load
        walk += step
        
        # Calculate the hourly load increase using a sine function
        hour_of_day = hour % day_hours
        if hour_of_day % 12 == 0:
            day_night_load_amplitude = max_day_night_load_amplitude*np.random.uniform(0.2, 1.8)
        day_night_load_increase = day_night_load_amplitude * np.sin(2 * np.pi * hour_of_day / day_hours)
        
        # Introduce noise
        noise = np.random.normal(0, noise_factor)

        # Compute the load for this hour
        load = np.array(walk + day_night_load_increase + noise)
        load_data.append(load)

    # Normalize the load data
    min_load = np.min(load_data)
    max_load = np.max(load_data)

    load_data = (load_data - min_load) / (max_load - min_load)

    # Plot the load data
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_stamps, load_data, color='blue')
        plt.title('IT Load of Server Over the Course of a Year')
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.grid(True)
        plt.show()

    # Create a pandas DataFrame
    df = pd.DataFrame(load_data, columns=['Load'])

    return df

# Generate files
def IT_load_csv(number_of_files, file_base_name, file_dir, initial_base_load, random_walk_factor, max_daily_load_amplitude, noise_factor):
    # Create the directory if it does not exist
    if not os.path.exists(os.path.join(os.getcwd(), file_dir)):
        os.makedirs(os.path.join(os.getcwd(), file_dir))
    
    # Generate the files
    for i in range(number_of_files):
        df = IT_load_generator(initial_base_load, random_walk_factor, max_daily_load_amplitude, noise_factor, plot=True)
        file_name = f"{file_base_name}_{i}.csv"
        csv_path = os.path.join(file_dir,file_name)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # Generation settings
    number_of_files = 6
    file_base_name = 'IT_Load'
    file_dir = 'rl-testbed-for-energyplus/input_data/P_ITE_files_temp'


    # Define parameters
    initial_base_load = 1  # Initial base load of the server
    random_walk_factor = 0.01  # Maximum increase in base load
    max_daily_load_amplitude = 0.3  # Amplitude of the hourly load sine function
    noise_factor = 0.2*max_daily_load_amplitude  # Factor for introducing noise

    IT_load_csv(number_of_files, file_base_name, file_dir, initial_base_load, random_walk_factor, max_daily_load_amplitude, noise_factor)
