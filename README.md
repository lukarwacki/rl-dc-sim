[![unit tests](https://github.com/IBM/rl-testbed-for-energyplus/actions/workflows/test.yml/badge.svg)](https://github.com/IBM/rl-testbed-for-energyplus/actions/workflows/test.yml)

# Project Description
Reinforcement Learning Testbed for Power Consumption Optimization, modified for use on a single zone data center with complete HVAC model. This repository is based on the original testbed for EnergyPlus developed by IBM Research (the original code can be found [here](https://github.com/IBM/rl-testbed-for-energyplus) and paper [here](https://arxiv.org/abs/1808.10427)). The testbed is designed to evaluate the performance of reinforcement learning (RL) algorithms in optimizing power consumption of HVAC systems in buildings. The testbed uses EnergyPlus, a widely used building energy simulation tool, as the simulation environment. The testbed provides a Python interface to interact with EnergyPlus and to implement RL algorithms.

# Overview
The testbed consists of two main components: the EnergyPlus simulation environment and the Python interface. The EnergyPlus simulation environment is a building energy simulation tool that simulates the energy consumption of HVAC systems in buildings. The Python interface provides an API to interact with EnergyPlus, including setting the environment variables, running simulations, and retrieving the simulation results. The testbed also includes a set of predefined building models and weather data files for different locations.

**<u>Installation</u>** Running the project can be done using a Docker container (preferred) or by following the installation instructions below. The Docker container is the easiest way to run the project, more information on this can be found in [rl-testbed-for-energyplus/docker/README.md](rl-testbed-for-energyplus/docker/README.md). The installation instructions below provide a step-by-step guide to setting up the project natively on your machine.

**<u>Background information</u>** More information on how the repository is structured and how the code works can be found [here](rl-testbed-for-energyplus/README.md).

# Installation instructions

**<u>Note:</u>** The preferred option for installing and running the code is to use the Docker container. The Docker container automates the setup and ensures a reproducible work environment. More information on this can be found in the Docker README file [rl-testbed-for-energyplus/docker/README.md](rl-testbed-for-energyplus/docker/README.md).


If due to any reason you cannot use the Docker container, you can follow the installation instructions below to set up the project natively on your machine. <!-- The following steps will be followed in these installation instructions:--!>
<!-- 1. Install EnergyPlus prebuilt package:
	- Download the pre-built package of EnergyPlus and install it. This package provides pre-compiled binaries and data files that cannot be generated from source code.
2. Clone this repo.
3. Patch EnergyPlus such that it can interact with the python interface.
4. Set up a python environment.
5. Set up environment variables in the `.bashrc` file.
6. Run the code! -->

**<u>Note:</u>** While the original rl-testbed by IBM supports several EnergyPlus versions, the customized data center models in this code only work with EnergyPlus 9.5.0

## Supported platforms

The following platform has been tested:
- Ubuntu 20.04 LTS

The following platforms have been tested in the code from IBM:
- macOS High Sierra (Version 10.13.6)
- macOS Catalina (Version 10.15.3)


## Download prebuilt EnergyPlus installation

You can download every energyplus installer at https://github.com/NREL/EnergyPlus/releases/. Download the correct file for your platform (EnergyPlus version = 9.5.0). The file should end with a `.sh` extenstion.

### Installation when using Ubuntu

1. Download the prebuilt energyplus installer
	- Download the link directly using the provided link and the following commands:
	```
	$ cd <download_directory>
	$ wget https://github.com/NREL/EnergyPlus/releases/download/v9.5.0/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu20.04-x86_64.sh
	```

	OR
	-	Go to the web page shown above. Right click on relevant link in supported versions table and select `Save link As` to from the menu to download installation image.

	<!-- 3. (For Linux users with EnergyPlus version 9.1.0 only) Apply patch on the downloaded file (EnergyPlus 9.1.0 installation script unpacks in /usr/local instead of /usr/local/EnergyPlus-9.1.0).
	```
	$ cd <DOWNLOAD-DIRECTORY>
	$ patch -p0 < rl-testbed-for-energyplus/EnergyPlus/EnergyPlus-9.1.0-08d2e308bb-Linux-x86_64.sh.patch
	``` -->
3. Execute installation image. Below example is for EnergyPlus 9.5.0
	```
	$ sudo bash <download_directory>/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu20.04-x86_64.sh
	```

4. Enter your admin password if required.
Specify `/usr/local/EnergyPlus-9-5-0` for install directory.
Respond with `/usr/local/bin` if asked for symbolic link location.
The package will be installed at `/usr/local/EnergyPlus-9-5-0`

### Installation using macOS (Note, this hasn't been tested yet)

1. Go to the web page shown above.
2. Right click in supported versions table and select `Save link As` to from the menu to download installation image.
1. Double click the downloaded package, and follow the instructions.
The package will be installed in `/Applications/EnergyPlus-9-5-0`.

## Build patched EnergyPlus

<!-- First set up ssh with Azure DevOps and your machine (**TODO:** figure out how users outside of Coolgradient can do this). -->
### Installation using Ubuntu
Clone the RL-Project repo and the energyplus repo into the directory where you will work, from now on called: `<WORKING_DIRECTORY>`. The energyplus repo will be used to apply the patch and build the EnergyPlus executable.

```
$ cd <WORKING-DIRECTORY>
$ git clone git@ssh.dev.azure.com:v3/coolgradient-internal/research/RL-Project-Pepijn
$ cd RL-Project-Pepijn
$ git clone -b v9.5.0 https://github.com/NREL/EnergyPlus.git
```

Apply patch to EnergyPlus and build. If you will be using another EnergyPlus version, replace `9-5-0` by the correct version. The patch file is located at `rl-testbed-for-energyplus/EnergyPlus/RL-patch-for-EnergyPlus-9-5-0.patch`.

```
$ cd <WORKING-DIRECTORY>/EnergyPlus
$ patch -p1 < ../rl-testbed-for-energyplus/EnergyPlus/RL-patch-for-EnergyPlus-9-5-0.patch
```
Create a build directory and install the patch.
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local/EnergyPlus-9-5-0 ..    # Ubuntu case (please don't forget the two dots at the end)
$ cmake -DCMAKE_INSTALL_PREFIX=/Applications/EnergyPlus-9-5-0 .. # macOS case (please don't forget the two dots at the end)
```
Generate the executable using the following command, this may take up to **1.5 hours**.
```
$ make -j4
```

After building the executable, you can install it using the following command.

```
$ sudo make install
```
### Installation using MacOS
Clone the RL-Project repo and the energyplus repo into the directory where you will work, from now on called: `<WORKING_DIRECTORY>`. The energyplus repo will be used to apply the patch and build the EnergyPlus executable.

```
$ cd <WORKING-DIRECTORY>
$ git clone git@ssh.dev.azure.com:v3/coolgradient-internal/research/RL-Project-Pepijn
$ cd RL-Project-Pepijn
$ git clone -b v9.5.0 https://github.com/NREL/EnergyPlus.git
```

Apply patch to EnergyPlus and build. If you will be using another EnergyPlus version, replace `9-5-0` by the correct version. The patch file is located at `rl-testbed-for-energyplus/EnergyPlus/RL-patch-for-EnergyPlus-9-5-0.patch`.

```
$ cd <WORKING-DIRECTORY>/EnergyPlus
$ patch -p1 < ../rl-testbed-for-energyplus/EnergyPlus/RL-patch-for-EnergyPlus-9-5-0.patch
```
Create a build directory and install the patch.
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/Applications/EnergyPlus-9-5-0 .. # (please don't forget the two dots at the end)
```
Generate the executable using the following command, this may take up to **1.5 hours**.
```
$ make -j4
```

After building the executable, you can install it using the following command.

```
$ sudo make install
```

## Python set up

This code is written in Python and uses the OpenAI Baselines library. To run the code, you must set up a Python environment using the required packages.

Create an environment using python `version==3.11.5`. For example:
```
$ conda create --name RL-testbed python=3.11.5
$ conda activate RL-testbed
```
Use the `requirements.txt` file to install the required packages:
```
$ cd <working_directory>
$ pip install -r requirements.txt
```

If you get the error `ERROR: Failed building wheel for mpi4py`, first install mpi4py using conda before installing the requirements:
```
$ cd <working_directory>
$ conda install mpi4py==3.1.3
$ pip install -r requirements.txt
```

## Environment variables setup

Some environment variables must be defined in order for the program to work correctly.

EnergyPlus related:
- The `ENERGYPLUS` variable specifies the EnergyPlus installation.

The following variables are used for the input of the model:
- The `ENERGYPLUS_MODEL` variable specifies the EnergyPlus model file to be used. 
- The `ENERGYPLUS_WEATHER` variable specifies the weather file to be used. 
- The `ENERGYPLUS_IT_LOAD` variable specifies the IT load file to be used. 

The following variables specify locations of directories:
- The `ENERGYPLUS_LOGBASE` variable specifies the directory where the EnergyPlus log files will be stored.
- The `TENSORBOARD_LOGBASE` variable specifies the directory where the TensorBoard log files will be stored. 
- The `OPTUNA_LOGBASE` variable specifies the directory where the Optuna log files will be stored.
- The `P_ITE_DIR` variable specifies the directory where the yearly IT loads are stored.
- The `WEATHER_DIR` variable specifies the directory where the weather files are stored.

Copy the following lines to the end of `$~/.bashrc` and replace `<WORKING_DIRECTORY>` with your own working directory.  When it is copied, run the command `$ source ~/.bashrc` to activate the changes.

```
# Specify the top directory
TOP=<WORKING_DIRECTORY>/rl-testbed-for-energyplus
export PYTHONPATH=${PYTHONPATH}:${TOP}

if [ `uname` == "Darwin" ]; then   # If on macOS
	energyplus_instdir="/Applications"
else
	energyplus_instdir="/usr/local"
fi

ENERGYPLUS_VERSION="9-5-0"

ENERGYPLUS_DIR="${energyplus_instdir}/EnergyPlus-${ENERGYPLUS_VERSION}"
export ENERGYPLUS="${ENERGYPLUS_DIR}/energyplus"
MODEL_DIR="${TOP}/input_data/EnergyPlus/Model-${ENERGYPLUS_VERSION}"

INPUT_DATA_DIR="${TOP}/input_data"
export P_ITE_DIR="${INPUT_DATA_DIR}/P_ITE_files"
export WEATHER_DIR="${INPUT_DATA_DIR}/WeatherData"

# Weather file.
# Single weather file or multiple weather files separated by comma character.

# export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_FL_Tampa.Intl.AP.722110_TMY3.epw"
export ENERGYPLUS_WEATHER="${WEATHER_DIR}/Weather_amsterdam_2018.epw,${WEATHER_DIR}/Weather_amsterdam_2019.epw,${WEATHER_DIR}/Weather_amsterdam_2020.epw,${WEATHER_DIR}/Weather_amsterdam_2020.epw,${WEATHER_DIR}/Weather_amsterdam_2021.epw"
# export ENERGYPLUS_WEATHER="${WEATHER_DIR}/Weather_amsterdam_2023.epw"

# IT load file
export ENERGYPLUS_IT_LOAD="${P_ITE_DIR}/IT_Load_0.csv,${P_ITE_DIR}/IT_Load_1.csv,${P_ITE_DIR}/IT_Load_2.csv,${P_ITE_DIR}/IT_Load_3.csv"

# Ouput directory "openai-YYYY-MM-DD-HH-MM-SS-mmmmmm" is created in
# the directory specified by ENERGYPLUS_LOGBASE or in the current directory if not specified.
LOGBASE="${TOP}/output_data"
export ENERGYPLUS_LOGBASE="${LOGBASE}/eplog"
export TENSORBOARD_LOGBASE="${LOGBASE}/tblog"
export OPTUNA_LOGBASE="${LOGBASE}/optunalog"

# Model file. Uncomment one.
export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_ChillerTemp_ChillerMass_AirMass.idf"    # 1 Zone dc with all 3 action variables
# export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_ChillerTemp_AirTemp_ChillerMass_AirMass.idf"    # 1 Zone dc used for case studies
# export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_ChillerTemp_AirTemp.idf"                # 1 Zone dc with only temperature control used for baseline study

# Older IDFs
# export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_T_chill_SP.idf"                         # 1 Zone dc with only T_chill_SP control
# export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_T_chill_SP_reduced_state.idf"           # 1 Zone dc with only T_chill_SP control and reduced states
# export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_baseline.idf"                           # 1 Zone dc with T_chill_SP and T_air_SP as action for setting baseline study
# export ENERGYPLUS_MODEL="${MODEL_DIR}/RL_DC_continuous_study.idf"                   # 1 Zone dc with both mass flows and T_SP's as action for studying effects of changing ITE 
# export ENERGYPLUS_MODEL="${MODEL_DIR}/timestep_study/RL_DC_1_ph.idf"                # 1 Zone dc with 1 time step per hour

# Run command (example)
# $ time python3 -m baselines_energyplus.trpo_mpi.run_energyplus --num-timesteps 1000000000

# Monitoring (example)
# $ python3 -m common.plot_energyplus
```