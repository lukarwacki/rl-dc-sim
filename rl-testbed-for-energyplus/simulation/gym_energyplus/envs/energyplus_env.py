# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE
import os
import subprocess
import warnings
import gzip
import shutil
import shlex
from glob import glob
from argparse import ArgumentParser

import numpy as np
from gym import Env
from gym.utils import seeding

from simulation.gym_energyplus.envs.pipe_io import PipeIo
from simulation.gym_energyplus.envs.energyplus_build_model import build_ep_model

class EnergyPlusEnv(Env):
    """
    Gym environment for interacting with EnergyPlus simulation.

    Parameters:
    - energyplus_file (str): Path to the EnergyPlus executable file. If not specified, it will be obtained from the ENERGYPLUS environment variable.
    - model_file (str): Path to the EnergyPlus model file. If not specified, it will be obtained from the ENERGYPLUS_MODEL environment variable.
    - weather_file (str): Path to the EnergyPlus weather file. If not specified, it will be obtained from the ENERGYPLUS_WEATHER environment variable.
    - log_dir (str): Directory to store the log files. If not specified, it will be obtained from the ENERGYPLUS_LOG environment variable or default to 'log'.
    - verbose (bool): Whether to print verbose output. Defaults to False.
    - seed (int): Random seed for reproducibility.
    - framework (str): Framework to use for interacting with the EnergyPlus model. Defaults to "openai".

    Attributes:
    - metadata (dict): Metadata for the environment.

    Methods:
    - seed(seed=None): Set the random seed for the environment.
    - reset(): Reset the environment and start a new episode.
    - step(action): Take a step in the environment given an action.
    - start_instance(): Start a new EnergyPlus simulation instance.
    - stop_instance(): Stop the current EnergyPlus simulation instance.
    - send_action(): Send an action to the EnergyPlus simulation instance.
    - receive_observation(): Receive an observation from the EnergyPlus simulation instance.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 energyplus_file=None,
                 model_file=None,
                 weather_file=None,
                 IT_load_file=None,
                 log_dir=None,
                 verbose=False,
                 seed=None,
                 reward_file=None,
                 framework="openai"):
        self.energyplus_process = None
        self.pipe_io = None
        self.framework = framework

        # Verify path arguments
        if energyplus_file is None:
            energyplus_file = os.getenv('ENERGYPLUS')
        if energyplus_file is None:
            raise ValueError('energyplus_env: FATAL: EnergyPlus executable is not specified. Use environment variable ENERGYPLUS.')
        
        if model_file is None:
            model_file = os.getenv('ENERGYPLUS_MODEL')
        if model_file is None:
            raise ValueError('energyplus_env: FATAL: EnergyPlus model file is not specified. Use environment variable ENERGYPLUS_MODEL.')
        
        if weather_file is None:
            weather_file = os.getenv('ENERGYPLUS_WEATHER')
        if weather_file is None:
            raise ValueError('energyplus_env: FATAL: EnergyPlus weather file is not specified. Use environment variable ENERGYPLUS_WEATHER.')
        
        if IT_load_file is None:
            IT_load_file = os.getenv('ENERGYPLUS_IT_LOAD')
        if IT_load_file is None:
            warnings.warn('energyplus_env: WARNING: EnergyPlus IT load file is not specified. Use environment variable ENERGYPLUS_IT_LOAD.')

        if log_dir is None:
            log_dir = os.getenv('ENERGYPLUS_LOG')
        if log_dir is None:
            log_dir = 'log'

        if reward_file is None:
            warnings.warn('energyplus_env: WARNING: EnergyPlus reward file is not specified. Continuing with standard reward settings.')
        
        # Initialize paths
        self.energyplus_file = energyplus_file
        self.model_file = model_file
        self.weather_files = weather_file.split(',')
        self.IT_load_files = IT_load_file.split(',')
        self.log_dir = log_dir
        self.reward_file = reward_file
        
        # Create an EnergyPlus model
        self.ep_model = build_ep_model(model_file = self.model_file, log_dir = self.log_dir, reward_file = self.reward_file)

        self.action_space = self.ep_model.action_space
        self.observation_space = self.ep_model.observation_space
        # TODO: self.reward_space which defaults to [-inf,+inf]
        self.pipe_io = PipeIo()

        self.episode_idx = -1
        self.verbose = verbose

        self.seed(seed)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        # In case you forget to call env.stop()
        self.stop_instance()
        
    def reset(self):
        self.stop_instance()

        self.episode_idx += 1
        print(f"Episode {self.episode_idx} has finished")
        self.start_instance()
        self.timestep1 = 0
        self.ep_model.reset()
        return self.step(None)[0]

    def start_instance(self):
        print('Starting new environment')
        assert(self.energyplus_process is None)

        output_dir = self.get_output_dir()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.pipe_io.start()
        print('start_instance(): idx={}, model_file={}'.format(self.episode_idx, self.model_file))
        
        # Handling weather file override
        weather_files_override = glob(self.log_dir + '/*.epw')
        if len(weather_files_override) > 0:
            weather_files_override.sort()
            weather_files = weather_files_override
            print('start_instance(): weather file override')
        else:
            weather_files = self.weather_files

        # Handling of multiple weather files
        weather_idx = self.episode_idx % len(weather_files)
        weather_file = weather_files[weather_idx]

        # Handling IT load file override
        IT_load_files_override = glob(self.log_dir + '/IT_load.csv')
        if len(IT_load_files_override) > 0:
            IT_load_files_override.sort()
            IT_load_files = IT_load_files_override
            print('start_instance(): IT load file override')
        elif self.IT_load_files is not None:
            IT_load_files = self.IT_load_files
        else:
            IT_load_files = None

        if IT_load_files is not None:
            IT_load_idx = self.episode_idx % len(IT_load_files)
            IT_load_file = IT_load_files[IT_load_idx]
        else:
            IT_load_file = None
        
        # # Print the selected weather file and IT load file
        print(f'start_instance(): weather_files[{weather_idx}]={weather_file}, IT_load_files[{IT_load_idx}]={IT_load_file}')

        # Make copies of model file, weather file and IT load file into output dir, and use it for execution
        # This allow update of these files without affecting active simulation instances
        copy_model_file =  output_dir + '/' + os.path.basename(self.model_file)
        copy_weather_file = output_dir + '/' + os.path.basename(weather_file)
        copy_IT_load_file = output_dir + '/' + os.path.basename(IT_load_file) if IT_load_file is not None else None
        shutil.copy(self.model_file, output_dir)
        shutil.copy(weather_file, output_dir)
        if IT_load_file is not None:
            shutil.copy(IT_load_file, output_dir)
            shutil.move(copy_IT_load_file, output_dir + '/IT_load.csv')
        

        # Spawn a process
        cmd = self.energyplus_file \
              + ' -r ' \
              + ' -d ' + output_dir \
              + ' -w ' + copy_weather_file \
              + ' ' + copy_model_file
        print('Starting EnergyPlus with command: %s' % cmd)
        self.energyplus_process = subprocess.Popen(
            shlex.split(cmd),
            shell=False,
            cwd=output_dir
        )

    def stop_instance(self):
        if self.energyplus_process is not None:
            self.energyplus_process.terminate()
            self.energyplus_process = None
        if self.pipe_io is not None:
            self.pipe_io.stop()
        if self.episode_idx >= 0:
            def count_severe_errors(file):
                if not os.path.isfile(file):
                    return -1 # Error count is unknown
                # Sample: '   ************* EnergyPlus Completed Successfully-- 6214 Warning; 2 Severe Errors; Elapsed Time=00hr 00min  7.19sec'
                fd = open(file)
                lines = fd.readlines()
                fd.close()
                for line in lines:
                    if line.find('************* EnergyPlus Completed Successfully') >= 0:
                        tokens = line.split()
                        return int(tokens[6])
                return -1
            epsode_dir = self.get_output_dir()
            file_csv = epsode_dir + '/eplusout.csv'
            file_csv_gz = epsode_dir + '/eplusout.csv.gz'
            file_err = epsode_dir + '/eplusout.err'
            files_to_preserve = ['eplusout.csv', 'eplusout.err', 'eplustbl.htm']
            files_to_clean = ['eplusmtr.csv', 'eplusout.audit', 'eplusout.bnd',
                              'eplusout.dxf', 'eplusout.eio', 'eplusout.edd',
                              'eplusout.end', 'eplusout.eso', 'eplusout.mdd',
                              'eplusout.mtd', 'eplusout.mtr', 'eplusout.rdd',
                              'eplusout.rvaudit', 'eplusout.shd', 'eplusssz.csv',
                              'epluszsz.csv', 'sqlite.err']

            # Check for any severe error
            nerr = count_severe_errors(file_err)
            if nerr != 0:
                print('EnergyPlusEnv: Severe error(s) occurred. Error count: {}'.format(nerr))
                print('EnergyPlusEnv: Check contents of {}'.format(file_err))
                #sys.exit(1)

            # Compress csv file and remove unnecessary files
            # If csv file is not present in some reason, preserve all other files for inspection
            if os.path.isfile(file_csv):
                with open(file_csv, 'rb') as f_in:
                    with gzip.open(file_csv_gz, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(file_csv)

                if not os.path.exists("/tmp/verbose"):
                    for file in files_to_clean:
                        file_path = epsode_dir + '/' + file
                        if os.path.isfile(file_path):
                            os.remove(file_path)

    def step(self, action):
        self.timestep1 += 1
        # Send action to the environment
        if action is not None:
            # print(f"Action in 'EnergyPlusEnv.step()' = {action}, action is not None")
            # baselines 0.1.6 changed action type
            if isinstance(action, np.ndarray) and isinstance(action[0], np.ndarray):
                action = action[0]
            self.ep_model.set_action(
                normalized_action=action,
                framework=self.framework
            )

            if not self.send_action():
                print('EnergyPlusEnv.step(): Failed to send an action. Quitting.')
                observation = (self.observation_space.low + self.observation_space.high) * 0.5
                reward = 0.0
                done = True
                print('EnergyPlusEnv: (quit)')
                return observation, reward, done, {}
        
        # Receive observation from the environment    
        # Note that in our co-simulation environment, the state value of the last time step can not be retrived from EnergyPlus process
        # because EMS framework of EnergyPlus does not allow setting EMS calling point ater the last timestep is completed.
        # To remedy this, we assume set_raw_state() method of each model handle the case raw_state is None.
        raw_state, done = self.receive_observation() # raw_state will be None for for call at total_timestep + 1
        self.ep_model.set_raw_state(raw_state)
        observation = self.ep_model.get_state()
        reward = self.ep_model.compute_reward()

        if done:
            print('EnergyPlusEnv: (done)')
        return observation, reward, done, {}
    
    def send_action(self):
        # print(f"Action in 'EnergyPlusEnv.send_action()' = {self.ep_model.action}")
        num_data = len(self.ep_model.action)
        if self.pipe_io.writeline('{0:d}'.format(num_data)):
            return False
        for i in range(num_data):
            self.pipe_io.writeline('{0:f}'.format(self.ep_model.action[i]))
        self.pipe_io.flush()
        return True

    def receive_observation(self):
        line = self.pipe_io.readline()
        # Initialize done
        done = False
        if (line == ''):
            # This is the (usual) case when we send action data after all simulation timestep have finished.
            done = True
            return None, done
        num_data = int(line)
        # Number of data received may not be same as the size of observation_space
        # assert(num_data == len(self.observation_space.low))
        raw_state = np.zeros(num_data)
        for i in range(num_data):
            line = self.pipe_io.readline()
            if (line == ''):
                # This is usually system error
                done = True
                return None, done
            val = float(line)
            raw_state[i] = val
        return raw_state, done
    
    def render(self, mode='human'):
        if mode == 'human':
            return False
        
    def close(self):
        self.stop_instance()

    def plot(self, log_dir='', csv_file=''):
        self.ep_model.plot(log_dir=log_dir, csv_file=csv_file)

    def dump_timesteps(self, log_dir='', csv_file='', reward_file=''):
        self.ep_model.dump_timesteps(log_dir=log_dir, csv_file=csv_file)

    def dump_episodes(self, log_dir='', csv_file='', reward_file=''):
        self.ep_model.dump_episodes(log_dir=log_dir, csv_file=csv_file)

    def get_output_dir(self):
        return self.log_dir + '/output/episode-{:08}-{:05}'.format(self.episode_idx, os.getpid())
        
def parser():
    usage = 'Usage: python {} [--verbose] [--energyplus <file>] [--model <file>] [--weather <file>] [--simulate] [--plot] [--help]'.format(__file__)
    argparser = ArgumentParser(usage=usage)
    #argparser.add_argument('fname', type=str,
    #                       help='echo fname')
    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help='Show verbose message')
    argparser.add_argument('-e', '--energyplus', type=str,
                           dest='energyplus',
                           help='EnergyPlus executable file')
    argparser.add_argument('-m', '--model', type=str,
                           dest='model',
                           help='Model file')
    argparser.add_argument('-w', '--weather', type=str,
                           dest='weather',
                           help='Weather file')
    argparser.add_argument('-s', '--simulate',
                           action='store_true',
                           help='Do simulation')
    argparser.add_argument('-p', '--plot',
                           action='store_true',
                           help='Do plotting')
    return argparser.parse_args()

def easy_agent(next_state, target, hi, lo):
    sensitivity_pos = 1.0
    sensitivity_neg = 1.0
    act_west_prev = 0
    act_east_prev = 0
    alpha = 0.4

    delta_west = next_state[1] - target
    if delta_west >= 0:
        act_west = target - delta_west * sensitivity_pos
    else:
        act_west = target - delta_west * sensitivity_neg
    act_west = act_west * alpha + act_west_prev * (1 - alpha)
    act_west_prev = act_west
    
    delta_east = next_state[2] - target
    if delta_east >= 0:
        act_east = target - delta_east * sensitivity_pos
    else:
        act_east = target - delta_east * sensitivity_neg
    act_east = act_east * alpha + act_east_prev * (1 - alpha)
    act_east_prev = act_east

    act_west = max(lo, min(act_west, hi))
    act_east = max(lo, min(act_east, hi))
    action = np.array([act_west, act_west, act_west, act_west, act_east, act_east, act_east, act_east])
    return action

def original_tester():
    args = parser()
    print('args={}'.format(args))
    args.simulate = True
    lo = 0.0
    hi = 40.0
    target = 23.0
    
    # obs[0]: Eronment:Site Outdoor Air Drybulb Temperature [C](TimeStep)
    # obs[1]: Workload level (not implemented yet)
    #obs_space = spaces.Box(np.array([-20.0, 0.0]),
    #                       np.array([ 50.0, 1.0]))

    # act[0]: WestZoneDECOutletNode_setpoint
    # act[1]: WestZoneIECOutletNode_setpoint
    # act[2]: WestZoneCCoilAirOutletNode_setpoint
    # act[3]: WestAirLoopOutletNode_setpoint
    # act[4]: EastZoneDECOutletNode_setpoint
    # act[5]: EastZoneIECOutletNode_setpoint
    # act[6]: EastZoneCCoilAirOutletNode_setpoint
    # act[7]: EastAirLoopOutletNode_setpoint
    #act_space = spaces.Box(np.array([ lo, lo, lo, lo, lo, lo, lo, lo]),
    #                       np.array([ hi, hi, hi, hi, hi, hi, hi, hi]))
        
    # just for testing
    env = EnergyPlusEnv(verbose = args.verbose)
    if env is None:
        quit()

    if (args.simulate):
        for ep in range(1):
            PUE_min = 100.
            PUE_max = 0.
            PUE_sum = 0.
            PUE_count = 0
            next_state = env.reset()

            for i in range(1000000):
                #if args.verbose:
                #    os.system('clear')
                #    print('Step {}'.format(i))
                    
                #action = env.action_space.sample()
                action = easy_agent(next_state, target, hi, lo)
                PUE = next_state[3]
                PUE_sum += PUE
                PUE_min = min(PUE, PUE_min)
                PUE_max = max(PUE, PUE_max)
                PUE_count += 1

                next_state, reward, done, _ = env.step(action)
                PUE_ave = PUE_sum / PUE_count

                if args.verbose:
                    print('========= count={} PUE={} PUEave={} PUEmin={} PUEmax={}'.format(PUE_count, PUE, PUE_ave, PUE_min, PUE_max))
                if done:
                    break
            PUE_ave = PUE_sum / PUE_count
            print('============================= Episodo done. count={} PUEave={} PUEmin={} PUEmax={}'.format(PUE_count, PUE_ave, PUE_min, PUE_max))
            #env.close()

    env.plot()

def test_env():
    args = parser()
    print('args={}'.format(args))
    args.simulate = True
    env = EnergyPlusEnv(verbose=args.verbose)
    action_chiller_temperature_setpoint = 18
    action_chiller_pump_mass_flow = 2
    action_airloop_fan_mass_flow = 4
    action = np.array([action_chiller_temperature_setpoint, action_chiller_pump_mass_flow, action_airloop_fan_mass_flow])
    next_state = env.reset()

    count = 0
    for i in range(1000):
        next_state, reward, done, _ = env.step(action)
        count += 1
        print("======= count={},reward={}".format(count,reward))


if __name__ == '__main__':
    test_env()