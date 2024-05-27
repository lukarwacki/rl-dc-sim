__all__ = ['Monitor', 'get_monitor_files', 'load_results', 'LoggingCallback', 'WriteSettingsCallback', 'TensorBoardCallback']

import time
import os
import os.path as osp
import csv
import json
import uuid
from glob import glob

import pandas
import numpy as np

import gym
from gym.core import Wrapper
from stable_baselines3.common.callbacks import BaseCallback

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        print('Monitor: filename={}'.format(filename))
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n'%json.dumps({"t_start": self.tstart, 'env_id' : env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+reset_keywords)
            self.logger.writeheader()
            self.f.flush()

        self.reset_keywords = reset_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        # time.sleep(5)
        # print("")
        # print(f"Action in 'Monitor.step() = {action}")
        ob, rew, done, info = self.env.step(action)
        if not done:
            self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            
            # Write rewards into the output dir
            output_dir = self.env.get_output_dir()
            if output_dir:
                rewards_file = os.path.join(output_dir, "rewards.csv")
                with open(rewards_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(['Reward'])
                    for rew in self.rewards:
                        writer.writerow([rew])

            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)

    def close(self):


        if self.f is not None:
            self.f.close()
        if self.env:
            return self.env.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times
    
    def unnormalize_rew(self, norm_rew):
        """
        Unnormalize normalized reward(s) to original scale.

        Args:
            norm_rew (numpy.ndarray): Normalized reward(s) to unnormalize.

        Returns:
            numpy.ndarray: Unnormalized reward(s).
        """
        return (1 - norm_rew) * self.env.ep_model.rew_high

class LoadMonitorResultsError(Exception):
    pass

def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) + 
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df

class LoggingCallback(BaseCallback):
    """
    Callback to log reward, action, and observation at each timestep during training.
    """

    def __init__(self, folder_path, args, mapping, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.folder_path = folder_path
        self.rewards = []
        self.actions = []
        self.observations = []
        self.header_written = False
        self.args = args
        self.mapper = mapping

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each environment step during training.

        Returns:
        -------
        bool:
            Whether or not training should continue.
        """
        # Log reward, action, and observation at each timestep
        reward = self.locals["rewards"][0]      # Extract reward from array
        action = self.locals["actions"][0][0]   # Extract action from array
        observation = self.locals["new_obs"][0] # Extract array of observations from array of arrays

        self.rewards.append(reward)
        self.actions.append(action)
        self.observations.append(observation)
        # print("Rewards: ",self.rewards)
        # print("Actions: ", self.actions)
        # print("Obs: ", self.observations)
        # time.sleep(10.0)


        # Check if the environment has been reset
        if self.locals.get("done", False):
            # Convert observations list of arrays into list of lists
            self.observations = [arr.tolist() for arr in self.observations]
            
            # Save rewards, actions, and observations to a file
            data = np.column_stack((self.rewards, self.actions, self.observations))
            
            # If the header has not been written yet, write it
            if not self.header_written:
                # Create header
                obs_names = self.mapper.state[:-1]
                header_names = ['rewards', 'actions']
                header_names.extend(obs_names)
                header = ",".join(header_names)

                file_path = os.path.join(self.folder_path, 'training_data.csv')
                np.savetxt(file_path, data, delimiter=',', header=header, comments='')
                self.header_written = True
        
            else:
                # Append data to the file
                file_path = os.path.join(self.folder_path, 'training_data.csv')
                with open(file_path, 'ab') as f:
                    np.savetxt(f, data, delimiter=',')
            
            # Clear arrays
            self.rewards = []
            self.actions = []
            self.observations = []

        return True  # Return True to continue training

class WriteSettingsCallback(BaseCallback):
    def __init__(self, env, args, verbose=0):
        super(WriteSettingsCallback,self).__init__(verbose)
        self.env = env
        self.folder_path = env.log_dir
        self.args = args
        self.action_limits = env.ep_model.action_space_limits
    
    def _on_training_start(self) -> None:
        """
        This method writes the argparser argumetns to a file on training start
        """
        file_path = os.path.join(self.folder_path, 'settings.txt')
        reward_check_path = os.path.join(self.folder_path, 'reward_check.txt')
        with open(file_path, 'w') as f:
            # Write arguments specified by args
            for arg in vars(self.args):
                f.write(f"{arg}: {getattr(self.args, arg)}\n")
            
            # Write shapes of spaces
            f.write("\n")
            f.write(f"Normalized action Space: {self.training_env.action_space}\n")
            f.write(f"Real action space: {self.action_limits}\n")
            f.write(f"Observation Space: {self.training_env.observation_space}\n")

            # # Write information on reward function
            f.write("\n")
            f.write("Reward double check:\n")
            f.write("\n")
            f.write(f"Power reward type: {self.env.ep_model.RewardCalc.reward_P_type}\n")
            f.write(f"Temperature reward type: {self.env.ep_model.RewardCalc.reward_T_type}\n")
            f.write(f"Temperature penalty type: {self.env.ep_model.RewardCalc.penalty_T_type}\n")
            f.write(f"Fluctuation penalty type: {self.env.ep_model.RewardCalc.penalty_a_fluctuation_type}\n")
            
            f.write(f"Temperature reward weight: {self.env.ep_model.RewardCalc.lambda_T_r}\n")
            f.write(f"Temperature penalty weight: {self.env.ep_model.RewardCalc.lambda_T_p}\n")
            f.write(f"Fluctuation penalty weight: {self.env.ep_model.RewardCalc.lambda_a_p}\n")
            
            f.write(f"T_constr: {self.env.ep_model.RewardCalc.T_constr}\n")
            f.write(f"T bandwith: {self.env.ep_model.RewardCalc.T_SP_bandwidth}\n")
            f.write(f"Beta: {self.env.ep_model.RewardCalc.beta}\n")
            f.write(f"Softplus shift: {self.env.ep_model.RewardCalc.softplus_shift}\n")
            f.write(f"Action bandwith: {self.env.ep_model.RewardCalc.action_bandwidth}\n")
        
    def _on_step(self) -> bool:
        # Do nothing
        return True

class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.upper_action_fraction = 0.0
        self.lower_action_fraction = 0.0
    
    def _on_step(self):
        return True

    def _on_rollout_end(self) -> None:
        actions = self.model.rollout_buffer.actions  # Get all actions in the rollout buffer
        upper_action_count = np.sum(actions >= 1, axis=0) # Count occurrences of lower limit action (-1)
        lower_action_count = np.sum(actions <= -1, axis=0)  # Count occurrences of lower limit action (-1)
        total_actions = len(actions)
        
        if total_actions > 0:
            upper_action_fractions = upper_action_count / total_actions
            lower_action_fractions = lower_action_count / total_actions
            self.upper_action_fraction = np.mean(upper_action_fractions)
            self.lower_action_fraction = np.mean(lower_action_fractions)
        else:
            self.upper_action_fraction = 0.0
            self.lower_action_fraction = 0.0

        self.logger.record("train/action_fraction_upper", self.upper_action_fraction)
        self.logger.record("train/action_fraction_lower", self.lower_action_fraction)

def test_monitor():
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _obs, _reward, done, _info = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)

if __name__ == '__main__':
    test_monitor()