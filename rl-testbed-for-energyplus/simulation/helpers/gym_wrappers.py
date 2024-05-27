import gym
import numpy as np

class NormalizeObservationMinMax(gym.core.Wrapper):
    """
    A Gym environment wrapper to normalize observations between 0 and 1
    based on known minimum and maximum values.

    Args:
        env (gym.Env): The environment to wrap.
        low (numpy.ndarray): Array representing the minimum observation values.
        high (numpy.ndarray): Array representing the maximum observation values.
    """
    def __init__(
        self,
        env, 
        low, 
        high,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        
        assert env.observation_space.shape == (len(low),), \
            f"Observation space shape {env.observation_space.shape} does not match the shape of the provided lower limit {(len(low),)}."
        assert env.observation_space.shape == (len(high),), \
            f"Observation space shape {env.observation_space.shape} does not match the shape of the provided upper limit {(len(high),)}."
        
        self.low = np.array(low)
        self.high = np.array(high)

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        if not return_info:
            return obs
        else:
            return obs, info

    def normalize(self, obs):
        """
        Normalize observation(s) between 0 and 1.

        Args:
            obs (numpy.ndarray): Observation(s) to normalize.

        Returns:
            numpy.ndarray: Normalized observation(s).
        """
        return (obs - self.low) / (self.high - self.low)

class NormalizeRewardMinMax(gym.core.Wrapper):
    def __init__(
        self,
        env,
        low,
        high,
        gamma=0.99
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.gamma = gamma
        self.returns = np.zeros(self.num_envs)
        
        self.low = low
        self.high = high

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, dones, infos


    def normalize(self, rew):
        """
        Normalize reward(s) between 0 and 1.

        Args:
            rew (numpy.ndarray): Reward(s) to normalize.

        Returns:
            numpy.ndarray: Normalized reward(s).
        """
        return (rew - self.low) / (self.high - self.low)
    
    


def test():
    low = np.array([2,6,10])
    high = np.array([10,20,30])
    obs = np.array([2,20,20])

    normalizer = NormalizeObservationMinMax(low,high)
    print(normalizer.normalize(obs))

if __name__=='__main__':
    test()