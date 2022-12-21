import gym
import numpy as np

class TasselEnvSB3Wrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env
    ):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=np.float64)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return observation['real_obs'], reward, done, info

    def valid_action_mask(self, env: gym.Env):
        # unpack wrappers
        # return env.env.env.env.legal_actions # alternative
        return self.env.env.env.legal_actions

    def reset(self, **kwargs):
        return self.env.reset()['real_obs']
