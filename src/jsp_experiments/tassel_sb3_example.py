import gym
import sb3_contrib

import numpy as np
import JSSEnv  # an ongoing issue with OpenAi's gym causes it to not import automatically
# external modules, see: https://github.com/openai/gym/issues/2809
# for older version of gym, you have to use
# env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from jsp_experiments.instance_loader import get_instance_std_path_by_name


class JssEnvSB3Wrapper(gym.Wrapper):

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


if __name__ == '__main__':
    # the https://github.com/prosysscience/JSSEnv REAMDE.md says the jsp files must follow the 'Taillard specification'
    # but they use the 'Standard specification' (http://jobshop.jjvh.nl/explanation.php#taillard_def) (date: 16.12.22)

    jsp_std_path = str(get_instance_std_path_by_name("ft06"))

    def wrapper_function(env):
        env = JssEnvSB3Wrapper(env)
        env = ActionMasker(env, action_mask_fn=env.valid_action_mask)
        return env


    venv = make_vec_env(
        env_id='jss-v1',
        env_kwargs={
            'env_config': {'instance_path': jsp_std_path}
        },
        wrapper_class=wrapper_function,
        n_envs=2)

    venv.reset()
    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
    )

    model.learn(total_timesteps=10_000)
