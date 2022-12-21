import gym
import torch.nn as nn
import torch

import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.tune import register_env

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from ray.rllib.agents.ppo import ppo, PPOTrainer


class NasutaTorchActionMaskModel(TorchModelV2, nn.Module):
    """
       Model that handles simple discrete action masking.
       This assumes the outputs are logits for a single Categorical action dist.
       PyTorch version of ActionMaskModel, derived from:
       https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
       https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
       """

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, gym.spaces.dict.Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        # self.register_variables(self.internal_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model(
            {"obs": input_dict["obs"]["observations"]}
        )

        # Convert action_mask into a [0.0 || -inf] mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


def env_creator(env_config):
    return DisjunctiveGraphJspEnv(**env_config)  # return an env instance


register_env("graph-jsp-env", env_creator)
ModelCatalog.register_custom_model("nasuta-torch", NasutaTorchActionMaskModel)


def run_mask_ppo_trainer(jsp_instance: np.ndarray) -> None:
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["env"] = "graph-jsp-env"
    config["env_config"] = {
        "jps_instance": jsp_instance,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
        "action_mode": 'task',
        "env_transform": 'mask',
        "dtype": "float64",
        "verbose": 0,
        "default_visualisations": [
            "gantt_console",
            # "graph_window",  # very expensive
        ]
    }
    config["framework"] = "torch"
    config["num_workers"] = 2
    config["model"] = {
        "custom_model": "nasuta-torch",
        "custom_model_config": {}
    }
    _, n_jobs, n_machines = jsp_instance.shape
    config["horizon"] = n_jobs * n_machines
    config["log_level"] = 'INFO'

    trainer = PPOTrainer(config=config)

    for i in range(1):
        train_data = trainer.train()
        #print(train_data)


if __name__ == '__main__':
    jsp = np.array([
        [
            [0, 1, 2, 3],  # job 0 (engineer’s hammer)
            [0, 2, 1, 3]  # job 1  (Nine Man Morris)
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])
    run_mask_ppo_trainer(jsp_instance=jsp)
