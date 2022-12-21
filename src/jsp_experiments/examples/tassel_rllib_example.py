import time

import gym
import ray
import torch.nn as nn
import torch
from gym.spaces import Dict

from ray import tune
from ray.rllib.agents.ppo import ppo, PPOTrainer
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

from jsp_experiments.instance_loader import get_instance_std_path_by_name

tf1, tf, tfv = try_import_tf()


class TasselJssTFModel(TFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        print(orig_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "real_obs" in orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["real_obs"]})
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.action_embed_model.value_function()


class TasselJssTorchModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

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
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "real_obs" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["real_obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["real_obs"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


ModelCatalog.register_custom_model("tassel_torch", TasselJssTorchModel)
ModelCatalog.register_custom_model("tassel_tf", TasselJssTFModel)


def run_mask_ppo_with_tune():
    jsp_std_path = str(get_instance_std_path_by_name("ft06"))

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "JSSEnv:jss-v1"
    config["env_config"] = {
        'env_config': {'instance_path': jsp_std_path}
    }
    config["model"] = {
        "custom_model": "tassel_torch",
        "custom_model_config": {},
    }

    config["framework"] = "torch"
    config["num_workers"] = 1
    config["log_level"] = 'INFO'

    stop = {
        #"time_total_s": 10 * 60,
        "time_total_s": 10,
    }

    ray.init()
    start_time = time.time()

    trainer = PPOTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        print(result)
        # wandb.config.update(config_update, allow_val_change=True)
    # trainer.export_policy_model("/home/jupyter/JSS/JSS/models/")

    ray.shutdown()


if __name__ == '__main__':
    # i_path = str(get_instance_std_path_by_name("ft06"))
    # e_config = {'instance_path': i_path}
    # e = gym.make('JSSEnv:jss-v1', env_config=e_config)
    # ray.rllib.utils.check_env(e)

    # https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py#L69
    run_mask_ppo_with_tune()
