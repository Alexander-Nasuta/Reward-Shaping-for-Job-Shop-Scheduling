from typing import Dict

import torch as th
import wandb

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

graph_mask_ppo_default_config = {
    "policy_type": MaskableActorCriticPolicy,

    "model_hyper_parameters": {
        "gamma": 0.99,  # discount factor,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "normalize_advantage": True,
        "policy_kwargs": {
            "net_arch": [{
                "pi": [64, 64],
                "vf": [64, 64],
            }],
            "ortho_init": True,
            "activation_fn": th.nn.Tanh,  # th.nn.ReLU
            "optimizer_kwargs": {  # for th.optim.Adam
                "eps": 1e-5
            }
        }
    },
}


def merge_configs_graph_jss_env(sweep_params: wandb.sdk.wandb_config.Config,
                                default_config: Dict = None) -> Dict:
    if default_config is None:
        default_config = graph_mask_ppo_default_config
    run_config = graph_mask_ppo_default_config

    run_config["benchmark_instance"] = sweep_params["benchmark_instance"]
    run_config["optimal_makespan"] = sweep_params["optimal_makespan"]

    # override default run config
    model_params = [
        "learning_rate",
        "n_steps",
        "n_epochs",
        "gamma",
        "batch_size",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        # "tensorboard_log",
        "create_eval_env",
        # "verbose",
        "seed",
        "device"
    ]
    for m_param in model_params:
        run_config["model_hyper_parameters"][m_param] = sweep_params[m_param]

    env_params = [
        "normalize_observation_space",
        "flat_observation_space",
        "perform_left_shift_if_possible",
        "dtype",
        "action_mode",
        "reward_function",
    ]

    policy_params = [
        "ortho_init",
        "normalize_images",
    ]
    for p_param in policy_params:
        run_config["model_hyper_parameters"]["policy_kwargs"][p_param] = sweep_params[p_param]

    net_arch = [{
        "pi": [sweep_params["net_arch_n_size"]] * sweep_params["net_arch_n_layers"],
        "vf": [sweep_params["net_arch_n_size"]] * sweep_params["net_arch_n_layers"],
    }]

    activation_fn = None
    if sweep_params["activation_fn"] == 'ReLu':
        activation_fn = th.nn.ReLU
    elif sweep_params["activation_fn"] == 'Tanh':
        activation_fn = th.nn.Tanh
    elif sweep_params["activation_fn"] == 'Hardtanh':
        activation_fn = th.nn.Hardtanh
    elif sweep_params["activation_fn"] == 'ELU':
        activation_fn = th.nn.ELU
    elif sweep_params["activation_fn"] == 'RRELU':
        activation_fn = th.nn.PReLU
    else:
        raise NotImplementedError(f"activation function '{activation_fn}' is not available/implemented. "
                                  f"You may need to add a case for your activation function")

    run_config["model_hyper_parameters"]["policy_kwargs"]["net_arch"] = net_arch
    run_config["model_hyper_parameters"]["policy_kwargs"]["activation_fn"] = activation_fn
    run_config["model_hyper_parameters"]["policy_kwargs"]["optimizer_kwargs"]["eps"] = sweep_params["optimizer_eps"]

    return run_config
