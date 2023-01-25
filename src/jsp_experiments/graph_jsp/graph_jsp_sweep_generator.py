import pprint
import os
import sys

import wandb as wb
from pathlib import Path

from graph_jsp_env.disjunctive_graph_logger import log
from typing import Dict, List

from jsp_experiments.instance_loader import get_instance_by_name_as_numpy_array


def generate_sweep_config_dict(
        *,
        benchmark_instance: str,
        n_machines: int,
        n_jobs: int,
        optimal_makespan: float,

        method: str = 'random',

        total_timesteps: int,

        reward_function: str,
        reward_function_additional_sweep_params=None,

        left_shift: bool,

        net_arch_n_layers: List[int],
        net_arch_n_size_min: int,
        net_arch_n_size_max: int
) -> Dict:
    if reward_function_additional_sweep_params is None:
        reward_function_additional_sweep_params = {}
    mask_ppo_sweep_config = {
        'method': method,
        'metric': {
            'name': 'optimality_gap',
            'goal': 'minimize'
        },
        'parameters': {
            # Constanst
            "total_timesteps": {
                'values': [total_timesteps]
            },
            "n_envs": {
                'values': [8]
            },
            "benchmark_instance": {
                'values': [benchmark_instance]
            },
            "optimal_makespan": {
                'value': optimal_makespan
            },
            "n_machines": {
                "values": [n_machines]
            },
            "n_jobs": {
                "values": [n_jobs]
            },

            # gamma: float = 0.99,
            # Discount factor
            "gamma": {
                "values": [1.0]
            },
            # gae_lambda: float = 0.95,
            # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            "gae_lambda": {
                "distribution": "uniform",
                "min": 0.8,
                "max": 1,
            },
            # max_grad_norm: float = 0.5,
            # The maximum value for the gradient clipping
            "max_grad_norm": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.7,
            },

            # learning_rate: Union[float, Schedule] = 3e-4,
            # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
            "learning_rate": {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3,
            },

            # batch_size: Optional[int] = 64,
            # Minibatch size
            "batch_size": {
                'distribution': 'q_log_uniform_values',
                'min': 16,
                'max': 512,
                "q": 16
            },
            # clip_range: Union[float, Schedule] = 0.2,
            # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
            "clip_range": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.7,
            },

            # clip_range_vf: Union[None, float, Schedule] = None,
            #
            # Clipping parameter for the value function,
            # it can be a function of the current progress remaining (from 1 to 0).
            # This is a parameter specific to the OpenAI implementation.
            # If None is passed (default), no clipping will be done on the value function.
            #
            # IMPORTANT: this clipping depends on the reward scaling.
            #
            "clip_range_vf": {
                'values': [
                    None
                ]
            },

            # vf_coef: float = 0.5,
            # Value function coefficient for the loss calculation
            "vf_coef": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.7,
            },

            # ent_coef: float = 0.0,
            # Entropy coefficient for the loss calculation
            "ent_coef": {
                "distribution": "q_uniform",
                "min": 0.0,
                "max": 0.0002458,
                "q": 0.00002
            },

            # normalize_advantage: bool = True
            # Whether to normalize or not the advantage
            "normalize_advantage": {
                'values': [True, False]
            },
            # n_epochs: int = 10,
            # Number of epoch when optimizing the surrogate loss
            "n_epochs": {
                "distribution": "q_uniform",
                "min": 4,
                "max": 20,
                "q": 1
            },

            # n_steps: int = 2048,
            # The number of steps to run for each environment per update
            # (i.e. batch size is n_steps * n_env where n_env is number of environment
            # copies running in parallel)
            "n_steps": {
                'values': [
                    512,
                    1024,
                    2048,
                ]
            },
            # device: Union[th.device, str] = "auto",
            #  Device (cpu, cuda, …) on which the code should be run. Setting it to auto,
            #  the code will be run on the GPU if possible.
            "device": {
                "values": ["auto"]  # cpu, mps, auto, cuda
            },
            # seed: Optional[int] = None,
            # Seed for the pseudo random generators
            "seed": {
                "values": [1337]
            },

            # verbose: int = 0,
            # the verbosity level: 0 no output, 1 info, 2 debug
            # "verbose": {
            #     "values": [0]
            # },

            # create_eval_env: bool = False,
            # Whether to create a second environment that will be used for evaluating the agent periodically.
            # (Only available when passing string for the environment)
            "create_eval_env": {
                "values": [False]
            },
            # tensorboard_log: Optional[str] = None,
            # the log location for tensorboard (if None, no logging)
            # "tensorboard_log": {
            #    "values": [None]
            # },

            # target_kl: Optional[float] = None,
            # Limit the KL divergence between updates, because the clipping
            # is not enough to prevent large update
            # see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            # By default, there is no limit on the kl div.
            "target_kl": {
                "values": [None]
            },

            # policy params

            # net_arch (Optional[List[Union[int, Dict[str, List[int]]]]]) –
            # The specification of the policy and value networks.

            # 'net_arch_n_layers' and 'net_arch_n_size' will result in a dict that will be passed to 'net_arch'
            # see code below
            "net_arch_n_layers": {
                'values': net_arch_n_layers
            },
            "net_arch_n_size": {
                "distribution": "q_uniform",
                "min": net_arch_n_size_min,
                "max": net_arch_n_size_max,
                "q": 10
            },

            # ortho_init: bool = True,
            # Whether to use or not orthogonal initialization
            "ortho_init": {
                'values': [True]
            },
            # normalize_images: bool = True,
            "normalize_images": {
                'values': [True]
            },
            # activation_fn: Type[nn.Module] = nn.Tanh
            # Activation function
            # https://pytorch.org/docs/stable/nn.html
            "activation_fn": {
                "values": [
                    "Tanh",  # th.nn.Tanh,
                    "ReLu",  # th.nn.ReLU
                    "Hardtanh",
                    "ELU",
                    "RRELU",
                ]
            },

            "optimizer_eps": {  # for th.optim.Adam
                "values": [1e-6, 1e-7, 1e-8]
            },

            # env params
            "action_mode": {
                'values': ['task']
            },
            "normalize_observation_space": {
                'values': [True]
            },
            "flat_observation_space": {
                'values': [True]
            },
            "perform_left_shift_if_possible": {
                'values': [left_shift]
            },
            "dtype": {
                'values': ["float32"]
            },

            "reward_function": {
                'value': reward_function
            },

            **reward_function_additional_sweep_params,
            "hyperparameter_tuning": {
                'value': True
            },

            # eval params
            "n_eval_episodes": {
                'value': 5
            }

        }
    }
    return mask_ppo_sweep_config


def generate_scripts(filename: str, target_dir, sweep_id: str) -> None:
    python_script = f"""
import wandb as wb

from jsp_experiments.graph_jsp.perform_sweep_run import perform_run

if __name__ == '__main__':
    sweep_id = '{sweep_id}'
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project="reward-functions-comparison"
    )
"""
    with open(f"{target_dir}/{filename}.py", "w+") as f:
        f.writelines(python_script)

    bash_script = f"""#!/bin/bash
for i in {{1..50}}; do python {filename}.py; done"""

    with open(f"{target_dir}/{filename}_wrapper.sh", "w+") as f:
        f.writelines(bash_script)


def generate_server_script(target_dir, benchmark_instance: str, n_jobs: int, n_machines: int) -> None:
    bash_script = f""""""""

    for screen_session, script_wrapper_name in zip(
            [
                "nasuta_ls",
                "nasuta_no_ls",

                "zhang_ls",
                "zhang_no_ls",

                "sam_ls",
                "sam_no_ls",

                "tassel_ls",
                "tassel_no_ls",

            ],
            [
                f"{benchmark_instance}_nasuta_ls_graph_jsp_env_tuning_wrapper.sh",
                f"{benchmark_instance}_nasuta_no_ls_graph_jsp_env_tuning_wrapper.sh",

                f"{benchmark_instance}_zhang_ls_graph_jsp_env_tuning_wrapper.sh",
                f"{benchmark_instance}_zhang_no_ls_graph_jsp_env_tuning_wrapper.sh",

                f"{benchmark_instance}_samsonov_ls_graph_jsp_env_tuning_wrapper.sh",
                f"{benchmark_instance}_samsonov_no_ls_graph_jsp_env_tuning_wrapper.sh",

                f"{benchmark_instance}_graph-tassel_ls_graph_jsp_env_tuning_wrapper.sh",
                f"{benchmark_instance}_graph-tassel_no_ls_graph_jsp_env_tuning_wrapper.sh",

            ]
    ):
        cmds = f'''# {screen_session}, {script_wrapper_name}
screen -r {screen_session} -X stuff "cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/{n_jobs}x{n_machines}^M"
screen -r {screen_session} -X stuff "bash /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/{n_jobs}x{n_machines}/{script_wrapper_name}^M"
'''
        bash_script += cmds

    with open(f"{target_dir}/run_sweeps_with_screen_sessions.sh", "w+") as f:
        f.writelines(bash_script)


def generate_sweep_code(
        *,
        total_timesteps,
        n_layers: List[int],
        n_layer_size_min: int,
        n_layer_size_max: int,

        benchmark_instance: str,
        optimal_makespan: float
) -> None:
    jsp_instance = get_instance_by_name_as_numpy_array(name=benchmark_instance)

    _, n_jobs, n_machines = jsp_instance.shape

    log.info(f"generating python code for sweep with jsp instance '{benchmark_instance}' (i)")

    graph_jsp_dir = Path(os.path.abspath(__file__)).parent
    target_dir = graph_jsp_dir.joinpath(f"{n_jobs}x{n_machines}")

    log.info(f"creating dircetory '{n_jobs}x{n_machines}'")
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"adding '__init__.py' '{n_jobs}x{n_machines}'")
    init_file = target_dir.joinpath("__init__.py")
    with open(init_file, "w+") as f:
        f.writelines(f"""""")

    for rf, rf_params in zip(
            ["nasuta", "zhang", "graph-tassel", "samsonov"],
            [{}, {}, {}, {
                "samsonov-gamma": {
                    "distribution": "uniform",
                    "min": 1.0,
                    "max": 1.1,
                }
            }]
    ):
        for ls in [True, False]:
            conf = generate_sweep_config_dict(
                benchmark_instance=benchmark_instance,
                n_machines=n_machines,
                n_jobs=n_jobs,
                optimal_makespan=optimal_makespan,

                method='random',
                total_timesteps=total_timesteps,

                left_shift=ls,

                reward_function=rf,
                reward_function_additional_sweep_params=rf_params,

                net_arch_n_layers=n_layers,
                net_arch_n_size_min=n_layer_size_min,
                net_arch_n_size_max=n_layer_size_max
            )

            sweep_id = wb.sweep(
                conf,
                entity="querry",
                project="reward-functions-comparison",
            )

            script_name = f"{benchmark_instance}_{rf}_{'ls' if ls else 'no_ls'}_graph_jsp_env_tuning"
            log.info(f"generating '{script_name}'")
            generate_scripts(
                filename=script_name,
                target_dir=target_dir,
                sweep_id=sweep_id,
            )

    log.info(f"generating server script")
    generate_server_script(
        target_dir=target_dir,
        n_jobs=n_jobs,
        n_machines=n_machines,
        benchmark_instance=benchmark_instance
    )
    log.info(f"done")


if __name__ == '__main__':
    n_layers = [3, 4, 5, 6]
    n_layer_size_min = 20
    n_layer_size_max = 200

    benchmark_instance: str = "orb04"
    optimal_makespan: float = 1005.0

    generate_sweep_code(
        total_timesteps=150_000,
        n_layers=n_layers,
        n_layer_size_min=n_layer_size_min,
        n_layer_size_max=n_layer_size_max,
        benchmark_instance=benchmark_instance,
        optimal_makespan=optimal_makespan
    )
