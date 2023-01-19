import sys

import graph_jsp_env.disjunctive_graph_jsp_visualizer
import gym
import pprint
import sb3_contrib

import numpy as np
import wandb as wb

from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from graph_jsp_env.disjunctive_graph_logger import log
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from jsp_experiments.graph_jsp.graph_jsp_callback import GraphJspLoggerCallback
from jsp_experiments.instance_loader import get_instance_by_name_as_numpy_array
from jsp_experiments.merge_default_and_sweep_config import merge_configs_graph_jss_env

gym.envs.register(
    id='gjsp-v0',
    entry_point='graph_jsp_env.disjunctive_graph_jsp_env:DisjunctiveGraphJspEnv',
    kwargs={},
)

mask_ppo_sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'optimality_gap',
        'goal': 'minimize'
    },
    'parameters': {
        # Constanst
        "total_timesteps": {
            'values': [150_000]
        },
        "n_envs": {
            'values': [8]
        },
        "benchmark_instance": {
            'values': ["ft06"]
        },
        "optimal_makespan": {
            'value': 55.0
        },
        "n_machines": {
            "values": [6]
        },
        "n_jobs": {
            "values": [6]
        },

        # gamma: float = 0.99,
        # Discount factor
        "gamma": {
            "distribution": "uniform",
            "min": 0.95,
            "max": 1,
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
            "values": ["cuda"]  # cpu, mps, auto, cuda
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
            'values': [2, 3, 4]
        },
        "net_arch_n_size": {
            "distribution": "q_uniform",
            "min": 20,
            "max": 100,
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
            'values': [False]
        },
        "dtype": {
            'values': ["float32"]
        },

        "reward_function": {
            'value': 'graph-tassel'
        },
        "hyperparameter_tuning": {
            'value': True
        },

        # eval params
        "n_eval_episodes": {
            'value': 5
        }

    }
}


def perform_run() -> None:
    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            # dir=f"{PATHS.WAND_OUTPUT_PATH}/"
    ) as run:
        log.info(f"run name: {run.name}, run id: {run.id}")

        sweep_params = wb.config
        log.info(f"hyper params: {pprint.pformat(sweep_params)}")

        run_config = merge_configs_graph_jss_env(
            sweep_params=sweep_params
        )

        log.info(f"run config: {pprint.pformat(run_config)}")

        wb_cb = WandbCallback(
            gradient_save_freq=100,
            # model_save_path=PATHS.WAND_OUTPUT_PATH.joinpath(f"{run.name}_{run.id}"),
            verbose=1,
        )

        logger_cb = GraphJspLoggerCallback(
            optimal_makespan=run_config["optimal_makespan"],
            wandb_ref=wb
        )

        callbacks = [logger_cb, wb_cb]

        def mask_fn(env):
            return env.valid_action_mask()

        jsp_instance = get_instance_by_name_as_numpy_array(name=run_config['benchmark_instance'])

        env_kwargs = {
            "jps_instance": jsp_instance,
            'reward_function': sweep_params['reward_function'],
            "normalize_observation_space": sweep_params['normalize_observation_space'],
            "flat_observation_space": sweep_params['flat_observation_space'],
            "perform_left_shift_if_possible": sweep_params['perform_left_shift_if_possible'],
            "reward_function_parameters": {},
            "default_visualisations": [
                "gantt_window",
            ]
        }

        venv = make_vec_env(
            env_id='gjsp-v0',
            env_kwargs=env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=sweep_params["n_envs"]
        )

        model = sb3_contrib.MaskablePPO(
            run_config["policy_type"],
            env=venv,
            verbose=1,
            # tensorboard_log=PATHS.WAND_OUTPUT_PATH.joinpath(f"{run.name}_{run.id}"),
            **run_config["model_hyper_parameters"],
        )

        log.info(f"training the agent")
        model.learn(total_timesteps=sweep_params["total_timesteps"], callback=callbacks)

        log.info("evaluating model performance")
        n_eval_episodes = sweep_params["n_eval_episodes"]
        makespans = []

        for _ in track(range(n_eval_episodes), description="evaluating model performance ..."):
            done = False
            obs = venv.reset()
            while not done:
                masks = np.array([env.action_masks() for env in model.env.envs])
                action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
                obs, rewards, dones, info = venv.step(action)
                done = np.all(dones == True)
                if done:
                    for sub_env_info in info:
                        makespans.append(sub_env_info["makespan"])

        from statistics import mean
        mean_return = mean(makespans)

        log.info(f"mean evaluation makespan: {mean_return:.2f}")
        wb.log({"mean_makespan": mean_return})

        obs = venv.reset()
        venv.close()
        del venv


if __name__ == '__main__':
    # sweep_id = wb.sweep(mask_ppo_sweep_config, project="reward-functions-comparison")
    sweep_id = 'a5np8wbg'
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project="reward-functions-comparison"
    )
