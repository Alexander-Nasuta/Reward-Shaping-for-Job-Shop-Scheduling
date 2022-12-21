import pprint

import gym
import sb3_contrib

import wandb as wb
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from jsp_experiments.instance_loader import get_instance_std_path_by_name

from graph_jsp_env.disjunctive_graph_logger import log

from jsp_experiments.merge_default_and_sweep_config import merge_configs_graph_jss_env
from jsp_experiments.tassel_jsp_env.tassel_jsp_callback import TasselLoggerCallback
from jsp_experiments.tassel_jsp_env.tassel_jsp_env_wrapper import TasselEnvSB3Wrapper

gym.envs.register(
    id='gjsp-v0',
    entry_point='graph_jsp.disjunctive_graph_jsp_env:DisjunctiveGraphJspEnv',
    kwargs={},
)

mask_ppo_sweep_config_ta41 = {
    'method': 'random',
    'metric': {
        'name': 'optimality_gap',
        'goal': 'minimize'
    },
    'parameters': {
        # only sweep over instances
        # all other values are fixed!
        "benchmark_instance": {
            'values': [
                *[f"ta{i}" for i in range(41, 51)],
                *[f"dmu{i}" for i in range(16, 20)]
            ]
        },

        # Constanst
        "total_timesteps": {
            'values': [1_000_000]
        },
        "n_envs": {
            'values': [8]
        },
        "optimal_makespan": {
            'value': 1906.0
        },

        # gamma: float = 0.99,
        # Discount factor
        "gamma": {
            'value': 'todo'
        },
        # gae_lambda: float = 0.95,
        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        "gae_lambda": {
            'value': 'todo'
        },
        # max_grad_norm: float = 0.5,
        # The maximum value for the gradient clipping
        "max_grad_norm": {
            'value': 'todo'
        },

        # learning_rate: Union[float, Schedule] = 3e-4,
        # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
        "learning_rate": {
            'value': 'todo'
        },

        # batch_size: Optional[int] = 64,
        # Minibatch size
        "batch_size": {
            'value': 'todo'
        },
        # clip_range: Union[float, Schedule] = 0.2,
        # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
        "clip_range": {
            'value': 'todo'
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
            'value': 'todo'
        },

        # ent_coef: float = 0.0,
        # Entropy coefficient for the loss calculation
        "ent_coef": {
            'value': 'todo'
        },

        # normalize_advantage: bool = True
        # Whether to normalize or not the advantage
        "normalize_advantage": {
            'value': 'todo'
        },
        # n_epochs: int = 10,
        # Number of epoch when optimizing the surrogate loss
        "n_epochs": {
            'value': 'todo'
        },

        # n_steps: int = 2048,
        # The number of steps to run for each environment per update
        # (i.e. batch size is n_steps * n_env where n_env is number of environment
        # copies running in parallel)
        "n_steps": {
            'value': 'todo'
        },
        # device: Union[th.device, str] = "auto",
        #  Device (cpu, cuda, …) on which the code should be run. Setting it to auto,
        #  the code will be run on the GPU if possible.
        "device": {
            "values": ["cpu"]  # cpu, mps, auto, cuda
        },
        # seed: Optional[int] = None,
        # Seed for the pseudo random generators
        "seed": {
            "values": [None]
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
            'value': 'todo'
        },
        "net_arch_n_size": {
            'value': 'todo'
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
            'value': 'todo'
        },

        "optimizer_eps": {  # for th.optim.Adam
            'value': 'todo'
        },

    }
}


def perform_run():
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

        logger_cb = TasselLoggerCallback(
            optimal_makespan=run_config['optimal_makespan'],
            wandb_ref=wb
        )

        callbacks = [wb_cb, logger_cb]

        jsp_std_path = get_instance_std_path_by_name(name=run_config['benchmark_instance'])

        def wrapper_function(env):
            env = TasselEnvSB3Wrapper(env)
            env = ActionMasker(env, action_mask_fn=env.valid_action_mask)
            return env

        venv = make_vec_env(
            env_id='JSSEnv:jss-v1',
            env_kwargs={
                'env_config': {'instance_path': jsp_std_path}
            },
            wrapper_class=wrapper_function,
            n_envs=sweep_params['n_envs'])

        model = sb3_contrib.MaskablePPO(
            run_config["policy_type"],
            env=venv,
            verbose=1,
            # tensorboard_log=PATHS.WAND_OUTPUT_PATH.joinpath(f"{run.name}_{run.id}"),
            **run_config["model_hyper_parameters"],
        )

        log.info(f"training the agent")
        model.learn(total_timesteps=sweep_params["total_timesteps"], callback=callbacks)


if __name__ == '__main__':
    sweep_id = wb.sweep(mask_ppo_sweep_config_ta41, project="reward-functions-comparison")
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project="reward-functions-comparison"
    )
