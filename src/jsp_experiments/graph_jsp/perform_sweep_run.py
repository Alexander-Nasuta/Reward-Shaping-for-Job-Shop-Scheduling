import pprint

import sb3_contrib
import wandb
import wandb as wb
import numpy as np

from graph_jsp_env.disjunctive_graph_logger import log
from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from jsp_experiments.graph_jsp.graph_jsp_callback import GraphJspLoggerCallback
from jsp_experiments.instance_loader import get_instance_by_name_as_numpy_array
from jsp_experiments.merge_default_and_sweep_config import merge_configs_graph_jss_env


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

        reward_function_parameters = {
            "gamma": sweep_params['samsonov-gamma'],
            't_opt': sweep_params['optimal_makespan'],
        } if sweep_params["samsonov"] else {
            'scaling_divisor': sweep_params['optimal_makespan'],
        }

        env_kwargs = {
            "jps_instance": jsp_instance,
            'reward_function': sweep_params['reward_function'],
            "normalize_observation_space": sweep_params['normalize_observation_space'],
            "flat_observation_space": sweep_params['flat_observation_space'],
            "perform_left_shift_if_possible": sweep_params['perform_left_shift_if_possible'],
            "reward_function_parameters": reward_function_parameters,
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