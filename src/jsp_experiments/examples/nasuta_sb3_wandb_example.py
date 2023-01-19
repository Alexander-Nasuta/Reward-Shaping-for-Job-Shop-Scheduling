import gym
import wandb as wb
import sb3_contrib
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from jsp_experiments.graph_jsp.graph_jsp_callback import GraphJspLoggerCallback
from jsp_experiments.instance_loader import get_instance_by_name_as_numpy_array
from graph_jsp_env.disjunctive_graph_logger import log

gym.envs.register(
    id='gjsp-v0',
    entry_point='graph_jsp_env.disjunctive_graph_jsp_env:DisjunctiveGraphJspEnv',
    kwargs={},
)

if __name__ == '__main__':
    jsp = get_instance_by_name_as_numpy_array("ft06")


    def mask_fn(env):
        return env.valid_action_mask()

    venv = make_vec_env(
        env_id='gjsp-v0',
        env_kwargs={
            "jps_instance": jsp,

            "normalize_observation_space": True,
            "flat_observation_space": True,
            "perform_left_shift_if_possible": True,
            "reward_function": 'nasuta',
            "reward_function_parameters": {
                "scaling_divisor": 1.0
            },
            "default_visualisations": [
                "gantt_window",
            ]
        },
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=8
    )

    venv.reset()
    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
        device='auto'  # cpu, mps (mac), cuda
    )

    log.info("training...")

    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            project="dev",
    ) as run:
        logger_cb = GraphJspLoggerCallback(
            optimal_makespan=55.0,
            wandb_ref=wb
        )
        wb_cb = WandbCallback(
            gradient_save_freq=100,
            verbose=1,
        )
        model.learn(total_timesteps=1_000, progress_bar=False, callback=[wb_cb, logger_cb])
