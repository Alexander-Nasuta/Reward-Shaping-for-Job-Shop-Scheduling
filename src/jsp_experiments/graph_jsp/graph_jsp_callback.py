import time

from statistics import mean
from typing import List

import wandb as wb

from types import ModuleType

from stable_baselines3.common.callbacks import BaseCallback


class GraphJspLoggerCallback(BaseCallback):

    def __init__(self, optimal_makespan: float, wandb_ref: ModuleType = wb, verbose=0):
        super(GraphJspLoggerCallback, self).__init__(verbose)

        self.wandb_ref = wandb_ref
        self.start_time = None

        self.optimal_makespan = optimal_makespan

        self.total_left_shifts = 0


        self.senv_fields = [
            "makespan",
        ]
        self.venv_fields = [
        ]

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _get_vals(self, field: str) -> List:
        return [env_info[field] for env_info in self.locals['infos'] if field in env_info.keys()]

    def _on_step(self) -> bool:
        elapsed_time = time.perf_counter() - self.start_time
        logs = {
            "num_timesteps": self.num_timesteps,
            "time [sec]": elapsed_time,
            "time [min]": elapsed_time / 60,
        }

        ls_list = self._get_vals("left_shift")
        if len(ls_list):
            self.total_left_shifts += sum(ls_list)
        if self.wandb_ref:
            logs = {
                **{
                    f"{f}_env_{i}": info[f]
                    for i, info in enumerate(self.locals['infos'])
                    for f in self.senv_fields
                    if f in info.keys()
                },
                **{
                    f"optimality_gap_env_{i}": info["makespan"] / self.optimal_makespan - 1.0
                    for i, info in enumerate(self.locals['infos'])
                    if "makespan" in info.keys()
                },
                **{f"{f}_mean": mean(self._get_vals(f)) for f in self.senv_fields if self._get_vals(f)},
                **{
                    f"optimality_gap": mean(self._get_vals("makespan")) / self.optimal_makespan - 1.0 for _ in [1]
                    if self._get_vals("makespan")
                },
                **{f"{f}": mean(self._get_vals(f)) for f in self.venv_fields if self._get_vals(f)},
                "total_left_shifts": self.total_left_shifts,
                "left_shift_pct": self.total_left_shifts / self.num_timesteps * 100,
                **logs
            }
            self.wandb_ref.log(logs)

        return True
