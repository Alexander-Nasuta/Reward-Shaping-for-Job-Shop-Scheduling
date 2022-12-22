import time

from types import ModuleType

import wandb as wb

from statistics import mean

from stable_baselines3.common.callbacks import BaseCallback


class TasselLoggerCallback(BaseCallback):

    def __init__(self, optimal_makespan: float, wandb_ref: ModuleType = wb, verbose=0):
        super(TasselLoggerCallback, self).__init__(verbose)
        self.optimal_makespan = optimal_makespan
        self.makespans = None
        self.opti_gaps = None
        self.wandb_ref = wandb_ref
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _on_step(self) -> bool:
        if not self.makespans:
            self.makespans = [0 for _ in range(self.training_env.num_envs)]
            self.opti_gaps = [0 for _ in range(self.training_env.num_envs)]

        elapsed_time = time.perf_counter() - self.start_time
        logs = {
            "num_timesteps": self.num_timesteps,
            "time [sec]": elapsed_time,
            "time [min]": elapsed_time / 60,
        }
        for i, (nbla, lts, cts) in enumerate(zip(
                self.training_env.get_attr("nb_legal_actions"),
                self.training_env.get_attr("last_time_step"),
                self.training_env.get_attr("current_time_step"),
        )):
            if cts == 0 and not lts == float('inf'):
                logs[f"makespan_env_{i}"] = lts
                opti_gap = lts / self.optimal_makespan - 1.0
                logs[f"optimality_gap_env_{i}"] = opti_gap
                self.makespans[i] = lts
                self.opti_gaps[i] = opti_gap
                logs[f"makespan_mean"] = mean([e for e in self.makespans if e != 0])
                logs[f"optimality_gap_mean"] = mean([e for e in self.opti_gaps if e != 0])
        if self.wandb_ref:
            self.wandb_ref.log(logs)
        return True

    def on_training_end(self) -> None:
        if self.wandb_ref:
            self.wandb_ref.log({
                "optimality_gap": mean([e for e in self.opti_gaps if e != 0])
            })
