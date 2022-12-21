import collections
import time
from typing import Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import wandb

from ortools.sat.python import cp_model

from graph_jsp_env.disjunctive_graph_logger import log
from graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from jsp_experiments.instance_loader import get_instance_by_name_as_numpy_array


class OrToolsWandbCallback(CpSolverSolutionCallback):
    """Display the objective value and time of intermediate solutions."""

    def __init__(self, optimal_makespan: float = None):
        CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.perf_counter()
        self.optimal_makespan = optimal_makespan
        wandb.log({
            'time [min]': 0,
            'time [sec]': 0,
            'solution_count': self.__solution_count
        })

    def on_solution_callback(self):
        """Called on each new solution."""
        obj = self.ObjectiveValue()
        self.__solution_count += 1

        elapsed_time = time.perf_counter() - self.start_time
        logs = {
            'time [sec]': elapsed_time,
            'time [min]': elapsed_time/60,
            'makespan': obj,
            'solution_count': self.__solution_count
        }
        if self.optimal_makespan:
            logs['optimality_gap'] = obj / self.optimal_makespan - 1.0
        wandb.log(logs)

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count


def solve_jsp(jsp_instance: np.ndarray, plot_results: bool = True, wb_config=None, max_time_sec=60) -> (
        float, str, pd.DataFrame, dict):
    if wb_config is None:
        wb_config = {
            "jsp_instance": jsp_instance
        }
    wandb.init(config=wb_config)

    # Create the model.
    model = cp_model.CpModel()

    machine_order = jsp_instance[0]
    processing_times = jsp_instance[1]

    machines_count = machine_order.max() + 1  # first machine is indexed 0
    all_machines = range(machines_count)

    horizon = np.sum(np.ravel(processing_times))

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, (job_machine_order, job_processing_times) in enumerate(zip(machine_order, processing_times)):
        for task_id, (machine, duration) in enumerate(zip(job_machine_order, job_processing_times)):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job_machine_order in enumerate(machine_order):
        for task_id in range(len(job_machine_order) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job_machine_order) - 1].end
        for job_id, job_machine_order in enumerate(machine_order)
    ])

    model.Minimize(obj_var)

    # Solve model.

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_sec
    wandb_callback = OrToolsWandbCallback(
        optimal_makespan=wb_config["optimal_makespan"]
    )

    start = time.perf_counter()

    status = solver.Solve(model, solution_callback=wandb_callback)

    end = time.perf_counter()

    solving_duration = end - start

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

        log.info(f"{status} solution found.")

        # Create one list of assigned tasks per machine.
        assigned_jobs = []
        for job_id, (job_machine_order, job_processing_times) in enumerate(zip(machine_order, processing_times)):
            for task_id, (machine, duration) in enumerate(zip(job_machine_order, job_processing_times)):
                assigned_jobs.append({
                    'Task': f'Job {job_id}',
                    'Start': solver.Value(all_tasks[job_id, task_id].start),
                    'Finish': solver.Value(all_tasks[job_id, task_id].start) + duration,
                    'Resource': f'Machine {machine}'
                })

        df = pd.DataFrame(assigned_jobs)

        if plot_results:
            # generate colors for visualizer
            c_map = plt.cm.get_cmap("rainbow")  # select the desired cmap
            arr = np.linspace(0, 1, machines_count)  # create a list with numbers from 0 to 1 with n items
            machine_colors = {m_id: c_map(val) for m_id, val in enumerate(arr)}
            colors = {f"Machine {m_id}": (r, g, b) for m_id, (r, g, b, a) in machine_colors.items()}

            visualizer = DisjunctiveGraphJspVisualizer(dpi=80)
            visualizer.gantt_chart_console(df, colors)

        makespan = solver.ObjectiveValue()

        log.info(f"or tools solving duration: {solving_duration:2f} sec")
        log.info(f'makespan: {makespan} ({status} solution)')
        # visualizer.render_gantt_in_window(df, colors=colors, wait=None)

        info = {
            "makespan": makespan,
            "solving_duration": solving_duration,
            "gantt_df": df,
            "or_tools_status": status,
            "or_tools_solving_duration": solving_duration,
            "or_tools_makespan": makespan
        }

        return makespan, status, df, info
    else:
        log.info("could not solve jsp instance. Check if your instance is well defined.")
        raise RuntimeError("could not solve jsp instance. Check if your instance is well defined.")


if __name__ == '__main__':
    instance_name = "ta41"
    jsp = get_instance_by_name_as_numpy_array(name=instance_name)

    wb_config = {
        "instance_name": instance_name,
        "optimal_makespan": 1906,
        "jsp_instance": jsp,
        "solving_environment": "or-tools",
        "max_time_sec": 60,
    }

    solve_jsp(jsp_instance=jsp, wb_config=wb_config, max_time_sec=wb_config["max_time_sec"])
