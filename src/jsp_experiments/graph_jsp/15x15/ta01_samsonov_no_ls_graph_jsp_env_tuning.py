
import gym
import wandb as wb

from jsp_experiments.graph_jsp.perform_sweep_run import perform_run

# this import is actually important
import graph_jsp_env.disjunctive_graph_jsp_visualizer

gym.envs.register(
    id='gjsp-v0',
    entry_point='graph_jsp_env.disjunctive_graph_jsp_env:DisjunctiveGraphJspEnv',
    kwargs={},
)

if __name__ == '__main__':
    sweep_id = 'ecz2ay3r'
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project="reward-functions-comparison"
    )
