
import gym
import wandb as wb

from jsp_experiments.graph_jsp.perform_sweep_run import perform_run

gym.envs.register(
    id='gjsp-v0',
    entry_point='graph_jsp_env.disjunctive_graph_jsp_env:DisjunctiveGraphJspEnv',
    kwargs={},
)

if __name__ == '__main__':
    sweep_id = 'dggnil9c'
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project="reward-functions-comparison"
    )
