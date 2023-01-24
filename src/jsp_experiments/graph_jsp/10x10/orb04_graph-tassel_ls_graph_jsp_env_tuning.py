
import wandb as wb

from jsp_experiments.graph_jsp.perform_sweep_run import perform_run

if __name__ == '__main__':
    sweep_id = 'f9uy8bd2'
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project="dev"
    )
