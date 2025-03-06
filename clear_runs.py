import mlflow, click
from mlflow.tracking import MlflowClient

def delete_all_runs_in_experiment(experiment_name, tracking_uri):
    """
    Deletes all runs within a given MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str, optional): MLflow tracking URI
    """

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI: {tracking_uri}")
    else:
        print(f"Using default MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return

        runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)

        for run in runs:
            run_id = run.info.run_id
            client.delete_run(run_id)
            print(f"Deleted run: {run_id}")

        print(f"Deleted {len(runs)} runs in experiment: {experiment_name}")

    except Exception as e:
        print(f"Error deleting runs: {e}")

@click.command()
@click.option('--experiment-name', required=True, help='Name of the MLflow experiment')
@click.option('--tracking-uri', default=None, help='MLflow tracking URI (e.g. sqlite:///mlflow.db, http://localhost:5000)')
def main(experiment_name, tracking_uri):
    """Clear runs given an experiment name"""
    delete_all_runs_in_experiment(experiment_name=experiment_name, tracking_uri=tracking_uri)

if __name__ == "__main__":
    main()