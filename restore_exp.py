import mlflow, click
from mlflow.tracking import MlflowClient

def restore_experiments(experiment_names, tracking_uri):
    """
    Restores deleted MLflow experiments by their names.

    Args:
        experiment_names (list): A list of experiment names to restore.
        tracking_uri (str, optional): MLflow tracking URI
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI: {tracking_uri}")
    else:
        print(f"Using default MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
    client = MlflowClient()

    try:
        # List all experiments, including deleted ones
        all_experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

        for experiment_name in experiment_names:
            # Find the deleted experiment by name
            deleted_experiment = next(
                (exp for exp in all_experiments if exp.name == experiment_name and exp.lifecycle_stage == "deleted"),
                None,
            )

            if deleted_experiment:
                client.restore_experiment(deleted_experiment.experiment_id)
                print(f"Experiment '{experiment_name}' restored successfully.")
            else:
                print(f"Deleted experiment '{experiment_name}' not found.")

    except Exception as e:
        print(f"Error restoring experiments: {e}")

@click.command()
@click.option('--experiment-names', required=True, multiple=True, help='Name of MLflow experiment (can be specified multiple times)')
@click.option('--tracking-uri', default=None, help='MLflow tracking URI (e.g. sqlite:///mlflow.db, http://localhost:5000)')
def main(experiment_names, tracking_uri):
    """Restore MLflow experiments given their names"""
    restore_experiments(experiment_names=experiment_names, tracking_uri=tracking_uri)

if __name__ == "__main__":
    main()