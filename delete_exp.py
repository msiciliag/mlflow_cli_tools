import mlflow, click
from mlflow.tracking import MlflowClient

def delete_mlflow_experiment(experiment_name, tracking_uri):
    """
    Deletes an MLflow experiment by its name.

    Args:
        experiment_name (str): The name of the MLflow experiment to delete.
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

        client.delete_experiment(experiment.experiment_id)
        print(f"Experiment '{experiment_name}' deleted successfully.")

    except Exception as e:
        print(f"Error deleting experiment '{experiment_name}': {e}")

@click.command()
@click.option('--experiment-name', required=True, help='Name of the MLflow experiment')
@click.option('--tracking-uri', default=None, help='MLflow tracking URI (e.g. sqlite:///mlflow.db, http://localhost:5000)')
def main(experiment_name, tracking_uri):
    """Delete an MLflow experiment given a name"""
    delete_mlflow_experiment(experiment_name=experiment_name, tracking_uri=tracking_uri)

if __name__ == "__main__":
    main()