import os
import click
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import json

def export_runs_detailed(
    experiment_name: str, 
    output_dir: str = None, 
    include_artifacts: bool = True,
    tracking_uri: str = None
):
    """
    Comprehensively export runs from an MLflow experiment with proper type preservation
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        output_dir (str, optional): Directory to export runs. Defaults to experiment name.
        include_artifacts (bool, optional): Whether to export run artifacts. Defaults to True.
        tracking_uri (str, optional): MLflow tracking URI. Defaults to current.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI: {tracking_uri}")
    else:
        print(f"Using default MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    client = MlflowClient()

    output_dir = output_dir or f"{experiment_name}_runs_export"
    os.makedirs(output_dir, exist_ok=True)

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            try:
                experiment = client.get_experiment(experiment_name)
                print(f"Found experiment by ID: {experiment.name}")
            except:
                print(f"Experiment '{experiment_name}' not found as name or ID. You may want to specify --tracking-uri or check the experiment name.")
                
                experiments = client.search_experiments()
                if experiments:
                    print("\nAvailable experiments:")
                    for exp in experiments:
                        print(f"  - Name: {exp.name}, ID: {exp.experiment_id}")
                return
        
        print(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
        
        detailed_runs_data = []
        metadata = {"parameters": [], "metrics": [], "tags": []}
        
        page_token = None
        total_runs = 0
        max_results_per_page = 5000

        while True:
            runs_page = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results_per_page,
                page_token=page_token
            )
            
            runs = runs_page
            total_runs += len(runs)
            print(f"Retrieved {len(runs)} runs. Total so far: {total_runs}")
            
            for run in runs:
                run_data = {
                    'run_id': run.info.run_id,
                    'experiment_id': run.info.experiment_id,
                    'user_id': run.info.user_id,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'lifecycle_stage': run.info.lifecycle_stage,
                }
                
                for k, v in run.data.metrics.items():
                    column_name = f"metric:{k}"
                    run_data[column_name] = v
                    if column_name not in metadata["metrics"]:
                        metadata["metrics"].append(column_name)
                
                for k, v in run.data.params.items():
                    column_name = f"param:{k}"
                    run_data[column_name] = v
                    if column_name not in metadata["parameters"]:
                        metadata["parameters"].append(column_name)
                
                for k, v in run.data.tags.items():
                    column_name = f"tag:{k}"
                    run_data[column_name] = v
                    if column_name not in metadata["tags"]:
                        metadata["tags"].append(column_name)
                    
                detailed_runs_data.append(run_data)
            
            page_token = runs_page.token
            
            if page_token is None:
                break
        
        print(f"Retrieved all {total_runs} runs from experiment")
        
        runs_df = pd.DataFrame(detailed_runs_data)
        
        csv_path = os.path.join(output_dir, f"{experiment.name}_runs.csv")
        runs_df.to_csv(csv_path, index=False)
        print(f"Runs summary saved to {csv_path}")
        
        metadata_path = os.path.join(output_dir, f"{experiment.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

        if include_artifacts:
            artifacts_dir = os.path.join(output_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            for i, run in enumerate(detailed_runs_data):
                run_id = run['run_id']
                print(f"Downloading artifacts for run {i+1}/{len(detailed_runs_data)}: {run_id}")
                run_artifacts_dir = os.path.join(artifacts_dir, run_id)
                os.makedirs(run_artifacts_dir, exist_ok=True)
                
                try:
                    artifacts = client.list_artifacts(run_id)
                    for artifact in artifacts:
                        if artifact.is_dir:
                            client.download_artifacts(run_id, artifact.path, dst_path=run_artifacts_dir)
                        else:
                            artifact_path = os.path.dirname(artifact.path) if os.path.dirname(artifact.path) else ""
                            dst_dir = os.path.join(run_artifacts_dir, artifact_path)
                            os.makedirs(dst_dir, exist_ok=True)
                            client.download_artifacts(run_id, artifact.path, dst_path=dst_dir)
                except Exception as e:
                    print(f"Error downloading artifacts for run {run_id}: {e}")

        summary_report = {
            'total_runs': total_runs,
            'successful_runs': len([r for r in detailed_runs_data if r['status'] == 'FINISHED']),
            'failed_runs': len([r for r in detailed_runs_data if r['status'] == 'FAILED']),
            'experiment_name': experiment.name,
            'experiment_id': experiment.experiment_id,
            'tracking_uri': mlflow.get_tracking_uri(),
            'parameters_count': len(metadata["parameters"]),
            'metrics_count': len(metadata["metrics"]),
            'tags_count': len(metadata["tags"])
        }
        
        with open(os.path.join(output_dir, 'export_summary.json'), 'w') as f:
            json.dump(summary_report, f, indent=2)

        print(f"Export completed for experiment: {experiment.name}")
        print(f"Total runs exported: {total_runs}")
        print(f"Parameters exported: {len(metadata['parameters'])}")
        print(f"Metrics exported: {len(metadata['metrics'])}")
        print(f"Tags exported: {len(metadata['tags'])}")
        print(f"Export directory: {output_dir}")

    except Exception as e:
        print(f"Error exporting runs: {e}")
        import traceback
        traceback.print_exc()

@click.command()
@click.option('--experiment-name', required=True, help='Name or ID of the MLflow experiment')
@click.option('--output-dir', default=None, help='Directory to export runs')
@click.option('--tracking-uri', default=None, help='MLflow tracking URI (e.g. sqlite:///mlflow.db, http://localhost:5000)')
@click.option('--include-artifacts/--no-include-artifacts', default=True, help='Export run artifacts')
def main(experiment_name, output_dir, tracking_uri, include_artifacts):
    """Export MLflow runs with clear type preservation"""
    export_runs_detailed(
        experiment_name=experiment_name,
        output_dir=output_dir,
        include_artifacts=include_artifacts,
        tracking_uri=tracking_uri
    )

if __name__ == "__main__":
    main()