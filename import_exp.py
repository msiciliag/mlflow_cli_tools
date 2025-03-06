import os
import click
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional, Tuple

def import_runs(
    csv_path: str,
    experiment_name: str,
    artifacts_dir: Optional[str] = None,
    import_artifacts: bool = True,
    create_experiment: bool = True,
    metadata_path: Optional[str] = None,
    tracking_uri: Optional[str] = None
):
    """
    Import runs from a CSV export with proper parameter and metric handling based on prefixes
    
    Args:
        csv_path (str): Path to the CSV file containing run data
        experiment_name (str): Destination experiment name
        artifacts_dir (str, optional): Directory containing run artifacts
        import_artifacts (bool): Whether to import artifacts
        create_experiment (bool): Whether to create the experiment if it doesn't exist
        metadata_path (str, optional): Path to the metadata JSON file (if available)
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
        if not experiment and create_experiment:
            experiment_id = client.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        elif not experiment:
            raise ValueError(f"Experiment '{experiment_name}' does not exist, you may want to specify --tracking-uri")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        return

    try:
        runs_df = pd.read_csv(csv_path)
        print(f"Loaded {len(runs_df)} runs from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    metadata = {"parameters": [], "metrics": [], "tags": []}
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata from {metadata_path}")
            print(f"Found {len(metadata['parameters'])} parameters, {len(metadata['metrics'])} metrics, and {len(metadata['tags'])} tags")
        except Exception as e:
            print(f"Error loading metadata, will rely on column prefixes: {e}")
    
    imported_runs = []
    failed_runs = []

    for index, run_data in runs_df.iterrows():
        try:
            print(f"Importing run {index+1}/{len(runs_df)}")
            
            params, metrics, tags = extract_run_data(run_data, metadata)
            
            with mlflow.start_run(experiment_id=experiment_id) as active_run:
                if params:
                    mlflow.log_params(params)
                    print(f"Logged {len(params)} parameters")

                if metrics:
                    mlflow.log_metrics(metrics)
                    print(f"Logged {len(metrics)} metrics")

                if tags:
                    mlflow.set_tags(tags)
                    print(f"Set {len(tags)} tags")

                if import_artifacts and artifacts_dir:
                    original_run_id = str(run_data.get('run_id', ''))
                    if original_run_id:
                        run_artifacts_path = os.path.join(artifacts_dir, 'artifacts', original_run_id)
                        
                        if os.path.exists(run_artifacts_path):
                            print(f"Importing artifacts from {run_artifacts_path}")
                            artifact_count = 0
                            for root, _, files in os.walk(run_artifacts_path):
                                for file in files:
                                    full_path = os.path.join(root, file)
                                    rel_path = os.path.relpath(full_path, run_artifacts_path)
                                    
                                    try:
                                        artifact_path = os.path.dirname(rel_path) if os.path.dirname(rel_path) else None
                                        mlflow.log_artifact(full_path, artifact_path=artifact_path)
                                        artifact_count += 1
                                    except Exception as artifact_error:
                                        print(f"Error logging artifact {rel_path}: {artifact_error}")
                            
                            print(f"Imported {artifact_count} artifacts")
                        else:
                            print(f"No artifacts directory found at {run_artifacts_path}")

                imported_runs.append(active_run.info.run_id)
                print(f"Successfully imported run to {active_run.info.run_id}")

        except Exception as e:
            print(f"Failed to import run {index}: {e}")
            import traceback
            traceback.print_exc()
            failed_runs.append(index)

    summary = {
        'total_runs_attempted': len(runs_df),
        'successful_imports': len(imported_runs),
        'failed_imports': len(failed_runs),
        'destination_experiment': experiment_name,
        'destination_experiment_id': experiment_id,
        'tracking_uri': mlflow.get_tracking_uri(),
        'failed_run_indices': failed_runs
    }

    summary_path = os.path.join(os.path.dirname(csv_path), 'import_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nImport Summary for Experiment '{experiment_name}':")
    print(f"Total Runs Attempted: {summary['total_runs_attempted']}")
    print(f"Successful Imports: {summary['successful_imports']}")
    print(f"Failed Imports: {summary['failed_imports']}")
    
    if failed_runs:
        print("Failed Run Indices:", failed_runs)

    return summary

def extract_run_data(run_data: pd.Series, metadata: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Extract parameters, metrics, and tags from a run data series based on prefixes
    
    Args:
        run_data (pd.Series): Series containing run data
        metadata (Dict): Dictionary with parameter, metric, and tag lists
    
    Returns:
        Tuple[Dict, Dict, Dict]: Tuple of parameters, metrics, and tags dictionaries
    """
    params = {}
    metrics = {}
    tags = {}
    
    system_columns = [
        'run_id', 'experiment_id', 'user_id', 'start_time', 
        'end_time', 'status', 'lifecycle_stage'
    ]
    
    for col in run_data.index:
        if pd.isna(run_data[col]) or col in system_columns:
            continue
            
        if col.startswith('param:'):
            param_name = col[6:]  
            params[param_name] = str(run_data[col])
        elif col.startswith('metric:'):
            metric_name = col[7:] 
            try:
                metrics[metric_name] = float(run_data[col])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert metric '{metric_name}' to float. Value: {run_data[col]}")
        elif col.startswith('tag:'):
            tag_name = col[4:]  
            tags[tag_name] = str(run_data[col])
        elif col in metadata['parameters']:
            param_name = col
            params[param_name] = str(run_data[col])
        elif col in metadata['metrics']:
            metric_name = col
            try:
                metrics[metric_name] = float(run_data[col])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert metric '{metric_name}' to float. Value: {run_data[col]}")
        elif col in metadata['tags']:
            tag_name = col
            tags[tag_name] = str(run_data[col])
        else:
            try:
                float_val = float(run_data[col])
                # If it's an integer-like float, it's likely a parameter, only used when no metadata specified.
                if float_val.is_integer() and abs(float_val) < 1000:
                    params[col] = str(run_data[col])
                else:
                    metrics[col] = float_val
                    print(f"Warning: Inferring '{col}' as metric based on value type.")
            except (ValueError, TypeError):
                params[col] = str(run_data[col])
                print(f"Warning: Inferring '{col}' as parameter based on value type.")
    
    return params, metrics, tags

@click.command()
@click.option('--csv-path', required=True, help='Path to the runs CSV file')
@click.option('--experiment-name', required=True, help='Destination experiment name')
@click.option('--artifacts-dir', default=None, help='Directory containing run artifacts')
@click.option('--metadata-path', default=None, help='Path to metadata JSON file')
@click.option('--tracking-uri', default=None, help='MLflow tracking URI (e.g. sqlite:///mlflow.db, http://localhost:5000)')
@click.option('--import-artifacts/--no-import-artifacts', default=True, help='Import run artifacts')
@click.option('--create-experiment/--no-create-experiment', default=True, help='Create experiment if it does not exist')
def main(csv_path, experiment_name, artifacts_dir, metadata_path, tracking_uri, import_artifacts, create_experiment):
    """Import MLflow runs from a CSV export with proper type handling"""
    import_runs(
        csv_path=csv_path, 
        experiment_name=experiment_name,
        artifacts_dir=artifacts_dir,
        import_artifacts=import_artifacts,
        create_experiment=create_experiment,
        metadata_path=metadata_path,
        tracking_uri=tracking_uri
    )

if __name__ == "__main__":
    main()