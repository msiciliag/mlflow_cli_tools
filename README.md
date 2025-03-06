# MLflow CLI Tools

This repository contains a set of command-line interface (CLI) tools for managing MLflow experiments. The tools are implemented in Python and use the MLflow library to interact with default MLflow tracking servers.

## Setup

```sh
git clone https://github.com/msiciliag/mlflow_cli_tools.git
cd mlflow_cli_tools

python -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
```


## Tools Overview

### 1. `clear_runs.py`

Deletes all runs in a specified MLflow experiment.

#### Usage
```sh
python clear_runs.py [OPTIONS]
```

#### Options
- `--experiment-name TEXT` (required): Name of the MLflow experiment.
- `--tracking-uri TEXT`: MLflow tracking URI (e.g. `sqlite:///mlflow.db`, `http://localhost:5000`).

#### Examples
```sh
# Clear runs in the 'MyExperiment' experiment using a specified tracking URI
python clear_runs.py --experiment-name MyExperiment --tracking-uri http://localhost:5000
```

### 2. `delete_exp.py`

Deletes a specified MLflow experiment.

#### Usage
```sh
python delete_exp.py [OPTIONS]
```

#### Options
- `--experiment-name TEXT` (required): Name of the MLflow experiment.
- `--tracking-uri TEXT`: MLflow tracking URI (e.g. `sqlite:///mlflow.db`, `http://localhost:5000`).

#### Examples
```sh
# Delete the 'MyExperiment' experiment using a specified tracking URI
python delete_exp.py --experiment-name MyExperiment --tracking-uri http://localhost:5000
```

### 3. `restore_exp.py`

Restores specified MLflow experiments.

#### Usage
```sh
python restore_exp.py [OPTIONS]
```

#### Options
- `--experiment-names TEXT` (required) (can be specified multiple times): Names of the MLflow experiments.
- `--tracking-uri TEXT`: MLflow tracking URI (e.g. `sqlite:///mlflow.db`, `http://localhost:5000`).

#### Examples
```sh
# Restore multiple experiments using a specified tracking URI
python restore_exp.py --experiment-names MyExperiment --experiment-names "MyExperiment 2" --tracking-uri http://localhost:5000
```

### 4. `export_exp.py`

Exports runs from a specified MLflow experiment to a CSV file, also exports metadata for future imports.
> **_NOTE:_** Only exports parameters, metrics and artifacts, other details are not considered.

#### Usage
```sh
python export_exp.py [OPTIONS]
```

#### Options
- `--experiment-name TEXT` (required): Name or ID of the MLflow experiment.
- `--output-dir TEXT`: Directory to export runs.
- `--tracking-uri TEXT`: MLflow tracking URI (e.g. `sqlite:///mlflow.db`, `http://localhost:5000`).
- `--include-artifacts/--no-include-artifacts`: Export run artifacts.

#### Examples
```sh
# Export runs from the 'LogisticRegression' experiment to the default directory
python export_exp.py --experiment-name MyExperiment

# Export runs from the 'LogisticRegression' experiment to a specified directory
python export_exp.py --experiment-name MyExperiment --output-dir ./exports
```

### 5. `import_exp.py`

Imports runs from a CSV file into a specified MLflow experiment.
> **_NOTE:_**  This might not work as desired if the CSV provided is not an output of `export_exp.py` or it was modified

#### Usage
```sh
python import_exp.py [OPTIONS]
```

#### Options
- `--csv-path TEXT` (required): Path to the runs CSV file.
- `--experiment-name TEXT` (required): Destination experiment name.
- `--artifacts-dir TEXT`: Directory containing run artifacts.
- `--metadata-path TEXT`: Path to metadata JSON file.
- `--tracking-uri TEXT`: MLflow tracking URI (e.g. `sqlite:///mlflow.db`, `http://localhost:5000`).
- `--import-artifacts/--no-import-artifacts`: Import run artifacts.
- `--create-experiment/--no-create-experiment`: Create experiment if it does not exist.

#### Examples
```sh
# Import runs from a CSV file into the 'NewExperiment' experiment
python import_exp.py --csv-path MyPath/MyExperiment_runs_export/MyExperiment_runs.csv --experiment-name NewExperiment

# Import runs from a CSV file into the 'NewExperiment' experiment including artifacts
python import_exp.py --csv-path ./exports/MyExperiment_runs_export/MyExperiment_runs.csv  --experiment-name NewExperiment --artifacts-dir ./exports/MyExperiment_runs_export/artifacts --import-artifacts
```
