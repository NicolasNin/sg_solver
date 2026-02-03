"""
MLflow utilities for loading and managing models.
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from board_detection.ml import config


def setup_mlflow():
    """Initialize MLflow with project settings."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    

def get_or_create_experiment(name: str = None) -> str:
    """Get or create an MLflow experiment."""
    setup_mlflow()
    name = name or config.MLFLOW_EXPERIMENT_NAME
    
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id
    
    return experiment_id


def load_sklearn_model(run_id: str, artifact_path: str = "model") -> Any:
    """
    Load a sklearn model from an MLflow run.
    
    Args:
        run_id: The MLflow run ID
        artifact_path: Path to the model artifact (default: "model")
        
    Returns:
        Loaded sklearn model
    """
    setup_mlflow()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)


def load_pytorch_model(run_id: str, artifact_path: str = "model") -> Any:
    """
    Load a PyTorch model from an MLflow run.
    
    Args:
        run_id: The MLflow run ID
        artifact_path: Path to the model artifact
        
    Returns:
        Loaded PyTorch model
    """
    setup_mlflow()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.pytorch.load_model(model_uri)


def get_run_params(run_id: str) -> Dict[str, Any]:
    """Get parameters logged for a specific run."""
    setup_mlflow()
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def get_run_metrics(run_id: str) -> Dict[str, float]:
    """Get metrics logged for a specific run."""
    setup_mlflow()
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics


def list_runs(
    experiment_name: str = None,
    filter_string: str = None,
    max_results: int = 100
) -> pd.DataFrame:
    """
    List runs from an experiment.
    
    Args:
        experiment_name: Name of experiment (default: config.MLFLOW_EXPERIMENT_NAME)
        filter_string: Optional filter (e.g., "params.model_type = 'RandomForest'")
        max_results: Maximum number of runs to return
        
    Returns:
        DataFrame with run info (run_id, params, metrics, start_time)
    """
    setup_mlflow()
    experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=["start_time DESC"]
    )
    
    return runs


def get_best_run(
    metric: str,
    experiment_name: str = None,
    filter_string: str = None,
    ascending: bool = False
) -> Optional[str]:
    """
    Get the run ID with the best value for a given metric.
    
    Args:
        metric: Name of metric to optimize
        experiment_name: Name of experiment
        filter_string: Optional filter
        ascending: If True, lower is better (default: False = higher is better)
        
    Returns:
        Run ID of best run, or None if no runs found
    """
    runs = list_runs(experiment_name, filter_string)
    
    if runs.empty:
        return None
    
    metric_col = f"metrics.{metric}"
    if metric_col not in runs.columns:
        return None
    
    runs = runs.dropna(subset=[metric_col])
    if runs.empty:
        return None
    
    if ascending:
        best_idx = runs[metric_col].idxmin()
    else:
        best_idx = runs[metric_col].idxmax()
    
    return runs.loc[best_idx, "run_id"]


def compare_runs(run_ids: List[str]) -> pd.DataFrame:
    """
    Compare multiple runs side-by-side.
    
    Args:
        run_ids: List of run IDs to compare
        
    Returns:
        DataFrame with runs as rows, params/metrics as columns
    """
    setup_mlflow()
    client = MlflowClient()
    
    rows = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        row = {
            "run_id": run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
        }
        # Add params
        for k, v in run.data.params.items():
            row[f"param.{k}"] = v
        # Add metrics
        for k, v in run.data.metrics.items():
            row[f"metric.{k}"] = v
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_latest_run_by_model_type(model_type: str) -> Optional[Tuple[str, Dict]]:
    """
    Get the most recent run for a specific model type.
    
    Args:
        model_type: Value of the 'model_type' param to filter by
        
    Returns:
        Tuple of (run_id, params dict) or None if not found
    """
    runs = list_runs(filter_string=f"params.model_type = '{model_type}'")
    
    if runs.empty:
        return None
    
    run_id = runs.iloc[0]["run_id"]
    params = get_run_params(run_id)
    
    return run_id, params


def log_model_with_params(
    model: Any,
    model_type: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifact_path: str = "model",
    run_name: str = None
):
    """
    Convenience function to log a model with parameters and metrics.
    
    Args:
        model: The model to save
        model_type: Type identifier (e.g., "rf_patch_compare", "empty_cnn")
        params: Dictionary of hyperparameters
        metrics: Dictionary of evaluation metrics
        artifact_path: Name for the model artifact
        run_name: Optional name for the run
    """
    setup_mlflow()
    experiment_id = get_or_create_experiment()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Log model type as a param
        mlflow.log_param("model_type", model_type)
        
        # Log all params
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        # Log all metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Log the model
        if hasattr(model, 'predict'):  # sklearn-like
            mlflow.sklearn.log_model(model, artifact_path)
        else:  # assume PyTorch
            mlflow.pytorch.log_model(model, artifact_path)
        
        return mlflow.active_run().info.run_id


# ============================================================
# Notebook-friendly helpers (lean one-liners)
# ============================================================

def list_experiments() -> pd.DataFrame:
    """List all experiments. One-liner for notebooks."""
    setup_mlflow()
    client = MlflowClient()
    exps = client.search_experiments()
    return pd.DataFrame([{
        "name": e.name,
        "id": e.experiment_id,
        "artifact_location": e.artifact_location
    } for e in exps])


def list_runs_df(experiment_name: str, top_n: int = 20) -> pd.DataFrame:
    """
    List runs with key columns for easy viewing. One-liner for notebooks.
    
    Returns:
        DataFrame with: run_id, run_name, pool_size, embedding_dim, best_val_f1, etc.
    """
    setup_mlflow()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return pd.DataFrame()
    
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.best_val_f1 DESC"],
        max_results=top_n
    )
    
    # Select useful columns
    cols = ["run_id", "run_name", "status"]
    param_cols = [c for c in runs.columns if c.startswith("params.")]
    metric_cols = [c for c in runs.columns if c.startswith("metrics.best")]
    
    keep = cols + param_cols + metric_cols
    keep = [c for c in keep if c in runs.columns]
    
    df = runs[keep].copy()
    df["run_id"] = df["run_id"].str[:8]  # Shorten for display
    return df


def load_model_with_info(run_id: str, experiment_name: str = None):
    """
    Load a model and its params so you know what input it expects.
    
    Args:
        run_id: MLflow run ID (full or 8-char prefix)
        experiment_name: Experiment to search in (for short run_id resolution)
        
    Returns:
        (model, params_dict) - model ready to use, params dict with colorspace etc.
    
    Example:
        model, params = load_model_with_info("797fce77", "cnn_white")
        print(params["colorspace"])  # "lab"
        # Now YOU create TorchPatchDataset with that colorspace and run inference
    """
    setup_mlflow()
    
    # Handle short run_id
    if len(run_id) < 32 and experiment_name:
        runs = list_runs(experiment_name)
        if not runs.empty:
            match = runs[runs["run_id"].str.startswith(run_id)]
            if not match.empty:
                run_id = match.iloc[0]["run_id"]
    
    # Load params first
    params = get_run_params(run_id)
    
    # Load model
    model_uri = f"runs:/{run_id}/model"
    model_type = params.get("model_type", "")
    
    if "cnn" in model_type or params.get("architecture"):
        model = mlflow.pytorch.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)
    
    return model, params


