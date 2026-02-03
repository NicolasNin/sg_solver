#!/usr/bin/env python3
"""
Train a RandomForest model for patch comparison (same color prediction).

This model predicts whether two patches from the same image have the same color.
It's used in the pipeline to build the similarity matrix for clustering.

Usage:
    # Single run
    python -m board_detection.ml.training.train_rf_compare --n-estimators 100

    # Multiple folds with different random states (single MLflow run with step metrics)
    python -m board_detection.ml.training.train_rf_compare --n-folds 5

    # Specific random states
    python -m board_detection.ml.training.train_rf_compare --random-states 42 43 44 45 46
"""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Add project root to path

from board_detection.ml import config
from board_detection.ml.datasets import BoardPatchDataset
from board_detection.ml.utils import setup_mlflow, get_or_create_experiment


def train_rf_compare(
    data_dir: Path,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    random_states: list = None,
    test_size: float = 0.2,
    cache_path: Path = None,
    force_reload: bool = False
):
    """
    Train RandomForest for patch comparison with optional multiple folds.
    
    When multiple random_states are provided:
    - All folds logged in a SINGLE MLflow run with step=fold_idx
    - Confusion matrices logged per fold
    - Only the BEST model (by test_accuracy) is stored
    - Summary stats (mean, std) logged at the end
    
    Args:
        data_dir: Directory with annotated images
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split
        random_states: List of random states to try (multiple folds)
        test_size: Fraction for test set
        cache_path: Path to cache processed dataset (speeds up reruns)
        force_reload: If True, recompute dataset even if cache exists
    """
    if random_states is None:
        random_states = [43]
    
    n_folds = len(random_states)
    
    # Setup MLflow with dedicated experiment for this model type
    setup_mlflow()
    experiment_id = get_or_create_experiment("rf_patch_compare")
    
    # Load dataset (with caching)
    if cache_path:
        dataset = BoardPatchDataset.load_or_create(
            cache_path=cache_path,
            data_dir=data_dir,
            skip_bad=True,
            force_reload=force_reload
        )
    else:
        print(f"Loading data from {data_dir}...")
        dataset = BoardPatchDataset()
        dataset.load_from_directory(data_dir, skip_bad=True)
    
    print(f"Loaded {dataset}")
    
    # Get pair features (once for all folds)
    print("Computing color distance features...")
    X, y, image_ids = dataset.get_color_distance_features()
    print(f"Total pairs: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create run name
    if n_folds > 1:
        run_name = f"rf_patch_compare_{n_folds}folds"
    else:
        run_name = f"rf_patch_compare_rs{random_states[0]}"
    
    # Single MLflow run for all folds
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Log fixed parameters (same across folds)
        mlflow.log_param("model_type", "rf_patch_compare")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth if max_depth else "None")
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_images", len(dataset))
        mlflow.log_param("data_dir", str(data_dir))
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("random_states", str(random_states))
        
        # Track best model
        best_model = None
        best_accuracy = -1
        best_fold_idx = -1
        best_random_state = -1
        
        # Collect metrics across folds
        all_metrics = []
        
        for fold_idx, rs in enumerate(random_states):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{n_folds} (random_state={rs})")
            print(f"{'='*50}")
            
            # Split by image (no leakage)
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
            train_idx, test_idx = next(gss.split(X, y, groups=image_ids))
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Train model
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=rs,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            
            # Compute metrics
            metrics = {
                "train_accuracy": accuracy_score(y_train, y_pred_train),
                "train_f1": f1_score(y_train, y_pred_train),
                "test_accuracy": accuracy_score(y_test, y_pred_test),
                "test_precision": precision_score(y_test, y_pred_test),
                "test_recall": recall_score(y_test, y_pred_test),
                "test_f1": f1_score(y_test, y_pred_test),
            }
            all_metrics.append({"random_state": rs, "fold_idx": fold_idx, **metrics})
            
            # Log metrics WITH STEP (this gives multi-bar chart!)
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step=fold_idx)
            
            # Log random_state as metric so it shows in chart
            mlflow.log_metric("random_state", rs, step=fold_idx)
            
            print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
            print(f"Test F1: {metrics['test_f1']:.4f}")
            
            # Confusion matrix for this fold
            cm = confusion_matrix(y_test, y_pred_test)
            
            # Save as JSON
            cm_dict = {
                "confusion_matrix": cm.tolist(),
                "labels": ["different", "same"],
                "random_state": rs,
                "fold_idx": fold_idx,
                "test_accuracy": metrics["test_accuracy"],
                "test_f1": metrics["test_f1"]
            }
            mlflow.log_text(
                json.dumps(cm_dict, indent=2), 
                f"metrics/confusion_matrix_fold{fold_idx}_rs{rs}.json"
            )
            
            # Save confusion matrix as image
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["different", "same"])
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Fold {fold_idx} (rs={rs})\nacc={metrics['test_accuracy']:.4f}")
            plt.tight_layout()
            
            cm_path = Path("/tmp") / f"cm_fold{fold_idx}_rs{rs}.png"
            plt.savefig(cm_path, dpi=100)
            plt.close()
            mlflow.log_artifact(str(cm_path), "confusion_matrices")
            
            # Track best model
            if metrics["test_accuracy"] > best_accuracy:
                best_accuracy = metrics["test_accuracy"]
                best_model = clf
                best_fold_idx = fold_idx
                best_random_state = rs
                # Also track feature importances of best
                best_importances = clf.feature_importances_
        
        # Log summary statistics (no step = final values)
        test_accs = [m["test_accuracy"] for m in all_metrics]
        test_f1s = [m["test_f1"] for m in all_metrics]
        
        mlflow.log_metric("mean_test_accuracy", np.mean(test_accs))
        mlflow.log_metric("std_test_accuracy", np.std(test_accs))
        mlflow.log_metric("mean_test_f1", np.mean(test_f1s))
        mlflow.log_metric("std_test_f1", np.std(test_f1s))
        mlflow.log_metric("best_fold_idx", best_fold_idx)
        mlflow.log_metric("best_random_state", best_random_state)
        
        # Log feature importances from best model
        feature_names = ["delta_theta", "delta_chroma", "delta_L", "delta_hue", "min_chroma"]
        for name, importance in zip(feature_names, best_importances):
            mlflow.log_metric(f"importance_{name}", importance)
        
        # Save ONLY the best model
        print(f"\nSaving best model (fold {best_fold_idx}, rs={best_random_state}, acc={best_accuracy:.4f})")
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_param("best_fold_random_state", best_random_state)
        
        # Log dataset info
        mlflow.log_text(dataset.get_image_hash_summary(), "dataset/image_hashes.txt")
        
        # Log all fold results as JSON for later analysis
        mlflow.log_text(json.dumps(all_metrics, indent=2), "metrics/all_folds_summary.json")
        
        run_id = mlflow.active_run().info.run_id
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        if n_folds > 1:
            print(f"Folds: {n_folds}")
            print(f"Random states: {random_states}")
            print(f"Test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
            print(f"Test F1:       {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
            print(f"\nBest fold: {best_fold_idx} (rs={best_random_state})")
        else:
            print(f"Test accuracy: {test_accs[0]:.4f}")
            print(f"Test F1:       {test_f1s[0]:.4f}")
        
        print(f"\nRun ID: {run_id}")
        print(f"View at: {config.MLFLOW_TRACKING_URI}")
        
        return all_metrics, run_id


def main():
    parser = argparse.ArgumentParser(description="Train RF model for patch comparison")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/my_photos/"),
        help="Directory with annotated images"
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Min samples to split")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    
    # Caching options
    parser.add_argument(
        "--cache-path", 
        type=Path, 
        default=Path("/home/nicolas/code/star_genius_solver/data/dataset_cache.pkl"),
        help="Path to cache dataset (speeds up reruns)"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--force-reload", action="store_true", help="Recompute dataset even if cache exists")
    
    # Random state options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random-state", type=int, default=43, help="Single random seed")
    group.add_argument("--random-states", type=int, nargs="+", help="Multiple random seeds")
    group.add_argument("--n-folds", type=int, help="Number of folds (auto-generates random states)")
    
    args = parser.parse_args()
    
    # Determine random states to use
    if args.random_states:
        random_states = args.random_states
    elif args.n_folds:
        random_states = list(range(42, 42 + args.n_folds))
    else:
        random_states = [args.random_state]
    
    # Determine cache path
    cache_path = None if args.no_cache else args.cache_path
    
    train_rf_compare(
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_states=random_states,
        test_size=args.test_size,
        cache_path=cache_path,
        force_reload=args.force_reload
    )


if __name__ == "__main__":
    main()

