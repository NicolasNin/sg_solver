#!/usr/bin/env python3
"""
Hyperparameter sweep for CNN white detection.

Runs multiple training jobs with different hyperparameter combinations
and logs all results to MLflow for comparison.

Usage:
    # Default sweep (N=1-4, embedding_dim=64/128)
    python -m board_detection.ml.training.sweep_cnn_white

    # Quick sweep (fewer combinations, fewer epochs)
    python -m board_detection.ml.training.sweep_cnn_white --epochs 10

    # Custom sweep values
    python -m board_detection.ml.training.sweep_cnn_white \\
        --pool-sizes 1 2 3 4 \\
        --embedding-dims 32 64 128 \\
        --attention-temps 0.5 1.0 2.0
"""
import argparse
import itertools
import sys
from pathlib import Path
from datetime import datetime


from board_detection.ml.training.train_cnn_white import train_cnn_white, CNNHyperparams


def run_sweep(
    data_dir: Path,
    cache_path: Path,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.001,
    random_state: int = 42,
    # Sweep ranges
    pool_sizes: list = None,
    embedding_dims: list = None,
    attention_temps: list = None,
):
    """
    Run hyperparameter sweep over specified parameter ranges.
    
    All combinations are tried (grid search).
    """
    if pool_sizes is None:
        pool_sizes = [1, 2, 3, 4]
    if embedding_dims is None:
        embedding_dims = [64, 128]
    if attention_temps is None:
        attention_temps = [0.5,0.8,2]
    
    # Generate all combinations
    combinations = list(itertools.product(pool_sizes, embedding_dims, attention_temps))
    n_total = len(combinations)
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Pool sizes (N): {pool_sizes}")
    print(f"Embedding dims: {embedding_dims}")
    print(f"Attention temps: {attention_temps}")
    print(f"Total combinations: {n_total}")
    print(f"Epochs per run: {epochs}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, (pool_size, emb_dim, att_temp) in enumerate(combinations):
        print(f"\n{'#'*60}")
        print(f"# Run {i+1}/{n_total}: N={pool_size}, emb={emb_dim}, temp={att_temp}")
        print(f"{'#'*60}\n")
        
        cnn_params = CNNHyperparams(
            pool_size=pool_size,
            embedding_dim=emb_dim,
            attention_temp=att_temp
        )
        
        try:
            model, run_id = train_cnn_white(
                data_dir=data_dir,
                cache_path=cache_path,
                colorspace="lab",
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                random_state=random_state,
                cnn_params=cnn_params
            )
            
            results.append({
                "pool_size": pool_size,
                "embedding_dim": emb_dim,
                "attention_temp": att_temp,
                "run_id": run_id,
                "status": "success"
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "pool_size": pool_size,
                "embedding_dim": emb_dim,
                "attention_temp": att_temp,
                "run_id": None,
                "status": f"failed: {e}"
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Successful runs: {sum(1 for r in results if r['status'] == 'success')}/{n_total}")
    
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        print(f"  {status} N={r['pool_size']}, emb={r['embedding_dim']}, temp={r['attention_temp']}")
        if r["run_id"]:
            print(f"      run_id: {r['run_id']}")
    
    print("\nCompare in MLflow UI or use:")
    print("  mlflow ui --port 5000")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for CNN white detection")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/my_photos/"),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/dataset_cache.pkl"),
    )
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--random-state", type=int, default=42)
    
    # Sweep ranges
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Pool sizes to try (N parameter)")
    parser.add_argument("--embedding-dims", type=int, nargs="+", default=[64, 128],
                        help="Embedding dimensions to try")
    parser.add_argument("--attention-temps", type=float, nargs="+", default=[0.5,0.8],
                        help="Attention temperatures to try")
    
    args = parser.parse_args()
    
    run_sweep(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        random_state=args.random_state,
        pool_sizes=args.pool_sizes,
        embedding_dims=args.embedding_dims,
        attention_temps=args.attention_temps,
    )


if __name__ == "__main__":
    main()
