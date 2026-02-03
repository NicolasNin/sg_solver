#!/usr/bin/env python3
"""
Train a CNN for white patch detection.

This model predicts whether a patch is white (background) vs non-white.

Usage:
    python -m board_detection.ml.training.train_cnn_white --epochs 50 --lr 0.001

    # With different colorspace and architecture params
    python -m board_detection.ml.training.train_cnn_white --colorspace gray --pool-size 2 --embedding-dim 32
"""
import argparse
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


from board_detection.ml import config
from board_detection.ml.datasets import BoardPatchDataset, TorchPatchDataset
from board_detection.ml.utils import setup_mlflow, get_or_create_experiment
from board_detection.cnns import EmptyNetAttention


@dataclass
class CNNHyperparams:
    """Architecture hyperparameters for EmptyNetAttention."""
    pool_size: int = 4              # N in AdaptiveAvgPool2d((N, N))
    embedding_dim: int = 64         # Hidden layer size
    attention_temp: float = 1.0     # Spatial attention temperature
    dropout1: float = 0.4           # Dropout after pooling
    dropout2: float = 0.3           # Dropout before final layer
    
    def to_dict(self):
        return asdict(self)


def train_cnn_white(
    data_dir: Path,
    cache_path: Path = None,
    force_reload: bool = False,
    colorspace: str = "lab",
    # Training hyperparams
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    random_state: int = 42,
    test_size: float = 0.2,
    patch_jitter: bool = True,
    weighted_sampling: bool = False,
    # Architecture hyperparams
    cnn_params: CNNHyperparams = None,
    device: str = None
):
    """
    Train CNN for white patch detection with MLflow logging.
    """
    if cnn_params is None:
        cnn_params = CNNHyperparams()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
    
    # Setup MLflow
    setup_mlflow()
    experiment_id = get_or_create_experiment("cnn_white")
    
    # Load data
    if cache_path:
        dataset = BoardPatchDataset.load_or_create(
            cache_path=cache_path,
            data_dir=data_dir,
            skip_bad=True,
            force_reload=force_reload
        )
    else:
        dataset = BoardPatchDataset()
        dataset.load_from_directory(data_dir, skip_bad=True)
    
    print(f"Loaded {dataset}")
    
    # Get all patches with file grouping
    all_patches = dataset.get_all_patches()
    file_groups = [p.file_path for p in all_patches]
    
    # Split by image
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.arange(len(all_patches)), groups=file_groups))
    
    train_patches = [all_patches[i] for i in train_idx]
    test_patches = [all_patches[i] for i in test_idx]
    
    print(f"Train: {len(train_patches)}, Test: {len(test_patches)}")
    
    # Count white patches
    n_white_train = sum(1 for p in train_patches if p.label_raw == "w")
    n_white_test = sum(1 for p in test_patches if p.label_raw == "w")
    print(f"White patches - Train: {n_white_train}, Test: {n_white_test}")
    
    # Create augmentation transforms
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ])
    
    # Create datasets
    in_channels = 1 if colorspace == "gray" else 3
    train_ds = TorchPatchDataset(
        train_patches, dataset,
        colorspace=colorspace,
        label_type="white",
        transform=train_transform,
        patch_jitter=patch_jitter
    )
    test_ds = TorchPatchDataset(
        test_patches, dataset,
        colorspace=colorspace,
        label_type="white",
        transform=None,
        patch_jitter=False
    )
    
    # Weighted Sampler logic
    train_sampler = None
    shuffle_train = True
    
    if weighted_sampling:
        print("Using WeightedRandomSampler for class balance...")
        # Calculate weights based on TRAINING set only
        n_train = len(train_patches)
        n_neg = n_train - n_white_train
        
        weight_white = 1.0 / n_white_train
        weight_neg = 1.0 / n_neg
        
        # Assign weight to each sample
        sample_weights = []
        for p in train_patches:
            if p.label_raw == "w":
                sample_weights.append(weight_white)
            else:
                sample_weights.append(weight_neg)
        
        sample_weights = torch.DoubleTensor(sample_weights)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_train,
            replacement=True
        )
        shuffle_train = False # Mutually exclusive with sampler
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle_train, 
        sampler=train_sampler,
        num_workers=0
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model with explicit hyperparameters
    model = EmptyNetAttention(
        in_channels=in_channels,
        N=cnn_params.pool_size,
        embedding_dim=cnn_params.embedding_dim,
        attention_temp=cnn_params.attention_temp
    ).to(device)
    
    # Count parameters
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params_total:,} total, {n_params_trainable:,} trainable")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    run_name = f"cnn_white_{colorspace}_N{cnn_params.pool_size}_emb{cnn_params.embedding_dim}"
    if weighted_sampling:
        run_name += "_balanced"
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Log all parameters
        mlflow.log_param("model_type", "cnn_white")
        mlflow.log_param("architecture", "EmptyNetAttention")
        mlflow.log_param("n_params_total", n_params_total)
        mlflow.log_param("n_params_trainable", n_params_trainable)
        
        # Training params
        mlflow.log_param("colorspace", colorspace)
        mlflow.log_param("in_channels", in_channels)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("patch_jitter", patch_jitter)
        mlflow.log_param("weighted_sampling", weighted_sampling)
        
        # Architecture params
        mlflow.log_param("pool_size", cnn_params.pool_size)
        mlflow.log_param("embedding_dim", cnn_params.embedding_dim)
        mlflow.log_param("attention_temp", cnn_params.attention_temp)
        mlflow.log_param("dropout1", cnn_params.dropout1)
        mlflow.log_param("dropout2", cnn_params.dropout2)
        
        # Data params
        mlflow.log_param("n_train", len(train_patches))
        mlflow.log_param("n_test", len(test_patches))
        mlflow.log_param("n_white_train", n_white_train)
        mlflow.log_param("n_white_test", n_white_test)
        mlflow.log_param("device", device)
        
        best_val_f1 = 0
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = model(X).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(y.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X).squeeze(-1)
                    loss = criterion(logits, y)
                    
                    val_loss += loss.item()
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(y.cpu().numpy())
            
            val_loss /= len(test_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
            
            # Log metrics per epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            
            # Track best model
            if val_f1 >= best_val_f1 and val_loss<best_val_loss:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            
            if epoch % 1 == 0 or epoch == epochs - 1: #lets print everything
                print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f},val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")
        
        # Final metrics (best model)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("final_val_acc", val_acc)
        mlflow.log_metric("final_val_precision", val_precision)
        mlflow.log_metric("final_val_recall", val_recall)
        mlflow.log_metric("final_val_f1", val_f1)
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "labels": ["non-white", "white"],
        }
        mlflow.log_text(json.dumps(cm_dict, indent=2), "metrics/confusion_matrix.json")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-white", "white"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"White Detection (val_f1={val_f1:.4f})")
        plt.tight_layout()
        cm_path = Path("/tmp/confusion_matrix_cnn_white.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()
        mlflow.log_artifact(str(cm_path), "metrics")
        
        # Save best model
        model.load_state_dict(best_model_state)
        print(f"\nSaving best model (epoch {best_epoch}, val_f1={best_val_f1:.4f})")
        mlflow.pytorch.log_model(model, "model")
        
        # Log dataset hashes
        mlflow.log_text(dataset.get_image_hash_summary(), "dataset/image_hashes.txt")
        
        # Log architecture params as JSON artifact
        mlflow.log_text(json.dumps(cnn_params.to_dict(), indent=2), "config/cnn_params.json")
        
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Best epoch: {best_epoch}")
        print(f"Best val F1: {best_val_f1:.4f}")
        print(f"Final val acc: {val_acc:.4f}")
        print(f"Run ID: {run_id}")
        
        return model, run_id


def main():
    parser = argparse.ArgumentParser(description="Train CNN for white detection")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/my_photos/"),
        help="Directory with annotated images"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/dataset_cache.pkl"),
        help="Path to cache dataset"
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    
    # Training hyperparams
    parser.add_argument("--colorspace", choices=["lab", "gray", "rgb"], default="lab")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-jitter", action="store_true", help="Disable patch jitter")
    parser.add_argument("--weighted-sampling", action="store_true", help="Use WeightedRandomSampler for class balance")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    
    # Architecture hyperparams
    parser.add_argument("--pool-size", type=int, default=4, help="AdaptiveAvgPool size N")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--attention-temp", type=float, default=1.0, help="Attention temperature")
    parser.add_argument("--dropout1", type=float, default=0.4, help="Dropout after pooling")
    parser.add_argument("--dropout2", type=float, default=0.3, help="Dropout before output")
    
    args = parser.parse_args()
    
    cache_path = None if args.no_cache else args.cache_path
    
    cnn_params = CNNHyperparams(
        pool_size=args.pool_size,
        embedding_dim=args.embedding_dim,
        attention_temp=args.attention_temp,
        dropout1=args.dropout1,
        dropout2=args.dropout2
    )
    
    train_cnn_white(
        data_dir=args.data_dir,
        cache_path=cache_path,
        force_reload=args.force_reload,
        colorspace=args.colorspace,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        random_state=args.random_state,
        test_size=args.test_size,
        patch_jitter=not args.no_jitter,
        weighted_sampling=args.weighted_sampling,
        cnn_params=cnn_params,
        device=args.device
    )


if __name__ == "__main__":
    main()
