#!/usr/bin/env python3
"""Training Script for GCP Shape Classifier Model (3 classes: L-Shape, Square, Cross)."""

import argparse
from glob import glob
from pathlib import Path
import sys
from typing import Tuple
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add src to python path if executing script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.data.annotations import load_annotations
from gcp_localizer.data.splits import split_annotations
from gcp_localizer.data.datasets import ShapeDataset, RawShapeDataset
from gcp_localizer.models.classifier import GCPShapeClassifier, GCPClassifier
from gcp_localizer.training.classifier import train_classifier


def create_synthetic_shape_dataset(num_samples: int = 60) -> Tuple[DataLoader, DataLoader]:
    """Create in-memory synthetic dataset for smoke tests and verification."""
    from torch.utils.data import TensorDataset
    torch.manual_seed(42)
    # 3 balanced classes: 0, 1, 2
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.tensor([i % 3 for i in range(num_samples)], dtype=torch.long)

    split = int(0.8 * num_samples)
    train_ds = TensorDataset(images[:split], labels[:split])
    val_ds = TensorDataset(images[split:], labels[split:])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train GCP Shape Classifier (L-Shape, Square, Cross).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/classifier.yaml",
        help="Path to classifier or base YAML config."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw image dataset directory or processed dataset directory."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to annotations JSON file."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)."
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Crop size for marker tiles (e.g. 256 for tight zoom, 512 for wide)."
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=None,
        help="Max random offset jitter for training crops (e.g. 16 for centered micro-jitter)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability in classifier head (e.g. 0.3)."
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden projection dimension for MLP head (e.g. 256)."
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Label smoothing factor for cross-entropy (e.g. 0.05)."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Output checkpoint file path."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run quick training on synthetic data for pipeline validation."
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Extract configs supporting both configs/base.yaml and configs/classifier.yaml formats
    data_cfg = config.get("data", {})
    if "classifier" in config.get("model", {}):
        model_cfg = config["model"]["classifier"]
    else:
        model_cfg = config.get("model", {})

    if "classifier" in config.get("training", {}):
        train_cfg = config["training"]["classifier"]
    else:
        train_cfg = config.get("training", {})

    # Determine hyperparameters
    num_classes = model_cfg.get("num_classes", 3)
    pretrained = model_cfg.get("pretrained", True)
    hidden_dim = args.hidden_dim or model_cfg.get("hidden_dim", 256)
    dropout = args.dropout if args.dropout is not None else model_cfg.get("dropout", 0.3)
    num_epochs = args.epochs or train_cfg.get("num_epochs", 20)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    num_workers = train_cfg.get("num_workers", 2)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else train_cfg.get("label_smoothing", 0.05)
    class_weights = train_cfg.get("class_weights", "balanced")
    scheduler_type = train_cfg.get("scheduler", "cosine")
    crop_size = args.crop_size or data_cfg.get("crop_size", 256)
    max_offset = args.max_offset if args.max_offset is not None else data_cfg.get("max_offset", 16)
    save_path = args.save_path or train_cfg.get("save_path", "models/classifier.pt")

    print("==================================================")
    print(" GCP Shape Classifier Training (Enhanced)")
    print("==================================================")
    print(f"Config: {config_path}")
    print(f"Num Classes: {num_classes} (0: L-Shape, 1: Square, 2: Cross)")
    print(f"Architecture: ResNet18 + MLP Head (hidden={hidden_dim}, dropout={dropout})")
    print(f"Crop Size: {crop_size} px | Max Offset Jitter: {max_offset} px")
    print(f"Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"Loss: CrossEntropy (weights={class_weights}, smoothing={label_smoothing}) | Scheduler: {scheduler_type}")
    print(f"Target Checkpoint: {save_path}")

    # Case 1: Synthetic flag passed
    if args.synthetic:
        print("[Mode: Synthetic Data Validation]")
        train_loader, val_loader = create_synthetic_shape_dataset()
    else:
        # Resolve data paths
        raw_images_dir = Path(args.data or data_cfg.get("raw_images_dir", "data/raw/train_dataset"))
        annotation_file = Path(args.annotations or data_cfg.get("annotation_file", "data/raw/gcp_marks.json"))
        processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))

        train_loader = None
        val_loader = None

        # Check for preprocessed .npz positive files
        train_pos_files = sorted(glob(str(processed_dir / "train" / "positive" / "*.npz")))
        val_pos_files = sorted(glob(str(processed_dir / "val" / "positive" / "*.npz")))

        if train_pos_files and val_pos_files:
            print(f"Found preprocessed crops: {len(train_pos_files)} train, {len(val_pos_files)} val.")
            train_ds = ShapeDataset(train_pos_files, is_train=True)
            val_ds = ShapeDataset(val_pos_files, is_train=False)

            if len(train_ds) > 0:
                print(f"Loaded ShapeDataset with augmentations: {len(train_ds)} train shapes, {len(val_ds)} val shapes.")
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Check for raw images + annotation file
        if train_loader is None and annotation_file.exists() and raw_images_dir.exists():
            print(f"Loading raw annotations from {annotation_file}...")
            annotations = load_annotations(annotation_file)
            val_ratio = data_cfg.get("val_ratio", 0.15)
            seed = data_cfg.get("seed", 42)
            train_anns, val_anns = split_annotations(annotations, val_ratio=val_ratio, seed=seed)

            train_ds = RawShapeDataset(
                train_anns,
                raw_images_dir,
                crop_size=crop_size,
                output_size=data_cfg.get("output_size", 224),
                is_train=True,
                max_offset=max_offset,
            )
            val_ds = RawShapeDataset(
                val_anns,
                raw_images_dir,
                crop_size=crop_size,
                output_size=data_cfg.get("output_size", 224),
                is_train=False,
                max_offset=0,
            )

            if len(train_ds) > 0:
                print(f"Loaded RawShapeDataset with augmentations (crop={crop_size}px, jitter={max_offset}px): {len(train_ds)} train, {len(val_ds)} val.")
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Fallback if no dataset found
        if train_loader is None:
            print(f"\n[Warning] No real dataset found at {processed_dir} or {raw_images_dir}.")
            print("To train on real data, please provide:")
            print("  1. Raw images and annotations: --data <images_dir> --annotations <gcp_marks.json>")
            print("  2. Or processed crops in: data/processed/train/positive/*.npz")
            print("Running in verification synthetic mode (--synthetic)...")
            train_loader, val_loader = create_synthetic_shape_dataset()

    # Instantiate Shape Classifier Model with MLP projection head and dropout
    model = GCPShapeClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
    )

    # Train Model with balanced class weighting and learning rate scheduling
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        scheduler_type=scheduler_type,
        save_path=save_path,
    )

    # Ensure models/ and artifacts/models/ have the checkpoint
    save_file = Path(save_path)
    if save_file.exists():
        # Also copy to artifacts/models/best_shape_model.pt if saved to models/classifier.pt
        alt_save = Path("artifacts/models/best_shape_model.pt")
        if save_file.resolve() != alt_save.resolve():
            alt_save.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(save_file, alt_save)
            print(f"Synchronized checkpoint to {alt_save}")

    print("\n==================================================")
    print(" Training Summary")
    print("==================================================")
    print(f"Best Validation Accuracy: {history['best_val_acc'] * 100:.2f}%")
    print(f"Best Validation F1 Score: {history['best_val_f1']:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
