#!/usr/bin/env python3
"""Training Script for GCP Localizer Model."""

import argparse
from glob import glob
from pathlib import Path
import sys
import yaml

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.data.datasets import FastGCPDataset
from gcp_localizer.models.localizer import GCPLocalizer
from gcp_localizer.training.localizer import train_localizer


def main():
    parser = argparse.ArgumentParser(description="Train GCP Localizer model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/localizer.yaml",
        help="Path to localizer YAML config."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    processed_dir = Path(data_cfg["processed_dir"])
    train_files = sorted(glob(str(processed_dir / "train" / "positive" / "*.npz")))
    val_files = sorted(glob(str(processed_dir / "val" / "positive" / "*.npz")))

    if not train_files:
        print("No processed training data found! Run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    print(f"Loaded {len(train_files)} training files and {len(val_files)} validation files for localizer.")

    train_ds = FastGCPDataset(train_files)
    val_ds = FastGCPDataset(val_files)

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = GCPLocalizer(
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2)
    )

    print("Starting localizer training...")
    history = train_localizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_cfg.get("num_epochs", 30),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        save_path=train_cfg.get("save_path", "artifacts/models/best_localizer_model.pt")
    )

    print(f"Training completed! Best Validation Loss (MAE): {history['best_val_loss']:.4f} px")


if __name__ == "__main__":
    main()
