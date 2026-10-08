#!/usr/bin/env python3
"""Evaluation Script for GCP Localizer and Classifier Models."""

import argparse
from glob import glob
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.data.datasets import FastGCPDataset, MarkerClassifierDataset
from gcp_localizer.models.localizer import GCPLocalizer
from gcp_localizer.models.classifier import GCPClassifier
from gcp_localizer.evaluation.metrics import evaluate_shape_maes
from gcp_localizer.evaluation.visualization import (
    plot_predictions,
    plot_shape_confusion,
    plot_dataset_distribution,
    plot_end_to_end_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GCP Localizer and Classifier models.")
    parser.add_argument(
        "--localizer-config",
        type=str,
        default="configs/localizer.yaml",
        help="Path to localizer YAML config."
    )
    parser.add_argument(
        "--classifier-config",
        type=str,
        default="configs/classifier.yaml",
        help="Path to classifier YAML config."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Optional override for processed data directory."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load localizer config
    loc_config_path = Path(args.localizer_config)
    loc_cfg = {}
    if loc_config_path.exists():
        with open(loc_config_path, "r") as f:
            loc_cfg = yaml.safe_load(f)

    # Load classifier config
    cls_config_path = Path(args.classifier_config)
    cls_cfg = {}
    if cls_config_path.exists():
        with open(cls_config_path, "r") as f:
            cls_cfg = yaml.safe_load(f)

    processed_dir_str = (
        args.processed_dir
        or loc_cfg.get("data", {}).get("processed_dir")
        or cls_cfg.get("data", {}).get("processed_dir")
        or "data/processed"
    )
    processed_dir = Path(processed_dir_str)

    # 1. Dataset Distribution Visualization
    if processed_dir.exists():
        dist_plot_path = Path("artifacts/plots/dataset_distribution.png")
        print(f"\nGenerating dataset class & shape distribution plot at {dist_plot_path}...")
        plot_dataset_distribution(processed_dir, save_path=dist_plot_path)

    localizer_model = None
    classifier_model = None
    sample_val_file = None

    # 2. Evaluate Localizer
    if loc_cfg:
        val_pos_files = sorted(glob(str(processed_dir / "val" / "positive" / "*.npz")))
        model_path = Path(loc_cfg.get("training", {}).get("save_path", "artifacts/models/best_localizer_model.pt"))

        if val_pos_files and model_path.exists():
            print(f"\nEvaluating Localizer model from {model_path}...")
            localizer_model = GCPLocalizer(pretrained=False)
            localizer_model.load_state_dict(torch.load(model_path, map_location=device))
            sample_val_file = val_pos_files[0]

            maes_res = evaluate_shape_maes(localizer_model, val_pos_files, device=device)
            print("--- Localizer MAE Metrics per Shape ---")
            for shape, info in maes_res.items():
                if shape != "overall":
                    print(f"Shape: {shape:10s} | MAE: {info['mae']:.2f} px | Samples: {info['count']}")

            print(f"Overall Validation MAE: {maes_res['overall']['mae']:.2f} px (Median: {maes_res['overall']['median']:.2f} px)")

            # Save localizer predictions plot
            plot_path = Path("artifacts/plots/localizer_predictions.png")
            val_ds = FastGCPDataset(val_pos_files)
            plot_predictions(localizer_model, val_ds, save_path=plot_path, device=device)
            print(f"Saved localizer prediction visualization to {plot_path}")
        else:
            print(f"Localizer model at {model_path} or validation files not found. Skipping localizer eval.")

    # 3. Evaluate Classifier
    if cls_cfg:
        val_pos = glob(str(processed_dir / "val" / "positive" / "*.npz"))
        val_neg = glob(str(processed_dir / "val" / "negative" / "*.npz"))
        val_files = sorted(val_pos + val_neg)
        model_path = Path(cls_cfg.get("training", {}).get("save_path", "artifacts/models/best_classifier_model.pt"))

        if val_files and model_path.exists():
            print(f"\nEvaluating Classifier model from {model_path}...")
            classifier_model = GCPClassifier(num_classes=2, pretrained=False)
            classifier_model.load_state_dict(torch.load(model_path, map_location=device))

            val_ds = MarkerClassifierDataset(val_files)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            plot_path = Path("artifacts/plots/classifier_confusion.png")
            plot_shape_confusion(classifier_model, val_loader, save_path=plot_path, device=device)
            print(f"Saved classifier confusion visualization to {plot_path}")
        else:
            print(f"Classifier model at {model_path} or validation files not found. Skipping classifier eval.")

    # 4. End-to-End Pipeline Visualization
    if localizer_model is not None and classifier_model is not None and sample_val_file:
        pipeline_plot_path = Path("artifacts/plots/end_to_end_pipeline.png")
        print(f"\nGenerating end-to-end pipeline flow visualization at {pipeline_plot_path}...")
        plot_end_to_end_pipeline(
            sample_npz_path=sample_val_file,
            classifier_model=classifier_model,
            localizer_model=localizer_model,
            save_path=pipeline_plot_path,
            device=device
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
