#!/usr/bin/env python3
"""Data Preparation Script for GCP Localizer.

Splits original images into train and validation sets (data-leakage free)
and generates positive/negative crop tiles saved as npz files.
"""

import argparse
from pathlib import Path
import sys
import yaml

# Add src to python path if executing script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.data.annotations import load_annotations
from gcp_localizer.data.splits import split_annotations
from gcp_localizer.data.preprocessing import generate_crops_for_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset splits and crop tiles.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/localizer.yaml",
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    annotation_file = Path(data_cfg["annotation_file"])
    raw_images_dir = Path(data_cfg["raw_images_dir"])
    processed_dir = Path(data_cfg["processed_dir"])

    if not annotation_file.exists():
        print(f"Annotation file {annotation_file} does not exist. Skipping data preparation.")
        print("Place raw annotations at data/raw/gcp_marks.json and images in data/raw/train_dataset.")
        return

    print(f"Loading annotations from {annotation_file}...")
    annotations = load_annotations(annotation_file)

    val_ratio = data_cfg.get("val_ratio", 0.15)
    seed = data_cfg.get("seed", 42)

    print(f"Splitting {len(annotations)} original images (val_ratio={val_ratio}, seed={seed})...")
    train_anns, val_anns = split_annotations(annotations, val_ratio=val_ratio, seed=seed)
    print(f"Train images: {len(train_anns)} | Validation images: {len(val_anns)}")

    # Generate Train crops
    print("Generating train split crop tiles...")
    train_pos, train_neg = generate_crops_for_dataset(
        annotations=train_anns,
        raw_images_dir=raw_images_dir,
        output_dir=processed_dir,
        split_name="train",
        crop_size=data_cfg.get("crop_size", 512),
        output_size=data_cfg.get("output_size", 224),
        num_pos_aug=data_cfg.get("num_pos_aug", 5),
        max_offset=data_cfg.get("max_offset", 128),
        num_neg_samples=data_cfg.get("num_neg_samples", 5),
        seed=seed
    )

    # Generate Val crops
    print("Generating validation split crop tiles...")
    val_pos, val_neg = generate_crops_for_dataset(
        annotations=val_anns,
        raw_images_dir=raw_images_dir,
        output_dir=processed_dir,
        split_name="val",
        crop_size=data_cfg.get("crop_size", 512),
        output_size=data_cfg.get("output_size", 224),
        num_pos_aug=1,  # 1 crop per val image for deterministic evaluation
        max_offset=0,   # No offset jittering for val evaluation
        num_neg_samples=1,
        seed=seed
    )

    print("Data preparation complete!")
    print(f"Train dataset: {train_pos} positive tiles, {train_neg} negative tiles.")
    print(f"Val dataset: {val_pos} positive tiles, {val_neg} negative tiles.")


if __name__ == "__main__":
    main()
