#!/usr/bin/env python3
"""Unified training CLI for GCP Localizer and Shape Classifier."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Unified GCP Model Training Runner.")
    parser.add_argument(
        "--component",
        type=str,
        choices=["localizer", "classifier", "all"],
        default="classifier",
        help="Model component to train ('localizer', 'classifier', or 'all')."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for smoke test / validation."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.component in ("classifier", "all"):
        print("\n>>> Launching Shape Classifier Training...")
        cmd = [sys.executable, str(project_root / "scripts" / "train_classifier.py"), "--config", args.config]
        if args.synthetic:
            cmd.append("--synthetic")
        res = subprocess.run(cmd)
        if res.returncode != 0 and args.component == "all":
            sys.exit(res.returncode)

    if args.component in ("localizer", "all"):
        print("\n>>> Launching Localizer Training...")
        cmd = [sys.executable, str(project_root / "scripts" / "train_localizer.py"), "--config", args.config]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
