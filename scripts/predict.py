#!/usr/bin/env python3
"""CLI Inference Script for GCP Detection, Localization, and Shape Classification."""

import argparse
import json
from pathlib import Path
import sys

# Add src to python path if executing script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.inference.predictor import GCPPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict GCP marker location and shape on an aerial image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to aerial image file (JPG, PNG, etc.)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated image (optional)."
    )
    parser.add_argument(
        "--localizer-weights",
        type=str,
        default=None,
        help="Path to localizer model checkpoint (optional)."
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Path to shape classifier model checkpoint (optional)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cpu' or 'cuda')."
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    predictor = GCPPredictor(
        localizer_model=args.localizer_weights,
        shape_classifier_model=args.classifier_weights,
        device=args.device,
    )

    if args.output:
        prediction, annotated_img = predictor.predict_and_annotate(image_path, save_path=args.output)
        print(f"Annotated image saved to: {args.output}")
    else:
        prediction = predictor.predict(image_path)

    # Print formatted JSON output
    print(json.dumps(prediction, indent=4))


if __name__ == "__main__":
    main()
