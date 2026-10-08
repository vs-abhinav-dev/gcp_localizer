from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from gcp_localizer.data.datasets import preprocess_image_tensor


def compute_localization_error(
    pred_coords: Union[np.ndarray, Tuple[float, float]],
    gt_coords: Union[np.ndarray, Tuple[float, float]]
) -> float:
    """Calculate Euclidean pixel error between predicted and ground-truth coordinates.

    Args:
        pred_coords: (x, y) predicted coordinates.
        gt_coords: (x, y) ground truth coordinates.

    Returns:
        Euclidean error in pixels.
    """
    pred = np.array(pred_coords, dtype=np.float32)
    gt = np.array(gt_coords, dtype=np.float32)
    return float(np.sqrt(np.sum((pred - gt) ** 2)))


def evaluate_shape_maes(
    localizer_model: nn.Module,
    val_files: List[Union[str, Path]],
    device: str = "cpu"
) -> Dict[str, Any]:
    """Calculate Mean Absolute Error (MAE) per shape type on validation set files.

    Args:
        localizer_model: Trained localizer model.
        val_files: List of paths to preprocessed validation `.npz` crop files.
        device: 'cuda' or 'cpu'.

    Returns:
        Dictionary mapping shape name to MAE info dict (mean, count, errors list).
    """
    localizer_model.eval()
    localizer_model.to(device)

    shape_errors = defaultdict(list)
    all_errors = []

    with torch.no_grad():
        for file_path in val_files:
            data = np.load(file_path, allow_pickle=True)
            shape = str(data["shape"]) if "shape" in data.files else "Unknown"
            image = data["image"]
            target = data["target"]

            image_tensor = preprocess_image_tensor(image).unsqueeze(0).to(device)
            pred = localizer_model(image_tensor).cpu().numpy()[0]

            error = compute_localization_error(pred, target)
            shape_errors[shape].append(error)
            all_errors.append(error)

    results = {}
    for shape, errors in shape_errors.items():
        results[shape] = {
            "mae": float(np.mean(errors)),
            "count": len(errors),
            "errors": errors
        }

    results["overall"] = {
        "mae": float(np.mean(all_errors)),
        "median": float(np.median(all_errors)),
        "count": len(all_errors)
    }

    return results
