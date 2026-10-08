import random
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split


def split_annotations(
    annotations: Dict[str, Any],
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministically split original image annotations into train and validation sets.

    This ensures that crops derived from the same original image do not appear in both
    training and validation splits, preventing data leakage.

    Args:
        annotations: Dictionary mapping image relative path to annotation info.
        val_ratio: Fraction of original images to allocate to validation.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_annotations, val_annotations)
    """
    image_paths = sorted(list(annotations.keys()))
    if not image_paths:
        return {}, {}

    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=val_ratio,
        random_state=seed
    )

    train_annotations = {p: annotations[p] for p in train_paths}
    val_annotations = {p: annotations[p] for p in val_paths}

    return train_annotations, val_annotations
