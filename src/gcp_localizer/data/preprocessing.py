import os
import random
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm


def crop_positive_sample(
    image: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int = 512,
    output_size: int = 224,
    max_offset: int = 128
) -> Dict[str, Any]:
    """Crop a positive tile centered around (cx, cy) with random offset.

    Args:
        image: Original RGB image.
        cx: Ground truth X coordinate.
        cy: Ground truth Y coordinate.
        crop_size: Unscaled crop tile dimension.
        output_size: Output model input tile size.
        max_offset: Maximum random pixel offset from GCP center.

    Returns:
        Dictionary containing cropped image tile (output_size x output_size),
        relative target coordinates [rel_x, rel_y], and offset.
    """
    h, w = image.shape[:2]
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)

    crop_center_x = cx + dx
    crop_center_y = cy + dy
    half = crop_size // 2

    x1 = int(crop_center_x - half)
    y1 = int(crop_center_y - half)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT
    )

    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    crop = padded[y1:y2, x1:x2]

    rel_x = cx - (crop_center_x - half)
    rel_y = cy - (crop_center_y - half)

    scale = output_size / crop_size
    rel_x *= scale
    rel_y *= scale

    crop = cv2.resize(crop, (output_size, output_size))

    return {
        "image": crop,
        "target": np.array([rel_x, rel_y], dtype=np.float32),
        "offset": np.array([dx, dy], dtype=np.int32)
    }


def crop_negative_sample(
    image: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int = 512,
    output_size: int = 224,
    min_distance: float = 400.0,
    max_attempts: int = 100
) -> Optional[np.ndarray]:
    """Crop a negative tile from image guaranteed to be > min_distance away from GCP mark.

    Args:
        image: Original RGB image.
        cx: Ground truth X coordinate of GCP.
        cy: Ground truth Y coordinate of GCP.
        crop_size: Crop size.
        output_size: Output tile size.
        min_distance: Minimum Euclidean distance required from GCP center.
        max_attempts: Safety limit for random attempts.

    Returns:
        Resized RGB negative crop tile or None if image too small.
    """
    h, w = image.shape[:2]
    half = crop_size // 2
    if w <= crop_size or h <= crop_size:
        return None

    for _ in range(max_attempts):
        rx = random.randint(half, w - half)
        ry = random.randint(half, h - half)

        dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
        if dist > min_distance:
            crop = image[ry - half : ry + half, rx - half : rx + half]
            crop = cv2.resize(crop, (output_size, output_size))
            return crop

    return None


def generate_crops_for_dataset(
    annotations: Dict[str, Any],
    raw_images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    split_name: str = "train",
    crop_size: int = 512,
    output_size: int = 224,
    num_pos_aug: int = 5,
    max_offset: int = 128,
    num_neg_samples: int = 5,
    seed: int = 42
) -> Tuple[int, int]:
    """Process original images and save positive and negative compressed npz crop tiles.

    Args:
        annotations: Dictionary of annotations for this split.
        raw_images_dir: Directory containing original uncropped images.
        output_dir: Directory to save generated npz files.
        split_name: 'train' or 'val'.
        crop_size: Unscaled crop size.
        output_size: Resized model input tile size.
        num_pos_aug: Number of positive crops per image.
        max_offset: Random offset range for positive crops.
        num_neg_samples: Number of negative crops per image.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (total_pos_crops, total_neg_crops).
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_dir = Path(raw_images_dir)
    pos_out_dir = Path(output_dir) / split_name / "positive"
    neg_out_dir = Path(output_dir) / split_name / "negative"
    pos_out_dir.mkdir(parents=True, exist_ok=True)
    neg_out_dir.mkdir(parents=True, exist_ok=True)

    pos_idx = 0
    neg_idx = 0

    for rel_path, info in tqdm(annotations.items(), desc=f"Processing {split_name} split"):
        if "mark" not in info or info["mark"] is None:
            continue

        cx = float(info["mark"]["x"])
        cy = float(info["mark"]["y"])
        shape = info.get("verified_shape", "Unknown")

        img_path = raw_dir / rel_path
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate Positive Crops
        for _ in range(num_pos_aug):
            crop_res = crop_positive_sample(
                image, cx, cy,
                crop_size=crop_size,
                output_size=output_size,
                max_offset=max_offset
            )
            save_path = pos_out_dir / f"pos_{pos_idx:06d}.npz"
            np.savez_compressed(
                save_path,
                image=crop_res["image"],
                target=crop_res["target"],
                path=rel_path,
                original_center=np.array([cx, cy], dtype=np.float32),
                offset=crop_res["offset"],
                shape=shape,
                label=1
            )
            pos_idx += 1

        # Generate Negative Crops
        for _ in range(num_neg_samples):
            neg_crop = crop_negative_sample(
                image, cx, cy,
                crop_size=crop_size,
                output_size=output_size
            )
            if neg_crop is not None:
                save_path = neg_out_dir / f"neg_{neg_idx:06d}.npz"
                np.savez_compressed(
                    save_path,
                    image=neg_crop,
                    path=rel_path,
                    shape="Non-Marker",
                    label=0
                )
                neg_idx += 1

    return pos_idx, neg_idx
