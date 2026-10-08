"""Reusable visualization utilities for GCP predictions."""

from pathlib import Path
from typing import Union, Optional, Tuple
import cv2
import numpy as np


def annotate_prediction(
    image: Union[np.ndarray, str, Path],
    x: float,
    y: float,
    shape: str,
    confidence: float,
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Generate an annotated image highlighting the predicted GCP marker and attributes.

    Includes:
    - Reticle / target circle at predicted GCP center (x, y)
    - Formatted callout card with coordinate text, predicted shape, and confidence

    Args:
        image: Original RGB image array (H, W, 3), or file path to image.
        x: Predicted X coordinate in original image space.
        y: Predicted Y coordinate in original image space.
        shape: Predicted GCP shape (e.g., 'Cross', 'Square', 'L-Shape').
        confidence: Prediction confidence score in [0, 1].
        save_path: Optional path to save the output annotated image.

    Returns:
        Annotated RGB numpy image array.
    """
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    h, w = img.shape[:2]
    cx = int(round(x))
    cy = int(round(y))

    # Dynamic scaling based on image resolution
    scale_factor = max(0.5, min(w, h) / 900.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, 0.6 * scale_factor)
    font_thickness = max(1, int(1.8 * scale_factor))
    line_spacing = int(24 * scale_factor)

    # Color scheme for shapes (RGB)
    shape_colors = {
        "Cross": (0, 230, 255),      # Cyan
        "Square": (255, 180, 0),     # Amber / Gold
        "L-Shape": (0, 255, 128),    # Neon Green
    }
    marker_color = shape_colors.get(shape, (255, 60, 60))

    # 1. Draw Reticle at Center
    radius = max(10, int(18 * scale_factor))
    cross_arm = max(16, int(28 * scale_factor))

    # Reticle circle
    cv2.circle(img, (cx, cy), radius, marker_color, max(2, int(2.5 * scale_factor)), cv2.LINE_AA)
    # Inner dot
    cv2.circle(img, (cx, cy), max(2, int(3 * scale_factor)), (255, 255, 255), -1, cv2.LINE_AA)
    # Crosshair tick marks
    cv2.line(img, (cx - cross_arm, cy), (cx - radius - 2, cy), marker_color, max(1, int(2 * scale_factor)), cv2.LINE_AA)
    cv2.line(img, (cx + radius + 2, cy), (cx + cross_arm, cy), marker_color, max(1, int(2 * scale_factor)), cv2.LINE_AA)
    cv2.line(img, (cx, cy - cross_arm), (cx, cy - radius - 2), marker_color, max(1, int(2 * scale_factor)), cv2.LINE_AA)
    cv2.line(img, (cx, cy + radius + 2), (cx, cy + cross_arm), marker_color, max(1, int(2 * scale_factor)), cv2.LINE_AA)

    # 2. Prepare Callout Card Text
    conf_pct = confidence * 100.0 if confidence <= 1.0 else confidence
    lines = [
        "GCP",
        f"x: {x:.1f}",
        f"y: {y:.1f}",
        f"Shape: {shape}",
        f"Confidence: {conf_pct:.1f}%",
    ]

    # Measure text bounding box
    text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines]
    max_text_w = max(s[0] for s in text_sizes)
    total_text_h = len(lines) * line_spacing

    pad = int(12 * scale_factor)
    box_w = max_text_w + 2 * pad
    box_h = total_text_h + pad

    # 3. Position the Callout Box (keep within image boundaries)
    offset_x = int(radius + 15 * scale_factor)
    offset_y = -int(box_h / 2)

    box_x1 = cx + offset_x
    box_y1 = cy + offset_y

    # Flip box to left if near right edge
    if box_x1 + box_w > w - 10:
        box_x1 = cx - offset_x - box_w

    # Adjust vertical if outside bounds
    if box_y1 < 10:
        box_y1 = 10
    elif box_y1 + box_h > h - 10:
        box_y1 = max(10, h - box_h - 10)

    box_x2 = min(w - 5, box_x1 + box_w)
    box_y2 = min(h - 5, box_y1 + box_h)
    box_x1 = max(5, box_x1)
    box_y1 = max(5, box_y1)

    # 4. Draw Connecting Leader Line from Reticle to Card
    leader_start = (cx + radius if box_x1 > cx else cx - radius, cy)
    leader_end = (box_x1 if box_x1 > cx else box_x2, int((box_y1 + box_y2) / 2))
    cv2.line(img, leader_start, leader_end, (200, 200, 200), max(1, int(1.5 * scale_factor)), cv2.LINE_AA)

    # 5. Draw Dark Translucent Card Background
    card_roi = img[box_y1:box_y2, box_x1:box_x2]
    dark_overlay = np.zeros_like(card_roi, dtype=np.uint8)
    dark_overlay[:] = (18, 24, 38)  # Slate dark navy
    cv2.addWeighted(dark_overlay, 0.85, card_roi, 0.15, 0, card_roi)
    img[box_y1:box_y2, box_x1:box_x2] = card_roi

    # Card border
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), marker_color, max(1, int(1.5 * scale_factor)), cv2.LINE_AA)

    # 6. Render Text Inside Card
    text_y = box_y1 + int(20 * scale_factor)
    text_x = box_x1 + pad

    for idx, line in enumerate(lines):
        line_color = marker_color if idx in (0, 3) else (240, 240, 240)
        cv2.putText(
            img,
            line,
            (text_x, text_y),
            font,
            font_scale,
            line_color,
            font_thickness,
            cv2.LINE_AA,
        )
        text_y += line_spacing

    # 7. Optionally Save Image
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)

    return img
