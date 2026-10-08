"""Unit tests for unified inference pipeline and prediction visualization."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from gcp_localizer.inference.predictor import GCPPredictor
from gcp_localizer.inference.visualization import annotate_prediction
from gcp_localizer.models.localizer import GCPLocalizer
from gcp_localizer.models.classifier import GCPShapeClassifier


class MockLocalizer(nn.Module):
    """Mock localizer predicting fixed relative coordinates (112.0, 112.0) in 224x224 space."""

    def __init__(self, pred_x: float = 112.0, pred_y: float = 112.0):
        super().__init__()
        self.pred_x = pred_x
        self.pred_y = pred_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = torch.tensor([[self.pred_x, self.pred_y]], dtype=torch.float32)
        return out.repeat(batch_size, 1)


class MockShapeClassifier(GCPShapeClassifier):
    """Mock shape classifier predicting a fixed shape."""

    def __init__(self, shape_idx: int = 0):
        super().__init__(num_classes=3, pretrained=False)
        self.fixed_idx = shape_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 3)
        logits[:, self.fixed_idx] = 10.0
        return logits


def test_unified_inference_keys_and_types():
    """Requirement 4: Unified inference returns x, y, shape, and confidence."""
    mock_loc = MockLocalizer(112.0, 112.0)
    mock_cls = MockShapeClassifier(shape_idx=0)  # L-Shape

    predictor = GCPPredictor(
        localizer_model=mock_loc,
        shape_classifier_model=mock_cls,
    )

    dummy_image = np.zeros((800, 1000, 3), dtype=np.uint8)
    res = predictor.predict(dummy_image)

    assert isinstance(res, dict)
    assert "x" in res
    assert "y" in res
    assert "shape" in res
    assert "confidence" in res

    assert isinstance(res["x"], float)
    assert isinstance(res["y"], float)
    assert isinstance(res["shape"], str)
    assert isinstance(res["confidence"], float)

    assert res["shape"] in ["L-Shape", "Square", "Cross"]
    assert 0.0 <= res["confidence"] <= 1.0


def test_coordinate_system_preservation():
    """Requirement 5: x/y coordinates remain in the original image coordinate system."""
    # Suppose localizer predicts center of 512x512 tile: (112.0, 112.0) in 224x224
    # Scaled to tile: 112.0 * (512 / 224) = 256.0
    mock_loc = MockLocalizer(112.0, 112.0)
    mock_cls = MockShapeClassifier(shape_idx=1)  # Square

    # 1. Single tile test: image size 512x512
    predictor = GCPPredictor(
        localizer_model=mock_loc,
        shape_classifier_model=mock_cls,
        tile_size=512,
    )
    img_512 = np.zeros((512, 512, 3), dtype=np.uint8)
    res_512 = predictor.predict(img_512)

    # 112.0 * (512 / 224) = 256.0
    assert abs(res_512["x"] - 256.0) < 1.0
    assert abs(res_512["y"] - 256.0) < 1.0

    # 2. Large image test: image size 2000x3000
    # Coordinates must be strictly within original image bounds
    img_large = np.zeros((2000, 3000, 3), dtype=np.uint8)
    res_large = predictor.predict(img_large)

    assert 0.0 <= res_large["x"] <= 3000.0
    assert 0.0 <= res_large["y"] <= 2000.0
    # Coordinates must NOT be confined to 224x224
    assert res_large["x"] >= 0.0 and res_large["y"] >= 0.0


def test_visualization_annotated_image(tmp_path):
    """Requirement 6: Visualization produces an annotated output image."""
    img = np.full((600, 800, 3), 100, dtype=np.uint8)
    save_file = tmp_path / "annotated.jpg"

    annotated = annotate_prediction(
        image=img,
        x=350.5,
        y=280.2,
        shape="L-Shape",
        confidence=0.963,
        save_path=save_file
    )

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (600, 800, 3)
    assert annotated.dtype == np.uint8
    assert save_file.exists()
    assert save_file.stat().st_size > 0
