"""Unit tests for GCP Shape Classifier model and datasets."""

import numpy as np
import pytest
import torch

from gcp_localizer.models.classifier import GCPClassifier, GCPShapeClassifier, DEFAULT_SHAPE_TO_IDX
from gcp_localizer.data.datasets import ShapeDataset, RawShapeDataset


def test_shape_classifier_output_dimensions():
    """Requirement 1: Classifier creates an output tensor with 3 classes."""
    model = GCPShapeClassifier(num_classes=3, pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (4, 3), f"Expected shape (4, 3), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values"


def test_shape_classifier_configurable_classes():
    """Verify classifier num_classes is configurable."""
    for n_cls in [2, 3, 5]:
        model = GCPClassifier(num_classes=n_cls, pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, n_cls)


def test_shape_classifier_predict_tile():
    """Requirement 3: Classifier prediction returns a valid class, class_id, and confidence."""
    model = GCPShapeClassifier(num_classes=3, pretrained=False)
    dummy_tile = np.zeros((224, 224, 3), dtype=np.uint8)

    pred = model.predict_tile(dummy_tile)

    assert isinstance(pred, dict)
    assert "class" in pred
    assert "class_id" in pred
    assert "confidence" in pred
    assert pred["class"] in ["L-Shape", "Square", "Cross"]
    assert pred["class_id"] in [0, 1, 2]
    assert 0.0 <= pred["confidence"] <= 1.0


def test_shape_dataset_valid_labels(tmp_path):
    """Requirement 2: Dataset returns valid classifier labels."""
    # Test all 3 verified shapes
    shapes = ["L-Shape", "Square", "Cross"]
    expected_ids = [0, 1, 2]

    for shape, exp_id in zip(shapes, expected_ids):
        npz_file = tmp_path / f"test_{shape}.npz"
        np.savez_compressed(
            npz_file,
            image=np.ones((224, 224, 3), dtype=np.uint8) * 128,
            shape=shape
        )
        ds = ShapeDataset([npz_file])
        assert len(ds) == 1
        tensor_img, tensor_lbl = ds[0]

        assert isinstance(tensor_img, torch.Tensor)
        assert tensor_img.shape == (3, 224, 224)
        assert tensor_lbl.item() == exp_id, f"Expected {exp_id} for {shape}, got {tensor_lbl.item()}"
