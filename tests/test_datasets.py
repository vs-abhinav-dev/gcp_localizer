import numpy as np
import torch
from gcp_localizer.data.datasets import FastGCPDataset, MarkerClassifierDataset, ShapeDataset


def test_fast_gcp_dataset(tmp_path):
    dummy_npz = tmp_path / "sample.npz"
    np.savez_compressed(
        dummy_npz,
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        target=np.array([100.0, 100.0], dtype=np.float32)
    )

    ds = FastGCPDataset([dummy_npz])
    assert len(ds) == 1

    img_tensor, target_tensor = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert target_tensor.shape == (2,)


def test_marker_classifier_dataset(tmp_path):
    dummy_npz = tmp_path / "sample_cls.npz"
    np.savez_compressed(
        dummy_npz,
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        label=1
    )

    ds = MarkerClassifierDataset([dummy_npz])
    assert len(ds) == 1

    img_tensor, label_tensor = ds[0]
    assert img_tensor.shape == (3, 224, 224)
    assert label_tensor.item() == 1


def test_shape_dataset(tmp_path):
    dummy_npz = tmp_path / "sample_shape.npz"
    np.savez_compressed(
        dummy_npz,
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        shape="Square"
    )

    ds = ShapeDataset([dummy_npz])
    assert len(ds) == 1

    img_tensor, label_tensor = ds[0]
    assert img_tensor.shape == (3, 224, 224)
    assert label_tensor.item() == 1  # Square -> 1
