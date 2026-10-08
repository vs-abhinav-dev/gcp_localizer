import numpy as np
from gcp_localizer.data.preprocessing import crop_positive_sample, crop_negative_sample
from gcp_localizer.data.splits import split_annotations


def test_split_annotations_no_leakage():
    sample_annotations = {f"img_{i}.jpg": {"mark": {"x": i, "y": i}} for i in range(100)}
    train_anns, val_anns = split_annotations(sample_annotations, val_ratio=0.2, seed=42)

    assert len(train_anns) == 80
    assert len(val_anns) == 20

    # Ensure no overlap between original image keys
    intersection = set(train_anns.keys()).intersection(set(val_anns.keys()))
    assert len(intersection) == 0


def test_crop_positive_sample():
    dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cx, cy = 500.0, 500.0

    res = crop_positive_sample(dummy_img, cx, cy, crop_size=512, output_size=224, max_offset=128)
    assert res["image"].shape == (224, 224, 3)
    assert res["target"].shape == (2,)


def test_crop_negative_sample():
    dummy_img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    cx, cy = 1000.0, 1000.0

    neg_crop = crop_negative_sample(dummy_img, cx, cy, crop_size=512, output_size=224, min_distance=400.0)
    assert neg_crop is not None
    assert neg_crop.shape == (224, 224, 3)
