#!/usr/bin/env python3
"""Integration / Smoke Test for GCP Localizer Two-Stage Pipeline."""

from glob import glob
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.data.datasets import FastGCPDataset, MarkerClassifierDataset, preprocess_image_tensor
from gcp_localizer.models.classifier import GCPClassifier
from gcp_localizer.models.localizer import GCPLocalizer


def find_processed_dir() -> Path:
    candidates = [
        Path("data/processed"),
        Path("/run/media/MrShah_21/48A291915D5A64CF/processed"),
    ]
    for c in candidates:
        if c.exists() and len(glob(str(c / "*" / "*" / "*.npz"))) > 0:
            return c
    raise FileNotFoundError("Could not find processed dataset directory.")


def test_integration():
    out_lines = ["============================================================",
                 "GCP LOCALIZER - INTEGRATION / SMOKE TEST RESULTS",
                 "============================================================"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_dir = find_processed_dir()
    val_pos_files = sorted(glob(str(processed_dir / "val" / "positive" / "*.npz")))
    val_neg_files = sorted(glob(str(processed_dir / "val" / "negative" / "*.npz")))

    # 1. Dataset Loading Test
    try:
        assert len(val_pos_files) > 0, "No positive validation files found"
        assert len(val_neg_files) > 0, "No negative validation files found"
        sample_data = np.load(val_pos_files[0], allow_pickle=True)
        assert "image" in sample_data.files, "Image tensor missing in npz file"
        assert "target" in sample_data.files, "Target coordinates missing in npz file"
        res_str = "PASS: processed dataset loading"
        print(res_str)
        out_lines.append(res_str)
    except Exception as e:
        res_str = f"FAIL: processed dataset loading ({e})"
        print(res_str)
        out_lines.append(res_str)
        sys.exit(1)

    # 2. Classifier Inference Test
    try:
        cls_ckpt = Path("artifacts/models/best_classifier_model.pt")
        classifier = GCPClassifier(num_classes=2, pretrained=False)
        classifier.load_state_dict(torch.load(cls_ckpt, map_location=device))
        classifier.eval().to(device)

        img_tensor = preprocess_image_tensor(sample_data["image"]).unsqueeze(0).to(device)
        cls_logits = classifier(img_tensor)
        assert cls_logits.shape == (1, 2), f"Unexpected classifier output shape {cls_logits.shape}"
        res_str = "PASS: presence classification"
        print(res_str)
        out_lines.append(res_str)
    except Exception as e:
        res_str = f"FAIL: presence classification ({e})"
        print(res_str)
        out_lines.append(res_str)
        sys.exit(1)

    # 3. Localizer Inference Test
    try:
        loc_ckpt = Path("artifacts/models/best_localizer_model.pt")
        localizer = GCPLocalizer(pretrained=False)
        localizer.load_state_dict(torch.load(loc_ckpt, map_location=device))
        localizer.eval().to(device)

        loc_preds = localizer(img_tensor)
        assert loc_preds.shape == (1, 2), f"Unexpected localizer output shape {loc_preds.shape}"
        res_str = "PASS: localization"
        print(res_str)
        out_lines.append(res_str)
    except Exception as e:
        res_str = f"FAIL: localization ({e})"
        print(res_str)
        out_lines.append(res_str)
        sys.exit(1)

    # 4. Integrated Inference Test
    try:
        with torch.no_grad():
            c_logits = classifier(img_tensor)
            c_pred = c_logits.argmax(dim=1).item()
            if c_pred == 1:
                coords = localizer(img_tensor).cpu().numpy()[0]
                assert len(coords) == 2, "Expected (x, y) 2-tuple output from localizer"

        res_str = "PASS: integrated inference"
        print(res_str)
        out_lines.append(res_str)
    except Exception as e:
        res_str = f"FAIL: integrated inference ({e})"
        print(res_str)
        out_lines.append(res_str)
        sys.exit(1)

    out_lines.append("------------------------------------------------------------")
    out_lines.append("ALL INTEGRATION TESTS PASSED SUCCESSFULLY.")

    out_dir = Path("reports/task5")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "integration_test.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Saved integration test report to {out_file}")


if __name__ == "__main__":
    test_integration()
