"""Unified inference pipeline for GCP detection, localization, and shape classification."""

from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from gcp_localizer.data.datasets import preprocess_image_tensor
from gcp_localizer.models.localizer import GCPLocalizer
from gcp_localizer.models.classifier import GCPClassifier, GCPShapeClassifier
from gcp_localizer.inference.visualization import annotate_prediction


class GCPPredictor:
    """Unified GCP predictor combining coordinate localization and shape classification.

    Given a full-resolution aerial image:
    1. Identifies the GCP candidate tile in the image.
    2. Runs the GCP localizer regression model to predict the marker center.
    3. Transforms local tile coordinates back to the original full-resolution image coordinate system.
    4. Extracts a centered GCP crop tile according to project preprocessing conventions.
    5. Runs the ResNet-18 shape classifier to predict marker shape ('L-Shape', 'Square', 'Cross').
    6. Returns the combined prediction dict and optionally an annotated visualization.
    """

    def __init__(
        self,
        localizer_model: Optional[Union[nn.Module, str, Path]] = None,
        shape_classifier_model: Optional[Union[nn.Module, str, Path]] = None,
        presence_classifier_model: Optional[Union[nn.Module, str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        tile_size: int = 512,
        tile_stride: int = 128,
        model_input_size: int = 224,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.model_input_size = model_input_size

        # 1. Initialize Localizer
        self.localizer_model = self._load_localizer(localizer_model)
        self.localizer_model.to(self.device).eval()

        # 2. Initialize Shape Classifier
        self.shape_classifier_model = self._load_shape_classifier(shape_classifier_model)
        self.shape_classifier_model.to(self.device).eval()

        # 3. Optional Presence Classifier (coarse tile filter)
        self.presence_classifier_model = self._load_presence_classifier(presence_classifier_model)
        if self.presence_classifier_model is not None:
            self.presence_classifier_model.to(self.device).eval()

    def _load_localizer(self, model_or_path: Optional[Union[nn.Module, str, Path]]) -> GCPLocalizer:
        if isinstance(model_or_path, nn.Module):
            return model_or_path

        localizer = GCPLocalizer(pretrained=False)
        candidate_paths = [
            model_or_path,
            "models/localizer.pt",
            "artifacts/models/best_localizer_model.pt",
        ]
        for p in candidate_paths:
            if p is not None and Path(p).exists():
                state_dict = torch.load(p, map_location="cpu")
                localizer.load_state_dict(state_dict)
                print(f"[GCPPredictor] Loaded localizer checkpoint from {p}")
                return localizer

        print("[GCPPredictor] No localizer checkpoint found on disk. Initializing with pretrained weights.")
        return GCPLocalizer(pretrained=True)

    def _load_shape_classifier(self, model_or_path: Optional[Union[nn.Module, str, Path]]) -> GCPShapeClassifier:
        if isinstance(model_or_path, nn.Module):
            return model_or_path

        classifier = GCPShapeClassifier(num_classes=3, pretrained=False)
        candidate_paths = [
            model_or_path,
            "models/classifier.pt",
            "artifacts/models/best_shape_model.pt",
        ]
        for p in candidate_paths:
            if p is not None and Path(p).exists():
                state_dict = torch.load(p, map_location="cpu")
                # Check fc weight size to verify num_classes == 3
                fc_weight = state_dict.get("backbone.fc.weight", state_dict.get("fc.weight"))
                if fc_weight is not None and fc_weight.shape[0] == 3:
                    classifier.load_state_dict(state_dict)
                    print(f"[GCPPredictor] Loaded shape classifier checkpoint from {p}")
                    return classifier

        print("[GCPPredictor] No 3-class shape classifier checkpoint found. Initializing with pretrained weights.")
        return GCPShapeClassifier(num_classes=3, pretrained=True)

    def _load_presence_classifier(self, model_or_path: Optional[Union[nn.Module, str, Path]]) -> Optional[GCPClassifier]:
        if isinstance(model_or_path, nn.Module):
            return model_or_path

        candidate_paths = [
            model_or_path,
            "artifacts/models/best_classifier_model.pt",
            "models/presence_classifier.pt",
        ]
        for p in candidate_paths:
            if p is not None and Path(p).exists():
                try:
                    state_dict = torch.load(p, map_location="cpu")
                    fc_weight = state_dict.get("backbone.fc.weight", state_dict.get("fc.weight"))
                    if fc_weight is not None and fc_weight.shape[0] == 2:
                        presence_model = GCPClassifier(num_classes=2, pretrained=False)
                        presence_model.load_state_dict(state_dict)
                        print(f"[GCPPredictor] Loaded marker presence classifier from {p}")
                        return presence_model
                except Exception:
                    pass
        return None

    def _find_best_candidate_tile(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Identify candidate tile in full image using presence classifier or fallback search."""
        h, w = image.shape[:2]

        # If image fits in one tile, return it directly
        if w <= self.tile_size or h <= self.tile_size:
            return image, 0, 0

        # If binary presence classifier is available, perform sliding window scan
        if self.presence_classifier_model is not None:
            best_prob = -1.0
            best_tile = image[0:self.tile_size, 0:self.tile_size]
            best_x, best_y = 0, 0

            stride = max(self.tile_stride, 128)
            tiles = []
            coords = []

            for y in range(0, h - self.tile_size + 1, stride):
                for x in range(0, w - self.tile_size + 1, stride):
                    tile = image[y:y + self.tile_size, x:x + self.tile_size]
                    tiles.append(tile)
                    coords.append((x, y))

            # Batch evaluate tiles
            batch_size = 16
            with torch.no_grad():
                for i in range(0, len(tiles), batch_size):
                    batch_tiles = tiles[i:i + batch_size]
                    batch_tensors = [
                        preprocess_image_tensor(cv2.resize(t, (self.model_input_size, self.model_input_size)))
                        for t in batch_tiles
                    ]
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    logits = self.presence_classifier_model(batch_tensor)
                    probs = torch.softmax(logits, dim=-1)[:, 1]  # positive marker class probability

                    max_idx = int(probs.argmax().item())
                    max_prob = float(probs[max_idx].item())

                    if max_prob > best_prob:
                        best_prob = max_prob
                        best_tile = batch_tiles[max_idx]
                        best_x, best_y = coords[i + max_idx]

            return best_tile, best_x, best_y

        # Fallback if no presence classifier: center crop
        cx, cy = w // 2, h // 2
        half = self.tile_size // 2
        x = max(0, min(w - self.tile_size, cx - half))
        y = max(0, min(h - self.tile_size, cy - half))
        return image[y:y + self.tile_size, x:x + self.tile_size], x, y

    def _extract_centered_crop(self, image: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Extract centered tile around (cx, cy) following preprocessing conventions with border reflection/padding."""
        h, w = image.shape[:2]
        half = self.tile_size // 2

        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = x1 + self.tile_size
        y2 = y1 + self.tile_size

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            x1 += pad_left
            x2 += pad_left
            y1 += pad_top
            y2 += pad_top
            crop = padded[y1:y2, x1:x2]
        else:
            crop = image[y1:y2, x1:x2]

        return cv2.resize(crop, (self.model_input_size, self.model_input_size))

    def predict(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """Run unified localization and shape classification on an aerial image.

        Args:
            image: Full-resolution image as RGB numpy array (H, W, 3) or file path.

        Returns:
            Dictionary with:
            - 'x': float, GCP center X in original full-resolution coordinates
            - 'y': float, GCP center Y in original full-resolution coordinates
            - 'shape': str, predicted shape ('Cross', 'Square', or 'L-Shape')
            - 'confidence': float, classifier prediction confidence [0, 1]
            - 'class_id': int, numeric class id
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Failed to read image at {img_path}")
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img_rgb = image
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        orig_h, orig_w = img_rgb.shape[:2]

        # 1. Obtain best candidate tile
        candidate_tile, tile_x, tile_y = self._find_best_candidate_tile(img_rgb)
        tile_h, tile_w = candidate_tile.shape[:2]

        # 2. Localize marker inside candidate tile
        tile_resized = cv2.resize(candidate_tile, (self.model_input_size, self.model_input_size))
        loc_tensor = preprocess_image_tensor(tile_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            local_coords = self.localizer_model(loc_tensor)[0].cpu().numpy()

        # Convert relative coords from model input space (224x224) to tile space
        scale_x = tile_w / float(self.model_input_size)
        scale_y = tile_h / float(self.model_input_size)
        local_x = local_coords[0] * scale_x
        local_y = local_coords[1] * scale_y

        # Global coordinates in full-resolution image space
        global_x = float(tile_x + local_x)
        global_y = float(tile_y + local_y)

        # Clamp inside image boundary
        global_x = max(0.0, min(float(orig_w), global_x))
        global_y = max(0.0, min(float(orig_h), global_y))

        # 3. Extract centered GCP tile for shape classifier using preprocessing conventions
        shape_input_crop = self._extract_centered_crop(img_rgb, global_x, global_y)

        # 4. Run Shape Classifier
        shape_pred = self.shape_classifier_model.predict_tile(shape_input_crop, device=self.device)

        return {
            "x": round(global_x, 2),
            "y": round(global_y, 2),
            "shape": shape_pred["class"],
            "confidence": round(shape_pred["confidence"], 4),
            "class_id": shape_pred["class_id"],
        }

    def predict_and_annotate(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Run unified prediction and generate the annotated visualization.

        Args:
            image: Full-resolution image array or file path.
            save_path: Optional path to save the annotated output.

        Returns:
            Tuple of (prediction_dict, annotated_image_numpy_array).
        """
        pred = self.predict(image)

        # Load raw image for visualization if given as path
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raw_rgb = image

        annotated = annotate_prediction(
            image=raw_rgb,
            x=pred["x"],
            y=pred["y"],
            shape=pred["shape"],
            confidence=pred["confidence"],
            save_path=save_path,
        )

        return pred, annotated
