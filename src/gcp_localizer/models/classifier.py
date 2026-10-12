from typing import Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from gcp_localizer.data.datasets import preprocess_image_tensor


DEFAULT_SHAPE_TO_IDX: Dict[str, int] = {
    "L-Shape": 0,
    "Square": 1,
    "Cross": 2,
}
DEFAULT_IDX_TO_SHAPE: Dict[int, str] = {v: k for k, v in DEFAULT_SHAPE_TO_IDX.items()}

DEFAULT_BINARY_TO_IDX: Dict[str, int] = {
    "Non-Marker": 0,
    "Marker": 1,
}
DEFAULT_IDX_TO_BINARY: Dict[int, str] = {v: k for k, v in DEFAULT_BINARY_TO_IDX.items()}


class GCPClassifier(nn.Module):
    """ResNet18-based classifier for marker presence (2 classes) or shape classification (3 classes)."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
        idx_to_class: Optional[Dict[int, str]] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        if hidden_dim is not None:
            layers = [
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.backbone.fc = nn.Sequential(*layers)
        elif dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        if idx_to_class is not None:
            self.idx_to_class = idx_to_class
        elif num_classes == 3:
            self.idx_to_class = DEFAULT_IDX_TO_SHAPE
        elif num_classes == 2:
            self.idx_to_class = DEFAULT_IDX_TO_BINARY
        else:
            self.idx_to_class = {i: f"Class_{i}" for i in range(num_classes)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.backbone(x)

    def predict_tile(
        self,
        tile: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Predict class and confidence for a single tile/image crop.

        Args:
            tile: Image numpy array (H, W, 3) in RGB or PyTorch tensor (3, H, W) or (1, 3, H, W).
            device: Target torch device. If None, uses device of model parameters.

        Returns:
            Dictionary containing predicted 'class', 'class_id', and 'confidence'.
        """
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        if isinstance(tile, np.ndarray):
            # Ensure 224x224
            if tile.shape[0] != 224 or tile.shape[1] != 224:
                import cv2
                tile = cv2.resize(tile, (224, 224))
            tensor = preprocess_image_tensor(tile).unsqueeze(0).to(device)
        elif isinstance(tile, torch.Tensor):
            if tile.ndim == 3:
                tensor = tile.unsqueeze(0).to(device)
            elif tile.ndim == 4:
                tensor = tile.to(device)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got shape {tile.shape}")
        else:
            raise TypeError(f"Unsupported tile type: {type(tile)}")

        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            class_id = int(probs.argmax().item())
            confidence = float(probs[class_id].item())

        if was_training:
            self.train()

        class_name = self.idx_to_class.get(class_id, f"Class_{class_id}")
        return {
            "class": class_name,
            "class_id": class_id,
            "confidence": round(confidence, 4)
        }

    def predict(
        self,
        tile: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Alias for predict_tile."""
        return self.predict_tile(tile, device=device)


class GCPShapeClassifier(GCPClassifier):
    """ResNet18-based 3-class GCP shape classifier (L-Shape, Square, Cross)."""

    SHAPE_TO_IDX: Dict[str, int] = DEFAULT_SHAPE_TO_IDX
    IDX_TO_SHAPE: Dict[int, str] = DEFAULT_IDX_TO_SHAPE

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
        idx_to_shape: Optional[Dict[int, str]] = None
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            hidden_dim=hidden_dim,
            idx_to_class=idx_to_shape or self.IDX_TO_SHAPE
        )

