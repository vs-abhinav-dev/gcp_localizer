import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GCPLocalizer(nn.Module):
    """ResNet18-based regression model for predicting GCP relative (x, y) coordinates within a tile."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output tensor of shape (B, 2) representing (x, y) relative coordinates.
        """
        return self.backbone(x)
