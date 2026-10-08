"""Inference module for GCP Localizer"""

from gcp_localizer.inference.predictor import GCPPredictor
from gcp_localizer.inference.visualization import annotate_prediction

__all__ = ["GCPPredictor", "annotate_prediction"]
