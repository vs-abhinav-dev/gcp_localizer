"""Training routines for GCP Localizer models"""

from gcp_localizer.training.localizer import train_localizer
from gcp_localizer.training.classifier import train_classifier

__all__ = ["train_localizer", "train_classifier"]
