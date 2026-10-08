import json
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd


def load_annotations(annotation_filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load GCP annotations JSON file.

    Args:
        annotation_filepath: Path to annotations JSON file.

    Returns:
        Dictionary mapping relative image paths to annotation data.
    """
    filepath = Path(annotation_filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Annotation file not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    return annotations


def annotations_to_df(annotations: Dict[str, Any]) -> pd.DataFrame:
    """Convert annotations dictionary into a pandas DataFrame for EDA/analysis.

    Args:
        annotations: Dictionary mapping relative image paths to mark info.

    Returns:
        DataFrame containing columns: path, x, y, shape.
    """
    rows = []
    for path, info in annotations.items():
        mark = info.get("mark")
        if mark is not None:
            rows.append({
                "path": path,
                "x": mark.get("x"),
                "y": mark.get("y"),
                "shape": info.get("verified_shape"),
            })

    return pd.DataFrame(rows)
