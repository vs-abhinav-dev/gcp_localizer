import json
import pytest
from gcp_localizer.data.annotations import load_annotations, annotations_to_df


def test_load_annotations(tmp_path):
    sample_data = {
        "img1.jpg": {"mark": {"x": 100.0, "y": 200.0}, "verified_shape": "Square"}
    }
    file_path = tmp_path / "test_marks.json"
    file_path.write_text(json.dumps(sample_data))

    loaded = load_annotations(file_path)
    assert "img1.jpg" in loaded
    assert loaded["img1.jpg"]["mark"]["x"] == 100.0


def test_load_annotations_missing():
    with pytest.raises(FileNotFoundError):
        load_annotations("non_existent_file.json")


def test_annotations_to_df():
    sample_data = {
        "img1.jpg": {"mark": {"x": 100.0, "y": 200.0}, "verified_shape": "Square"},
        "img2.jpg": {"mark": None}
    }
    df = annotations_to_df(sample_data)
    assert len(df) == 1
    assert list(df.columns) == ["path", "x", "y", "shape"]
    assert df.iloc[0]["shape"] == "Square"
