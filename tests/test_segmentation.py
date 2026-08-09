from pathlib import Path
import pytest
from src.data.segment_hands import HandCropper


def test_cropper_requires_explicit_task_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        HandCropper(Path(tmp_path) / "missing.task")
