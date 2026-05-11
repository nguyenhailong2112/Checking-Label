from pathlib import Path

import cv2
from PIL import Image
import numpy as np

from edge_inspector.training.fine_tune import FineTuneManager, FineTuneRequest


def test_prepare_dataset(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    Image.fromarray(image).save(img_path)

    req = FineTuneRequest(
        model_type="defect",
        image_label_pairs=[(img_path, "DefectNG")],
        base_model_path=tmp_path / "base.pt",
        output_dir=tmp_path / "out",
    )

    manager = FineTuneManager(workspace=tmp_path / "ws")
    dataset_yaml = manager.prepare_dataset(req)

    assert dataset_yaml.exists()
    label_file = dataset_yaml.parent / "labels" / "train" / "sample.txt"
    assert label_file.exists()
    assert "0 0.5 0.5 1.0 1.0" in label_file.read_text(encoding="utf-8")