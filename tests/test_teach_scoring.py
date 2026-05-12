import numpy as np

from edge_inspector.teach.scoring import aspect_ratio_score, bbox_iou, histogram_similarity


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_aspect_ratio_score() -> None:
    assert aspect_ratio_score((0, 0, 20, 10), (5, 5, 25, 15)) == 1.0
    assert aspect_ratio_score((0, 0, 10, 20), (0, 0, 20, 10)) < 1.0


def test_histogram_similarity_identical_images() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert histogram_similarity(image, image) >= 0.99
