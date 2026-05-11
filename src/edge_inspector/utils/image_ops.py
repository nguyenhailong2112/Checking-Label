from __future__ import annotations

import cv2
import numpy as np


def crop_from_xyxy(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2].copy()


def enhance_image(
    image: np.ndarray,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    alpha: float = 1.1,
    beta: int = 3,
) -> np.ndarray:
    output = image.copy()
    if enhance_contrast:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        output = cv2.filter2D(output, -1, kernel)
    return output


def visualize_boxes(image: np.ndarray, boxes: list[dict], color: tuple[int, int, int], prefix: str) -> np.ndarray:
    vis = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        conf = b["confidence"]
        cls = b["class_name"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{prefix}:{cls} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return vis