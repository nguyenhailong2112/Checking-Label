from __future__ import annotations

import cv2
import numpy as np


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def aspect_ratio_score(candidate: tuple[int, int, int, int], expected: tuple[int, int, int, int]) -> float:
    def ratio(box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        h = max(1, y2 - y1)
        return max(1, x2 - x1) / h

    cand_ratio = ratio(candidate)
    exp_ratio = ratio(expected)
    diff = abs(cand_ratio - exp_ratio) / max(cand_ratio, exp_ratio)
    return max(0.0, min(1.0, 1.0 - diff))


def histogram_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
    if image_a.size == 0 or image_b.size == 0:
        return 0.0
    a = cv2.resize(image_a, (128, 128))
    b = cv2.resize(image_b, (128, 128))
    hist_a = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, float((score + 1.0) / 2.0)))
