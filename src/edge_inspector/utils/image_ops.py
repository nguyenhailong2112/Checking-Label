from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1]
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    Image.fromarray(bgr_to_rgb(image)).save(path)


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
        pil_img = Image.fromarray(bgr_to_rgb(output))
        pil_img = ImageEnhance.Contrast(pil_img).enhance(alpha)
        output = np.asarray(pil_img)[:, :, ::-1]
        output = np.clip(output.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)
    if sharpen:
        pil_img = Image.fromarray(bgr_to_rgb(output)).filter(ImageFilter.SHARPEN)
        output = np.asarray(pil_img)[:, :, ::-1]
    return output


def visualize_boxes(image: np.ndarray, boxes: list[dict], color: tuple[int, int, int], prefix: str) -> np.ndarray:
    vis_rgb = bgr_to_rgb(image.copy())
    pil_img = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(pil_img)
    rgb_color = tuple(color[::-1])
    font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        conf = b["confidence"]
        cls = b["class_name"]
        draw.rectangle((x1, y1, x2, y2), outline=rgb_color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{prefix}:{cls} {conf:.2f}", fill=rgb_color, font=font)

    return np.asarray(pil_img)[:, :, ::-1]