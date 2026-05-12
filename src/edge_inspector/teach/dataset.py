from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from edge_inspector.teach.schemas import ApprovedAnnotation, TeachSample, TeachTarget
from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.image_ops import write_image
from edge_inspector.utils.time import utc_now

DEFAULT_CLASSES: dict[TeachTarget, list[str]] = {
    "label": ["Label"],
    "code": ["Code1D", "Code2D"],
    "defect": ["DefectNG"],
}


def xyxy_to_yolo(box: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    x1 = max(0, min(image_width, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("bbox is outside image bounds")
    return (
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def yolo_to_xyxy(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    cx, cy, width, height = box
    x1 = int(round((cx - width / 2.0) * image_width))
    y1 = int(round((cy - height / 2.0) * image_height))
    x2 = int(round((cx + width / 2.0) * image_width))
    y2 = int(round((cy + height / 2.0) * image_height))
    return x1, y1, x2, y2


class TeachDatasetWriter:
    """Write operator-approved Teach Mode samples in YOLO format."""

    def __init__(self, config: AppConfig | None = None, root: str | Path | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        elif config is not None:
            self.root = Path(config.get("teach.save_dir", "data/teach"))
        else:
            self.root = Path("data/teach")

    def save_sample(
        self,
        *,
        image: np.ndarray | Path,
        image_name: str,
        recipe_id: str,
        target: TeachTarget,
        annotations: list[ApprovedAnnotation],
        class_names: list[str] | None = None,
        negative: bool = False,
        source: str = "upload",
        approved_by: str | None = None,
        model_snapshot: dict[str, str] | None = None,
    ) -> TeachSample:
        classes = list(class_names or DEFAULT_CLASSES[target])
        class_to_id = {name: idx for idx, name in enumerate(classes)}
        for annotation in annotations:
            if annotation.target != target:
                raise ValueError(f"Annotation target {annotation.target} does not match sample target {target}")
            if annotation.class_name not in class_to_id:
                classes.append(annotation.class_name)
                class_to_id[annotation.class_name] = len(classes) - 1

        timestamp = utc_now()
        sample_id = f"{Path(image_name).stem}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        dataset_root = self.root / "exports" / f"{target}_dataset"
        images_dir = dataset_root / "images" / "train"
        labels_dir = dataset_root / "labels" / "train"
        metadata_dir = self.root / "sessions" / recipe_id / "metadata"
        negatives_dir = self.root / "sessions" / recipe_id / "negatives"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if negative:
            negatives_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / f"{sample_id}.jpg"
        if isinstance(image, Path):
            shutil.copy2(image, image_path)
            from PIL import Image

            with Image.open(image_path) as pil_image:
                image_width, image_height = pil_image.size
        else:
            write_image(image_path, image)
            image_height, image_width = image.shape[:2]

        label_path = labels_dir / f"{sample_id}.txt"
        lines: list[str] = []
        if not negative:
            for annotation in annotations:
                yolo_box = xyxy_to_yolo(annotation.xyxy, image_width, image_height)
                cls_id = class_to_id[annotation.class_name]
                lines.append(f"{cls_id} " + " ".join(f"{value:.6f}" for value in yolo_box))
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        data_yaml = {
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/train",
            "names": classes,
        }
        (dataset_root / "dataset.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

        if negative:
            shutil.copy2(image_path, negatives_dir / image_path.name)

        sample = TeachSample(
            sample_id=sample_id,
            recipe_id=recipe_id,
            target=target,
            image_name=image_name,
            image_path=str(image_path),
            annotations=annotations,
            negative=negative,
            source=source,  # type: ignore[arg-type]
            approved_by=approved_by,
            model_snapshot=model_snapshot or {},
            created_at=timestamp,
        )
        metadata_path = metadata_dir / f"{sample_id}.json"
        metadata_path.write_text(
            json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return sample
