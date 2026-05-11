from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from edge_inspector.core.schemas import CollectionRecord, InspectionResult
from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.image_ops import write_image
from edge_inspector.utils.time import utc_now


class ActiveLearningCollector:
    """Persist NG / low-confidence / manual samples for the PC training loop."""

    VALID_REASONS = {"NG", "LOW_CONFIDENCE", "MANUAL"}

    def __init__(self, config: AppConfig) -> None:
        self.root = Path(config.get("active_learning.save_dir", "data/collected"))
        self.save_visualization = bool(config.get("active_learning.save_visualization", True))
        self.save_crop = bool(config.get("active_learning.save_crop", True))

    def _reason_dir(self, reason: str) -> Path:
        normalized = reason.lower()
        return self.root / normalized

    def collect(
        self,
        *,
        image: np.ndarray | Path,
        result: InspectionResult,
        reason: str = "MANUAL",
        visualization: np.ndarray | None = None,
        crop: np.ndarray | None = None,
    ) -> CollectionRecord:
        reason = reason.upper()
        if reason not in self.VALID_REASONS:
            raise ValueError(f"Unsupported collection reason: {reason}")

        timestamp = utc_now()
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        stem = Path(result.image_name).stem or "sample"
        base_dir = self._reason_dir(reason)
        images_dir = base_dir / "images"
        metadata_dir = base_dir / "metadata"
        vis_dir = base_dir / "visualizations"
        crops_dir = base_dir / "crops"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.save_visualization:
            vis_dir.mkdir(parents=True, exist_ok=True)
        if self.save_crop:
            crops_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / f"{stem}_{ts}.jpg"
        if isinstance(image, Path):
            shutil.copy2(image, image_path)
        else:
            write_image(image_path, image)

        visualization_path: Path | None = None
        if visualization is not None and self.save_visualization:
            visualization_path = vis_dir / f"{stem}_{ts}_vis.jpg"
            write_image(visualization_path, visualization)

        crop_path: Path | None = None
        if crop is not None and self.save_crop:
            crop_path = crops_dir / f"{stem}_{ts}_crop.jpg"
            write_image(crop_path, crop)

        record = CollectionRecord(
            timestamp=timestamp,
            reason=reason,  # type: ignore[arg-type]
            image_name=result.image_name,
            source_image_path=str(image_path),
            visualization_path=str(visualization_path) if visualization_path else None,
            crop_path=str(crop_path) if crop_path else None,
            result=result,
        )
        metadata_path = metadata_dir / f"{stem}_{ts}.json"
        metadata_path.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record