from __future__ import annotations

from pathlib import Path
from typing import Any


class YOLOModel:
    def __init__(self, model_path: str, task_name: str) -> None:
        self.model_path = model_path
        self.task_name = task_name
        self.model: Any | None = None

        if not model_path:
            raise ValueError(f"{task_name} model path is empty")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{task_name} model not found: {model_path}")

    def _ensure_loaded(self) -> Any:
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
        return self.model

    def predict(
        self,
        image,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 50,
        device: str = "cpu",
    ):
        model = self._ensure_loaded()
        return model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            verbose=False,
        )