from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image
import yaml


@dataclass
class FineTuneRequest:
    model_type: str
    image_label_pairs: list[tuple[Path, str]]
    base_model_path: Path
    output_dir: Path
    epochs: int = 5
    image_size: int = 640


class FineTuneManager:
    SUPPORTED_CLASSES = {
        "label": ["PCLabel", "SanDisk", "WesternDigitalType1", "WesternDigitalType2", "WesternDigitalType3", "WesternDigitalType4"],
        "code": ["Code1D", "Code2D"],
        "defect": ["DefectNG"],
    }

    def __init__(self, workspace: Path = Path("data/fine_tune")) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, request: FineTuneRequest) -> Path:
        classes = self.SUPPORTED_CLASSES[request.model_type]
        class_to_id = {name: idx for idx, name in enumerate(classes)}

        dataset_root = self.workspace / f"{request.model_type}_dataset"
        images_dir = dataset_root / "images" / "train"
        labels_dir = dataset_root / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for src, class_name in request.image_label_pairs:
            if class_name not in class_to_id:
                raise ValueError(f"Unsupported class '{class_name}' for model {request.model_type}")
            with Image.open(src) as img:
                img.verify()

            dst_image = images_dir / src.name
            shutil.copy2(src, dst_image)

            cls_id = class_to_id[class_name]
            # default full-image bbox for fast adaptation workflow
            yolo_line = f"{cls_id} 0.5 0.5 1.0 1.0\n"
            (labels_dir / f"{src.stem}.txt").write_text(yolo_line, encoding="utf-8")

        data_yaml = {
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/train",
            "names": classes,
        }
        yaml_path = dataset_root / "dataset.yaml"
        yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        return yaml_path

    def run_training(self, request: FineTuneRequest, dataset_yaml: Path) -> Path:
        from ultralytics import YOLO

        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(request.base_model_path))
        result = model.train(
            data=str(dataset_yaml),
            epochs=request.epochs,
            imgsz=request.image_size,
            project=str(request.output_dir),
            name=f"{request.model_type}_finetune",
            exist_ok=True,
            verbose=False,
        )

        best_pt = Path(result.save_dir) / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Fine-tune completed but best.pt not found at {best_pt}")
        return best_pt