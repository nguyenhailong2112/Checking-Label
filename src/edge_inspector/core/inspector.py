from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from edge_inspector.core.models import YOLOModel
from edge_inspector.core.schemas import BoundingBox, DecodeResult, InspectionResult
from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.image_ops import crop_from_xyxy, enhance_image, visualize_boxes

logger = logging.getLogger(__name__)


class LabelBarcodeInspector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.label_model = YOLOModel(config.get("models.label_model_path"), "label")
        self.code_model = YOLOModel(config.get("models.code_model_path"), "code")
        self.defect_model = YOLOModel(config.get("models.defect_model_path"), "defect")

    def _predict_to_boxes(self, result) -> list[dict]:
        boxes: list[dict] = []
        names = result.names
        if result.boxes is None:
            return boxes

        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, clss):
            boxes.append(
                {
                    "xyxy": tuple(map(int, box.tolist())),
                    "confidence": float(conf),
                    "class_name": str(names.get(int(cls_id), cls_id)),
                }
            )
        return boxes


    def _decode_barcode(self, image: np.ndarray) -> DecodeResult:
        try:
            from pyzbar.pyzbar import decode as zbar_decode
        except ImportError:
            logger.warning("pyzbar is not installed. Skipping decode stage.")
            return DecodeResult(success=False, decoded_text=None, code_type=None)

        decoded_items = zbar_decode(image)
        if not decoded_items:
            return DecodeResult(success=False, decoded_text=None, code_type=None)

        first = decoded_items[0]
        return DecodeResult(
            success=True,
            decoded_text=first.data.decode("utf-8", errors="ignore"),
            code_type=first.type,
        )

    def run(self, image: np.ndarray, image_name: str = "input.jpg") -> tuple[InspectionResult, np.ndarray, np.ndarray | None]:
        conf = float(self.config.get("inference.conf_threshold", 0.25))
        iou = float(self.config.get("inference.iou_threshold", 0.45))
        imgsz = int(self.config.get("inference.image_size", 640))
        max_det = int(self.config.get("inference.max_det", 50))
        device = str(self.config.get("inference.device", "cpu"))

        label_pred = self.label_model.predict(image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)[0]
        label_boxes = sorted(self._predict_to_boxes(label_pred), key=lambda x: x["confidence"], reverse=True)

        if not label_boxes:
            result = InspectionResult(
                timestamp=datetime.utcnow(),
                image_name=image_name,
                decision="NG",
                total_confidence=0.0,
                label_box=None,
                code_boxes=[],
                defect_boxes=[],
                decode_result=DecodeResult(success=False),
                notes=["Không phát hiện label"],
            )
            return result, image, None

        top_label = label_boxes[0]
        label_crop = crop_from_xyxy(image, top_label["xyxy"])
        processed_crop = enhance_image(
            label_crop,
            enhance_contrast=bool(self.config.get("preprocess.enhance_contrast", True)),
            sharpen=bool(self.config.get("preprocess.sharpen", True)),
            alpha=float(self.config.get("preprocess.brightness_alpha", 1.1)),
            beta=int(self.config.get("preprocess.brightness_beta", 3)),
        )

        code_pred = self.code_model.predict(processed_crop, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)[0]
        defect_pred = self.defect_model.predict(processed_crop, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)[0]

        code_boxes = self._predict_to_boxes(code_pred)
        defect_boxes = self._predict_to_boxes(defect_pred)

        decode_result = self._decode_barcode(processed_crop)

        defect_found = len(defect_boxes) > 0
        decision = "NG" if defect_found else "OK"
        total_conf = float(
            np.mean(
                [top_label["confidence"]]
                + [x["confidence"] for x in code_boxes]
                + ([1.0 - min(1.0, defect_boxes[0]["confidence"])] if defect_boxes else [1.0])
            )
        )

        result = InspectionResult(
            timestamp=datetime.utcnow(),
            image_name=image_name,
            decision=decision,
            total_confidence=max(0.0, min(1.0, total_conf)),
            label_box=BoundingBox(
                x1=top_label["xyxy"][0],
                y1=top_label["xyxy"][1],
                x2=top_label["xyxy"][2],
                y2=top_label["xyxy"][3],
                confidence=top_label["confidence"],
                class_name=top_label["class_name"],
            ),
            code_boxes=[
                BoundingBox(
                    x1=b["xyxy"][0], y1=b["xyxy"][1], x2=b["xyxy"][2], y2=b["xyxy"][3],
                    confidence=b["confidence"], class_name=b["class_name"]
                )
                for b in code_boxes
            ],
            defect_boxes=[
                BoundingBox(
                    x1=b["xyxy"][0], y1=b["xyxy"][1], x2=b["xyxy"][2], y2=b["xyxy"][3],
                    confidence=b["confidence"], class_name=b["class_name"]
                )
                for b in defect_boxes
            ],
            decode_result=decode_result,
            notes=[],
        )

        vis = image.copy()
        vis = visualize_boxes(vis, [top_label], (255, 0, 0), "LABEL")
        vis_crop = visualize_boxes(processed_crop, code_boxes, (0, 255, 0), "CODE")
        vis_crop = visualize_boxes(vis_crop, defect_boxes, (0, 0, 255), "DEFECT")
        return result, vis, vis_crop

    def save_result(self, result: InspectionResult, visualization: np.ndarray | None = None) -> None:
        out_dir = Path(self.config.get("output.save_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
        stem = Path(result.image_name).stem
        json_path = out_dir / f"{stem}_{ts}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        if visualization is not None and bool(self.config.get("output.save_visualization", True)):
            cv2.imwrite(str(out_dir / f"{stem}_{ts}.jpg"), visualization)

        logger.info("Saved result to %s", out_dir)