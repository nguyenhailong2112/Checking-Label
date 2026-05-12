from __future__ import annotations

import ctypes.util
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

from edge_inspector.core.active_learning import ActiveLearningCollector
from edge_inspector.core.decision import DecisionEngine

from edge_inspector.core.models import YOLOModel
from edge_inspector.core.schemas import BoundingBox, DecodeResult, InspectionResult, RuntimeSettings
from edge_inspector.teach.recipe import RecipeStore
from edge_inspector.teach.schemas import ProductRecipe
from edge_inspector.teach.scoring import aspect_ratio_score, bbox_iou
from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.image_ops import crop_from_xyxy, enhance_image, visualize_boxes, write_image
from edge_inspector.utils.time import utc_now

logger = logging.getLogger(__name__)


class LabelBarcodeInspector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.label_model = YOLOModel(config.get("models.label_model_path"), "label")
        self.code_model = YOLOModel(config.get("models.code_model_path"), "code")
        self.defect_model = YOLOModel(config.get("models.defect_model_path"), "defect")
        self.collector = ActiveLearningCollector(config)
        self.recipe_store = RecipeStore(config)
        self.decision_engine = DecisionEngine(
            auto_collect_ng=bool(config.get("active_learning.auto_collect_ng", True)),
            auto_collect_low_conf=bool(config.get("active_learning.auto_collect_low_conf", True)),
        )

    def _predict_to_boxes(self, result: Any) -> list[dict]:
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

    def _runtime_settings(self) -> RuntimeSettings:
        mode = str(self.config.get("inspection.mode", "full"))
        if mode not in {"full", "label_only", "code_only", "defect_only"}:
            logger.warning("Unsupported inspection mode '%s'. Falling back to full mode.", mode)
            mode = "full"
        conf_threshold = float(self.config.get("inference.conf_threshold", 0.25))
        return RuntimeSettings(
            mode=mode,  # type: ignore[arg-type]
            conf_threshold=conf_threshold,
            low_conf_threshold=float(self.config.get("inspection.low_conf_threshold", 0.55)),
            require_code=bool(self.config.get("inspection.require_code", True)),
            require_decode=bool(self.config.get("inspection.require_decode", True)),
            inspect_defect=bool(self.config.get("inspection.inspect_defect", True)),
        )


    def _decode_barcode(self, image: np.ndarray) -> DecodeResult:
        if ctypes.util.find_library("zbar") is None:
            logger.warning("zbar shared library is not installed. Skipping decode stage.")
            return DecodeResult(success=False, decoded_text=None, code_type=None)
        if importlib.util.find_spec("pyzbar") is None:
            logger.warning("pyzbar is not installed. Skipping decode stage.")
            return DecodeResult(success=False, decoded_text=None, code_type=None)

        zbar_module = importlib.import_module("pyzbar.pyzbar")
        decoded_items = zbar_module.decode(image)
        if not decoded_items:
            return DecodeResult(success=False, decoded_text=None, code_type=None)

        first = decoded_items[0]
        return DecodeResult(
            success=True,
            decoded_text=first.data.decode("utf-8", errors="ignore"),
            code_type=first.type,
        )

    def _box_to_schema(self, box: dict) -> BoundingBox:
        return BoundingBox(
            x1=box["xyxy"][0],
            y1=box["xyxy"][1],
            x2=box["xyxy"][2],
            y2=box["xyxy"][3],
            confidence=box["confidence"],
            class_name=box["class_name"],
        )

    def _active_recipe(self) -> ProductRecipe | None:
        if not bool(self.config.get("teach.use_recipe", False)):
            return None
        recipe_id = str(self.config.get("teach.active_recipe_id", "") or "").strip()
        if not recipe_id:
            return None
        try:
            return self.recipe_store.load(recipe_id)
        except FileNotFoundError:
            logger.warning("Active recipe '%s' not found. Running without recipe.", recipe_id)
            return None

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return max(0.0, min(1.0, float(np.mean(values))))

    def _label_recipe_score(self, recipe: ProductRecipe, box: dict) -> dict[str, Any]:
        reason_codes = ["label_model_prediction"]
        model_conf = float(box["confidence"])
        components = [model_conf]
        expected_roi = recipe.label.expected_roi_xyxy
        if expected_roi is not None:
            roi_score = bbox_iou(box["xyxy"], expected_roi)
            aspect_score = aspect_ratio_score(box["xyxy"], expected_roi)
            components.extend([roi_score, aspect_score])
            reason_codes.append("label_roi_match" if roi_score >= 0.3 else "label_roi_mismatch")
        else:
            roi_score = None
            aspect_score = None
            reason_codes.append("label_recipe_no_roi")
        final_score = self._mean(components)
        if final_score >= recipe.label.min_recipe_score:
            reason_codes.append("recipe_assisted_label")
        return {
            "model_confidence": model_conf,
            "roi_score": roi_score,
            "aspect_score": aspect_score,
            "final_score": final_score,
            "reason_codes": reason_codes,
        }

    def _code_recipe_score(
        self,
        recipe: ProductRecipe,
        code_boxes: list[dict],
        decode_result: DecodeResult,
    ) -> dict[str, Any]:
        reason_codes: list[str] = []
        best_code = max(code_boxes, key=lambda x: x["confidence"], default=None)
        model_conf = float(best_code["confidence"]) if best_code else 0.0
        components = [model_conf]
        if not code_boxes:
            reason_codes.append("code_missing")
        elif recipe.code.expected_count and len(code_boxes) != recipe.code.expected_count:
            reason_codes.append("code_count_mismatch")
        else:
            reason_codes.append("code_detected")

        roi_score = None
        if best_code and recipe.code.expected_roi_xyxy is not None:
            roi_score = bbox_iou(best_code["xyxy"], recipe.code.expected_roi_xyxy)
            components.append(roi_score)
            reason_codes.append("code_roi_match" if roi_score >= 0.3 else "code_roi_mismatch")

        pattern_score = None
        if recipe.code.require_decode:
            pattern_score = 1.0 if decode_result.success else 0.0
            reason_codes.append("decode_success" if decode_result.success else "decode_failed")
            if decode_result.success and recipe.code.pattern:
                text = decode_result.decoded_text or ""
                matched = re.fullmatch(recipe.code.pattern, text) is not None
                pattern_score = 1.0 if matched else 0.0
                reason_codes.append("code_pattern_ok" if matched else "code_pattern_failed")
            components.append(pattern_score)

        final_score = self._mean(components)
        return {
            "model_confidence": model_conf,
            "roi_score": roi_score,
            "pattern_score": pattern_score,
            "final_score": final_score,
            "reason_codes": reason_codes,
        }

    def _defect_recipe_score(self, recipe: ProductRecipe, defect_boxes: list[dict]) -> dict[str, Any]:
        reason_codes: list[str] = []
        max_defect_conf = max((float(box["confidence"]) for box in defect_boxes), default=0.0)
        model_conf = 1.0 - max_defect_conf
        roi_score = None
        if defect_boxes:
            reason_codes.append("defect_found")
            if recipe.defect.roi_xyxy is not None:
                roi_score = max(bbox_iou(box["xyxy"], recipe.defect.roi_xyxy) for box in defect_boxes)
                reason_codes.append("defect_in_roi" if roi_score >= 0.3 else "defect_outside_roi")
        else:
            reason_codes.append("no_defect")
        final_score = model_conf
        return {
            "model_confidence": model_conf,
            "roi_score": roi_score,
            "final_score": final_score,
            "reason_codes": reason_codes,
        }

    def _recipe_scores(
        self,
        recipe: ProductRecipe | None,
        *,
        label_box: dict | None,
        code_boxes: list[dict],
        defect_boxes: list[dict],
        decode_result: DecodeResult,
    ) -> tuple[dict[str, Any], list[str], float | None]:
        if recipe is None:
            return {}, [], None
        scores: dict[str, Any] = {}
        reason_codes: list[str] = []
        if label_box is not None:
            scores["label"] = self._label_recipe_score(recipe, label_box)
            reason_codes.extend(scores["label"]["reason_codes"])
        if recipe.code.enabled:
            scores["code"] = self._code_recipe_score(recipe, code_boxes, decode_result)
            reason_codes.extend(scores["code"]["reason_codes"])
        if recipe.defect.enabled:
            scores["defect"] = self._defect_recipe_score(recipe, defect_boxes)
            reason_codes.extend(scores["defect"]["reason_codes"])
        final_values = [float(item["final_score"]) for item in scores.values()]
        return scores, reason_codes, self._mean(final_values) if final_values else None

    def run(
        self,
        image: np.ndarray,
        image_name: str = "input.jpg",
    ) -> tuple[InspectionResult, np.ndarray, np.ndarray | None]:
        runtime = self._runtime_settings()
        conf = runtime.conf_threshold
        iou = float(self.config.get("inference.iou_threshold", 0.45))
        imgsz = int(self.config.get("inference.image_size", 640))
        max_det = int(self.config.get("inference.max_det", 50))
        device = str(self.config.get("inference.device", "cpu"))
        notes: list[str] = []
        recipe = self._active_recipe()
        if recipe is not None:
            notes.append(f"Recipe active: {recipe.recipe_id}")

        label_pred = self.label_model.predict(image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)[0]
        label_boxes = sorted(self._predict_to_boxes(label_pred), key=lambda x: x["confidence"], reverse=True)

        if not label_boxes:
            outcome = self.decision_engine.evaluate(
                runtime=runtime,
                label_box=None,
                code_boxes=[],
                defect_boxes=[],
                decode_result=DecodeResult(success=False),
                base_notes=["Không phát hiện label"],
            )
            result = InspectionResult(
                timestamp=utc_now(),
                image_name=image_name,
                decision=outcome.decision,
                total_confidence=outcome.total_confidence,
                label_box=None,
                code_boxes=[],
                defect_boxes=[],
                decode_result=DecodeResult(success=False),
                is_low_confidence=outcome.is_low_confidence,
                collection_recommended=outcome.collection_recommended,
                collection_reason=outcome.collection_reason,
                runtime=runtime,
                notes=outcome.notes,
                recipe_id=recipe.recipe_id if recipe else None,
                model_confidence=0.0,
                recipe_confidence=0.0 if recipe else None,
                reason_codes=["label_missing"],
            )
            return result, image.copy(), None

        top_label = label_boxes[0]
        label_crop = crop_from_xyxy(image, top_label["xyxy"])
        processed_crop = enhance_image(
            label_crop,
            enhance_contrast=bool(self.config.get("preprocess.enhance_contrast", True)),
            sharpen=bool(self.config.get("preprocess.sharpen", True)),
            alpha=float(self.config.get("preprocess.brightness_alpha", 1.1)),
            beta=int(self.config.get("preprocess.brightness_beta", 3)),
        )

        run_code = runtime.mode in {"full", "code_only"}
        run_defect = runtime.mode in {"full", "defect_only"} and runtime.inspect_defect

        code_boxes: list[dict] = []
        defect_boxes: list[dict] = []
        decode_result = DecodeResult(success=False)

        if run_code:
            code_pred = \
            self.code_model.predict(processed_crop, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)[0]
            code_boxes = self._predict_to_boxes(code_pred)
            decode_target = processed_crop
            if code_boxes and bool(self.config.get("inspection.decode_on_code_crop", True)):
                best_code = sorted(code_boxes, key=lambda x: x["confidence"], reverse=True)[0]
                decode_target = crop_from_xyxy(processed_crop, best_code["xyxy"])
            decode_result = self._decode_barcode(decode_target)
        else:
            notes.append("Code inspection skipped by selected mode")

        if run_defect:
            defect_pred = \
            self.defect_model.predict(processed_crop, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)[
                0]
            defect_boxes = self._predict_to_boxes(defect_pred)
        else:
            notes.append("Defect inspection skipped by selected mode")

        label_box_schema = self._box_to_schema(top_label)
        code_box_schemas = [self._box_to_schema(b) for b in code_boxes]
        defect_box_schemas = [self._box_to_schema(b) for b in defect_boxes]
        scores, recipe_reason_codes, recipe_confidence = self._recipe_scores(
            recipe,
            label_box=top_label,
            code_boxes=code_boxes,
            defect_boxes=defect_boxes,
            decode_result=decode_result,
        )
        outcome = self.decision_engine.evaluate(
            runtime=runtime,
            label_box=label_box_schema,
            code_boxes=code_box_schemas,
            defect_boxes=defect_box_schemas,
            decode_result=decode_result,
            base_notes=notes,
        )

        result = InspectionResult(
            timestamp=utc_now(),
            image_name=image_name,
            decision=outcome.decision,
            total_confidence=outcome.total_confidence,
            label_box=label_box_schema,
            code_boxes=code_box_schemas,
            defect_boxes=defect_box_schemas,
            decode_result=decode_result,
            is_low_confidence=outcome.is_low_confidence,
            collection_recommended=outcome.collection_recommended,
            collection_reason=outcome.collection_reason,
            runtime=runtime,
            notes=outcome.notes,
            recipe_id=recipe.recipe_id if recipe else None,
            model_confidence=outcome.total_confidence,
            recipe_confidence=recipe_confidence,
            reason_codes=recipe_reason_codes,
            scores=scores,
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
        if bool(self.config.get("output.save_json", True)):
            json_path = out_dir / f"{stem}_{ts}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        if visualization is not None and bool(self.config.get("output.save_visualization", True)):
            write_image(out_dir / f"{stem}_{ts}.jpg", visualization)

        logger.info("Saved result to %s", out_dir)

    def collect_for_training(
        self,
        *,
        image: np.ndarray,
        result: InspectionResult,
        reason: str = "MANUAL",
        visualization: np.ndarray | None = None,
        crop: np.ndarray | None = None,
    ):
        return self.collector.collect(
            image=image,
            result=result,
            reason=reason,
            visualization=visualization,
            crop=crop,
        )
