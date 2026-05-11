from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from edge_inspector.core.schemas import BoundingBox, DecodeResult, RuntimeSettings

DecisionLabel = Literal["OK", "NG"]
CollectionReason = Literal["NG", "LOW_CONFIDENCE"]


@dataclass(frozen=True)
class DecisionOutcome:
    decision: DecisionLabel
    total_confidence: float
    is_low_confidence: bool
    collection_recommended: bool
    collection_reason: CollectionReason | None
    notes: list[str] = field(default_factory=list)


class DecisionEngine:
    """Rule-based OK/NG decision layer for the cascaded inspection pipeline."""

    def __init__(self, *, auto_collect_ng: bool = True, auto_collect_low_conf: bool = True) -> None:
        self.auto_collect_ng = auto_collect_ng
        self.auto_collect_low_conf = auto_collect_low_conf

    def evaluate(
        self,
        *,
        runtime: RuntimeSettings,
        label_box: BoundingBox | None,
        code_boxes: list[BoundingBox],
        defect_boxes: list[BoundingBox],
        decode_result: DecodeResult,
        base_notes: list[str] | None = None,
    ) -> DecisionOutcome:
        notes = list(base_notes or [])
        if label_box is None:
            if "Không phát hiện label" not in notes:
                notes.append("Không phát hiện label")
            return DecisionOutcome(
                decision="NG",
                total_confidence=0.0,
                is_low_confidence=True,
                collection_recommended=self.auto_collect_ng,
                collection_reason="NG" if self.auto_collect_ng else None,
                notes=notes,
            )

        run_code = runtime.mode in {"full", "code_only"}
        run_defect = runtime.mode in {"full", "defect_only"} and runtime.inspect_defect

        decision: DecisionLabel = "OK"
        if run_code and runtime.require_code and not code_boxes:
            decision = "NG"
            notes.append("Không phát hiện code")
        if run_code and runtime.require_decode and not decode_result.success:
            decision = "NG"
            notes.append("Không decode được barcode/QR")
        if run_defect and defect_boxes:
            decision = "NG"
            notes.append("Phát hiện defect")

        total_confidence = self._total_confidence(label_box, code_boxes, defect_boxes, run_defect=run_defect)
        is_low_confidence = total_confidence < runtime.low_conf_threshold
        if is_low_confidence:
            notes.append(f"Low confidence: {total_confidence:.3f} < {runtime.low_conf_threshold:.3f}")

        collection_recommended, collection_reason = self._collection_recommendation(decision, is_low_confidence)
        return DecisionOutcome(
            decision=decision,
            total_confidence=total_confidence,
            is_low_confidence=is_low_confidence,
            collection_recommended=collection_recommended,
            collection_reason=collection_reason,
            notes=notes,
        )

    def _collection_recommendation(
        self,
        decision: DecisionLabel,
        is_low_confidence: bool,
    ) -> tuple[bool, CollectionReason | None]:
        if decision == "NG" and self.auto_collect_ng:
            return True, "NG"
        if is_low_confidence and self.auto_collect_low_conf:
            return True, "LOW_CONFIDENCE"
        return False, None

    @staticmethod
    def _total_confidence(
        label_box: BoundingBox,
        code_boxes: list[BoundingBox],
        defect_boxes: list[BoundingBox],
        *,
        run_defect: bool,
    ) -> float:
        confidence_values = [label_box.confidence]
        confidence_values.extend(box.confidence for box in code_boxes)
        if defect_boxes:
            confidence_values.append(1.0 - min(1.0, max(box.confidence for box in defect_boxes)))
        elif run_defect:
            confidence_values.append(1.0)
        total_conf = float(np.mean(confidence_values)) if confidence_values else 0.0
        return max(0.0, min(1.0, total_conf))