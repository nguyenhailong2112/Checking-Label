from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from edge_inspector.identity.schemas import IdentityPrediction


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = Field(ge=0.0, le=1.0)
    class_name: str


class DecodeResult(BaseModel):
    success: bool
    decoded_text: str | None = None
    code_type: str | None = None


class RuntimeSettings(BaseModel):
    mode: Literal["full", "label_only", "code_only", "defect_only"] = "full"
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    low_conf_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    require_code: bool = True
    require_decode: bool = True
    inspect_defect: bool = True


class InspectionResult(BaseModel):
    timestamp: datetime
    image_name: str
    decision: Literal["OK", "NG"]
    total_confidence: float = Field(ge=0.0, le=1.0)
    label_box: BoundingBox | None
    code_boxes: list[BoundingBox]
    defect_boxes: list[BoundingBox]
    decode_result: DecodeResult
    is_low_confidence: bool = False
    collection_recommended: bool = False
    collection_reason: str | None = None
    runtime: RuntimeSettings | None = None
    notes: list[str] = Field(default_factory=list)
    recipe_id: str | None = None
    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    recipe_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    identity_prediction: IdentityPrediction | None = None


class CollectionRecord(BaseModel):
    timestamp: datetime
    reason: Literal["NG", "LOW_CONFIDENCE", "MANUAL"]
    image_name: str
    source_image_path: str
    visualization_path: str | None = None
    crop_path: str | None = None
    result: InspectionResult
