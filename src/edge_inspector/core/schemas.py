from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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


class InspectionResult(BaseModel):
    timestamp: datetime
    image_name: str
    decision: Literal["OK", "NG"]
    total_confidence: float = Field(ge=0.0, le=1.0)
    label_box: BoundingBox | None
    code_boxes: list[BoundingBox]
    defect_boxes: list[BoundingBox]
    decode_result: DecodeResult
    notes: list[str] = Field(default_factory=list)