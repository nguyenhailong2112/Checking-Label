from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from edge_inspector.utils.time import utc_now

TeachTarget = Literal["label", "code", "defect"]
TeachSource = Literal["upload", "camera", "auto_collect"]


class RecipeLabelSettings(BaseModel):
    classes: list[str] = Field(default_factory=lambda: ["Label"])
    expected_roi_xyxy: tuple[int, int, int, int] | None = None
    min_model_conf: float = Field(default=0.25, ge=0.0, le=1.0)
    min_recipe_score: float = Field(default=0.70, ge=0.0, le=1.0)
    min_accept_score: float = Field(default=0.75, ge=0.0, le=1.0)


class RecipeCodeSettings(BaseModel):
    enabled: bool = True
    classes: list[str] = Field(default_factory=lambda: ["Code1D", "Code2D"])
    expected_roi_xyxy: tuple[int, int, int, int] | None = None
    expected_count: int = Field(default=1, ge=0)
    allowed_types: list[str] = Field(default_factory=list)
    pattern: str | None = None
    require_decode: bool = True


class RecipeDefectSettings(BaseModel):
    enabled: bool = True
    classes: list[str] = Field(default_factory=lambda: ["DefectNG"])
    roi_xyxy: tuple[int, int, int, int] | None = None
    reject_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class ProductRecipe(BaseModel):
    recipe_id: str
    product_name: str
    version: int = Field(default=1, ge=1)
    active: bool = True
    label: RecipeLabelSettings = Field(default_factory=RecipeLabelSettings)
    code: RecipeCodeSettings = Field(default_factory=RecipeCodeSettings)
    defect: RecipeDefectSettings = Field(default_factory=RecipeDefectSettings)
    approved_samples: int = Field(default=0, ge=0)
    negative_samples: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None

    @field_validator("recipe_id")
    @classmethod
    def validate_recipe_id(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("recipe_id is required")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        if any(ch not in allowed for ch in normalized):
            raise ValueError("recipe_id may only contain letters, numbers, dash and underscore")
        return normalized

    @field_validator("product_name")
    @classmethod
    def validate_product_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("product_name is required")
        return value.strip()


class ApprovedAnnotation(BaseModel):
    target: TeachTarget
    class_name: str
    xyxy: tuple[int, int, int, int]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    suggested_by_model: bool = False

    @field_validator("xyxy")
    @classmethod
    def validate_xyxy(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        if min(value) < 0:
            raise ValueError("bbox coordinates must be non-negative")
        return value

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("class_name is required")
        return value.strip()


class TeachSession(BaseModel):
    session_id: str
    recipe_id: str
    created_at: datetime = Field(default_factory=utc_now)
    notes: str = ""


class TeachSample(BaseModel):
    sample_id: str
    recipe_id: str
    target: TeachTarget
    image_name: str
    image_path: str
    annotations: list[ApprovedAnnotation]
    negative: bool = False
    source: TeachSource = "upload"
    approved_by: str | None = None
    model_snapshot: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RecipeScore(BaseModel):
    target: TeachTarget
    model_confidence: float = Field(ge=0.0, le=1.0)
    roi_score: float | None = Field(default=None, ge=0.0, le=1.0)
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    pattern_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
