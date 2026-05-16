from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from edge_inspector.utils.time import utc_now

IdentityStatus = Literal[
    "DISABLED",
    "NO_GALLERY",
    "KNOWN_LABEL",
    "UNKNOWN_LABEL",
    "LOW_CONF_IDENTITY",
    "AMBIGUOUS_LABEL",
]


class IdentityMatch(BaseModel):
    class_name: str
    similarity: float = Field(ge=-1.0, le=1.0)
    exemplar_count: int = Field(default=0, ge=0)


class IdentityPrediction(BaseModel):
    enabled: bool = True
    status: IdentityStatus
    predicted_class: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    margin: float | None = Field(default=None, ge=0.0, le=1.0)
    gallery_id: str | None = None
    matches: list[IdentityMatch] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class IdentityGalleryClass(BaseModel):
    class_name: str
    exemplar_count: int = Field(default=0, ge=0)
    exemplar_paths: list[str] = Field(default_factory=list)
    prototype: list[float] | None = None

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("class_name is required")
        return cleaned


class IdentityGalleryMetadata(BaseModel):
    gallery_id: str = "default"
    encoder_name: str = "histogram_v1"
    embedding_dim: int = Field(default=0, ge=0)
    similarity_metric: Literal["cosine"] = "cosine"
    unknown_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    accept_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    ambiguous_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    classes: dict[str, IdentityGalleryClass] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None


class IdentityEnrollRecord(BaseModel):
    gallery_id: str
    class_name: str
    image_name: str
    exemplar_path: str
    exemplar_count: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)

