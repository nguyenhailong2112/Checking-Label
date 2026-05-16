from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edge_inspector.identity.encoder import normalize_embedding
from edge_inspector.identity.schemas import (
    IdentityEnrollRecord,
    IdentityGalleryClass,
    IdentityGalleryMetadata,
    IdentityPrediction,
)
from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.image_ops import write_image
from edge_inspector.utils.time import utc_now


class IdentityGalleryStore:
    """Persist few-shot label identity exemplars and class prototypes."""

    def __init__(self, config: AppConfig | None = None, root: str | Path | None = None, gallery_id: str | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        elif config is not None:
            self.root = Path(str(config.get("identity.gallery_dir", "data/identity/galleries")))
        else:
            self.root = Path("data/identity/galleries")
        self.gallery_id = gallery_id or (str(config.get("identity.gallery_id", "default")) if config else "default")

    @property
    def gallery_dir(self) -> Path:
        return self.root / self.gallery_id

    @property
    def metadata_path(self) -> Path:
        return self.gallery_dir / "gallery.json"

    def exists(self) -> bool:
        return self.metadata_path.exists()

    def load(self) -> IdentityGalleryMetadata:
        if not self.metadata_path.exists():
            return IdentityGalleryMetadata(gallery_id=self.gallery_id)
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return IdentityGalleryMetadata(**payload)

    def save(self, metadata: IdentityGalleryMetadata) -> Path:
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        metadata = metadata.model_copy(update={"updated_at": utc_now()})
        self.metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.metadata_path

    def add_exemplar(
        self,
        *,
        class_name: str,
        image: np.ndarray,
        image_name: str,
        embedding: np.ndarray,
        encoder_name: str,
        unknown_threshold: float,
        accept_threshold: float,
        ambiguous_margin: float,
    ) -> IdentityEnrollRecord:
        metadata = self.load()
        embedding = normalize_embedding(embedding)
        if metadata.embedding_dim not in {0, int(embedding.size)}:
            raise ValueError(
                f"Gallery embedding dim {metadata.embedding_dim} does not match encoder dim {embedding.size}"
            )

        class_dir = self.gallery_dir / "exemplars" / _safe_name(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        sample_id = f"{Path(image_name).stem}_{utc_now().strftime('%Y%m%d_%H%M%S_%f')}"
        exemplar_path = class_dir / f"{sample_id}.jpg"
        write_image(exemplar_path, image)

        current = metadata.classes.get(class_name)
        if current is None:
            current = IdentityGalleryClass(class_name=class_name)
        old_count = current.exemplar_count
        if current.prototype is None or old_count == 0:
            new_prototype = embedding
        else:
            old_prototype = np.asarray(current.prototype, dtype=np.float32)
            new_prototype = normalize_embedding((old_prototype * old_count + embedding) / float(old_count + 1))

        current.exemplar_count = old_count + 1
        current.exemplar_paths.append(str(exemplar_path))
        current.prototype = new_prototype.astype(float).tolist()
        metadata.classes[class_name] = current
        metadata.encoder_name = encoder_name
        metadata.embedding_dim = int(embedding.size)
        metadata.unknown_threshold = unknown_threshold
        metadata.accept_threshold = accept_threshold
        metadata.ambiguous_margin = ambiguous_margin
        self.save(metadata)

        return IdentityEnrollRecord(
            gallery_id=metadata.gallery_id,
            class_name=class_name,
            image_name=image_name,
            exemplar_path=str(exemplar_path),
            exemplar_count=current.exemplar_count,
        )

    def save_pending_unknown(
        self,
        *,
        image: np.ndarray,
        image_name: str,
        prediction: IdentityPrediction,
    ) -> Path:
        timestamp = utc_now()
        sample_id = f"{Path(image_name).stem}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        pending_root = self.root.parent / "pending_unknown"
        images_dir = pending_root / "images"
        metadata_dir = pending_root / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"{sample_id}.jpg"
        metadata_path = metadata_dir / f"{sample_id}.json"
        write_image(image_path, image)
        metadata_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp.isoformat(),
                    "image_name": image_name,
                    "crop_path": str(image_path),
                    "prediction": prediction.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return metadata_path

    def prototypes(self) -> tuple[list[str], np.ndarray, list[int]]:
        metadata = self.load()
        class_names: list[str] = []
        prototypes: list[np.ndarray] = []
        counts: list[int] = []
        for class_name, item in sorted(metadata.classes.items()):
            if item.prototype is None or item.exemplar_count <= 0:
                continue
            class_names.append(class_name)
            prototypes.append(normalize_embedding(np.asarray(item.prototype, dtype=np.float32)))
            counts.append(item.exemplar_count)
        if not prototypes:
            return [], np.empty((0, 0), dtype=np.float32), []
        return class_names, np.stack(prototypes).astype(np.float32), counts


def _safe_name(value: str) -> str:
    cleaned = value.strip().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    return "".join(ch if ch in allowed else "_" for ch in cleaned) or "class"
