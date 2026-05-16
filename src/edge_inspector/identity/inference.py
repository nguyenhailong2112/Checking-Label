from __future__ import annotations

import numpy as np

from typing import Any

from edge_inspector.identity.encoder import normalize_embedding
from edge_inspector.identity.gallery import IdentityGalleryStore
from edge_inspector.identity.schemas import IdentityMatch, IdentityPrediction


class IdentityRecognizer:
    def __init__(
        self,
        *,
        encoder: Any,
        gallery_store: IdentityGalleryStore,
        unknown_threshold: float = 0.72,
        accept_threshold: float = 0.82,
        ambiguous_margin: float = 0.05,
        top_k: int = 3,
    ) -> None:
        self.encoder = encoder
        self.gallery_store = gallery_store
        self.unknown_threshold = unknown_threshold
        self.accept_threshold = accept_threshold
        self.ambiguous_margin = ambiguous_margin
        self.top_k = top_k

    def predict(self, image: np.ndarray) -> IdentityPrediction:
        metadata = self.gallery_store.load()
        class_names, prototypes, counts = self.gallery_store.prototypes()
        threshold = float(metadata.unknown_threshold or self.unknown_threshold)
        accept_threshold = float(metadata.accept_threshold or self.accept_threshold)
        ambiguous_margin = float(metadata.ambiguous_margin or self.ambiguous_margin)
        if not class_names:
            return IdentityPrediction(
                status="NO_GALLERY",
                threshold=threshold,
                margin=ambiguous_margin,
                gallery_id=metadata.gallery_id,
                reason_codes=["identity_gallery_empty"],
            )

        query = normalize_embedding(self.encoder.encode(image))
        similarities = prototypes @ query
        order = np.argsort(-similarities)[: self.top_k]
        matches = [
            IdentityMatch(
                class_name=class_names[int(idx)],
                similarity=float(similarities[int(idx)]),
                exemplar_count=counts[int(idx)],
            )
            for idx in order
        ]
        best = matches[0]
        second_similarity = matches[1].similarity if len(matches) > 1 else None
        top_margin = best.similarity - second_similarity if second_similarity is not None else None

        reason_codes = ["identity_matched"]
        predicted_class = best.class_name
        confidence = max(0.0, min(1.0, float(best.similarity)))
        if confidence < threshold:
            status = "UNKNOWN_LABEL"
            predicted_class = None
            reason_codes.append("identity_unknown")
        elif top_margin is not None and top_margin < ambiguous_margin:
            status = "AMBIGUOUS_LABEL"
            reason_codes.append("identity_ambiguous")
        elif confidence < accept_threshold:
            status = "LOW_CONF_IDENTITY"
            reason_codes.append("identity_low_confidence")
        else:
            status = "KNOWN_LABEL"
            reason_codes.append("identity_known")

        return IdentityPrediction(
            status=status,
            predicted_class=predicted_class,
            confidence=confidence,
            threshold=threshold,
            margin=top_margin,
            gallery_id=metadata.gallery_id,
            matches=matches,
            reason_codes=reason_codes,
        )
