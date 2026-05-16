from pathlib import Path
from uuid import uuid4

import numpy as np

from edge_inspector.identity.encoder import HistogramIdentityEncoder
from edge_inspector.identity.gallery import IdentityGalleryStore
from edge_inspector.identity.inference import IdentityRecognizer


def solid_bgr(color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def workspace_tmp_root() -> Path:
    root = Path(".tmp_identity_tests") / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_histogram_encoder_returns_normalized_embedding() -> None:
    encoder = HistogramIdentityEncoder(image_size=64, hist_bins=4)

    embedding = encoder.encode(solid_bgr((0, 0, 255)))

    assert embedding.shape == (encoder.embedding_dim,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_gallery_store_adds_exemplar_and_persists_metadata() -> None:
    encoder = HistogramIdentityEncoder(image_size=64, hist_bins=4)
    store = IdentityGalleryStore(root=workspace_tmp_root() / "galleries", gallery_id="demo")
    image = solid_bgr((0, 0, 255))

    record = store.add_exemplar(
        class_name="RedLabel",
        image=image,
        image_name="red.jpg",
        embedding=encoder.encode(image),
        encoder_name=encoder.name,
        unknown_threshold=0.7,
        accept_threshold=0.8,
        ambiguous_margin=0.05,
    )
    metadata = store.load()

    assert Path(record.exemplar_path).exists()
    assert metadata.classes["RedLabel"].exemplar_count == 1
    assert metadata.embedding_dim == encoder.embedding_dim


def test_gallery_store_saves_pending_unknown() -> None:
    store = IdentityGalleryStore(root=workspace_tmp_root() / "galleries", gallery_id="demo")
    prediction = IdentityRecognizer(
        encoder=HistogramIdentityEncoder(image_size=64, hist_bins=4),
        gallery_store=store,
    ).predict(solid_bgr((0, 255, 0)))

    metadata_path = store.save_pending_unknown(
        image=solid_bgr((0, 255, 0)),
        image_name="unknown.jpg",
        prediction=prediction,
    )

    assert metadata_path.exists()
    assert metadata_path.parent.name == "metadata"


def test_identity_recognizer_matches_known_and_rejects_unknown() -> None:
    encoder = HistogramIdentityEncoder(image_size=64, hist_bins=4)
    store = IdentityGalleryStore(root=workspace_tmp_root() / "galleries", gallery_id="demo")
    red = solid_bgr((0, 0, 255))
    green = solid_bgr((0, 255, 0))
    store.add_exemplar(
        class_name="RedLabel",
        image=red,
        image_name="red.jpg",
        embedding=encoder.encode(red),
        encoder_name=encoder.name,
        unknown_threshold=0.9,
        accept_threshold=0.95,
        ambiguous_margin=0.01,
    )
    recognizer = IdentityRecognizer(
        encoder=encoder,
        gallery_store=store,
        unknown_threshold=0.9,
        accept_threshold=0.95,
        ambiguous_margin=0.01,
    )

    known = recognizer.predict(red)
    unknown = recognizer.predict(green)

    assert known.status == "KNOWN_LABEL"
    assert known.predicted_class == "RedLabel"
    assert unknown.status == "UNKNOWN_LABEL"
    assert unknown.predicted_class is None
