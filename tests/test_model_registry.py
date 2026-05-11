from pathlib import Path

from edge_inspector.core.model_registry import ModelRegistry
from edge_inspector.utils.config import AppConfig


def test_stage_model_artifact_records_manifest(tmp_path: Path) -> None:
    cfg = AppConfig(data={"models": {"deploy_dir": str(tmp_path / "staged")}})
    registry = ModelRegistry(cfg)

    record = registry.stage_artifact(
        target="label",
        source_name="best.pt",
        payload=b"fake model bytes",
        note="unit test",
    )

    assert record.artifact_path.exists()
    assert record.artifact_path.read_bytes() == b"fake model bytes"
    history = registry.list_artifacts("label")
    assert len(history) == 1
    assert history[0].target == "label"
    assert history[0].note == "unit test"


def test_stage_model_artifact_rejects_unknown_suffix(tmp_path: Path) -> None:
    cfg = AppConfig(data={"models": {"deploy_dir": str(tmp_path / "staged")}})
    registry = ModelRegistry(cfg)

    try:
        registry.stage_artifact(target="label", source_name="bad.txt", payload=b"bad")
    except ValueError as exc:
        assert "Unsupported model artifact suffix" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported suffix")