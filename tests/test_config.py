from pathlib import Path

from edge_inspector.utils.config import load_config


def test_load_config() -> None:
    cfg = load_config(Path("configs/config.example.yaml"))
    assert cfg.get("models.label_model_path") == "weights/label_model.pt"
    assert cfg.get("inference.image_size") == 640