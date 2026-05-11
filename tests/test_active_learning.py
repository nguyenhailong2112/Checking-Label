from pathlib import Path

import numpy as np

from edge_inspector.utils.time import utc_now

from edge_inspector.core.active_learning import ActiveLearningCollector
from edge_inspector.core.schemas import DecodeResult, InspectionResult
from edge_inspector.utils.config import AppConfig


def test_collect_for_training_writes_image_and_metadata(tmp_path: Path) -> None:
    cfg = AppConfig(
        data={
            "active_learning": {
                "save_dir": str(tmp_path / "collected"),
                "save_visualization": True,
                "save_crop": True,
            }
        }
    )
    result = InspectionResult(
        timestamp=utc_now(),
        image_name="sample.jpg",
        decision="NG",
        total_confidence=0.4,
        label_box=None,
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
        is_low_confidence=True,
        collection_recommended=True,
        collection_reason="NG",
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    record = ActiveLearningCollector(cfg).collect(
        image=image,
        result=result,
        reason="NG",
        visualization=image,
        crop=image,
    )

    assert Path(record.source_image_path).exists()
    assert record.visualization_path is not None and Path(record.visualization_path).exists()
    assert record.crop_path is not None and Path(record.crop_path).exists()
    metadata_files = list((tmp_path / "collected" / "ng" / "metadata").glob("*.json"))
    assert len(metadata_files) == 1
    assert '"reason": "NG"' in metadata_files[0].read_text(encoding="utf-8")