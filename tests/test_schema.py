from datetime import datetime

from edge_inspector.core.schemas import DecodeResult, InspectionResult


def test_inspection_result_schema() -> None:
    result = InspectionResult(
        timestamp=datetime.utcnow(),
        image_name="a.jpg",
        decision="OK",
        total_confidence=0.9,
        label_box=None,
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
        notes=[],
    )
    payload = result.model_dump(mode="json")
    assert payload["decision"] == "OK"