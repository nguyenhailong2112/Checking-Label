from edge_inspector.utils.time import utc_now

from edge_inspector.core.schemas import DecodeResult, InspectionResult


def test_inspection_result_schema() -> None:
    result = InspectionResult(
        timestamp=utc_now(),
        image_name="a.jpg",
        decision="OK",
        total_confidence=0.9,
        label_box=None,
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
        notes=[],
        recipe_id="demo_recipe",
        recipe_confidence=0.8,
        reason_codes=["recipe_assisted_label"],
    )
    payload = result.model_dump(mode="json")
    assert payload["decision"] == "OK"
    assert payload["recipe_id"] == "demo_recipe"
    assert payload["reason_codes"] == ["recipe_assisted_label"]
