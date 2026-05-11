from edge_inspector.core.decision import DecisionEngine
from edge_inspector.core.schemas import BoundingBox, DecodeResult, RuntimeSettings


def box(confidence: float = 0.9, class_name: str = "x") -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=confidence, class_name=class_name)


def runtime(**overrides) -> RuntimeSettings:
    data = {
        "mode": "full",
        "conf_threshold": 0.25,
        "low_conf_threshold": 0.55,
        "require_code": True,
        "require_decode": True,
        "inspect_defect": True,
    }
    data.update(overrides)
    return RuntimeSettings(**data)


def test_decision_ok_when_required_code_decode_pass_and_no_defect() -> None:
    outcome = DecisionEngine(auto_collect_ng=True, auto_collect_low_conf=True).evaluate(
        runtime=runtime(),
        label_box=box(0.95, "label"),
        code_boxes=[box(0.9, "Code1D")],
        defect_boxes=[],
        decode_result=DecodeResult(success=True, decoded_text="ABC", code_type="QRCODE"),
    )

    assert outcome.decision == "OK"
    assert not outcome.collection_recommended
    assert not outcome.is_low_confidence


def test_decision_ng_when_missing_label() -> None:
    outcome = DecisionEngine(auto_collect_ng=True, auto_collect_low_conf=True).evaluate(
        runtime=runtime(),
        label_box=None,
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
    )

    assert outcome.decision == "NG"
    assert outcome.total_confidence == 0.0
    assert outcome.collection_reason == "NG"
    assert "Không phát hiện label" in outcome.notes


def test_decision_ng_when_required_code_missing() -> None:
    outcome = DecisionEngine().evaluate(
        runtime=runtime(require_decode=False),
        label_box=box(0.9, "label"),
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
    )

    assert outcome.decision == "NG"
    assert "Không phát hiện code" in outcome.notes


def test_decision_ng_when_decode_required_and_fails() -> None:
    outcome = DecisionEngine().evaluate(
        runtime=runtime(require_code=True, require_decode=True),
        label_box=box(0.9, "label"),
        code_boxes=[box(0.9, "Code2D")],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
    )

    assert outcome.decision == "NG"
    assert "Không decode được barcode/QR" in outcome.notes


def test_decision_ng_when_defect_found() -> None:
    outcome = DecisionEngine().evaluate(
        runtime=runtime(require_decode=False),
        label_box=box(0.9, "label"),
        code_boxes=[box(0.9, "Code1D")],
        defect_boxes=[box(0.8, "DefectNG")],
        decode_result=DecodeResult(success=False),
    )

    assert outcome.decision == "NG"
    assert "Phát hiện defect" in outcome.notes


def test_low_confidence_can_recommend_collection_without_ng() -> None:
    outcome = DecisionEngine(auto_collect_ng=False, auto_collect_low_conf=True).evaluate(
        runtime=runtime(mode="label_only", require_code=False, require_decode=False, inspect_defect=False, low_conf_threshold=0.8),
        label_box=box(0.6, "label"),
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
    )

    assert outcome.decision == "OK"
    assert outcome.is_low_confidence
    assert outcome.collection_reason == "LOW_CONFIDENCE"


def test_mode_skips_code_and_defect_rules() -> None:
    outcome = DecisionEngine().evaluate(
        runtime=runtime(mode="label_only", require_code=True, require_decode=True, inspect_defect=True),
        label_box=box(0.95, "label"),
        code_boxes=[],
        defect_boxes=[],
        decode_result=DecodeResult(success=False),
    )

    assert outcome.decision == "OK"
    assert "Không phát hiện code" not in outcome.notes
    assert "Không decode được barcode/QR" not in outcome.notes