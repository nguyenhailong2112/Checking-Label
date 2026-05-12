from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from edge_inspector.core.inspector import LabelBarcodeInspector
from edge_inspector.core.model_registry import ModelRegistry
from edge_inspector.teach.dataset import DEFAULT_CLASSES, TeachDatasetWriter
from edge_inspector.teach.recipe import RecipeStore
from edge_inspector.teach.schemas import ApprovedAnnotation, ProductRecipe
from edge_inspector.utils.config import AppConfig, load_config
from edge_inspector.utils.image_ops import bgr_to_rgb, visualize_boxes


MODEL_TARGETS = {
    "label": "models.label_model_path",
    "code": "models.code_model_path",
    "defect": "models.defect_model_path",
}

TARGET_TITLES = {
    "label": "Label",
    "code": "Code 1D/2D",
    "defect": "DefectNG",
}


def render_app_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.35rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.35; }
        div.stButton > button { border-radius: 6px; font-weight: 600; }
        div[data-testid="stAlert"] { border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def missing_model_paths(config: AppConfig) -> list[Path]:
    paths: list[Path] = []
    for key_path in MODEL_TARGETS.values():
        model_path = Path(str(config.get(key_path)))
        if not model_path.exists():
            paths.append(model_path)
    return paths


def render_model_readiness(config: AppConfig) -> None:
    missing = missing_model_paths(config)
    if not missing:
        st.success("Model readiness: all 3 model files are available.")
        return

    st.warning(
        "Model readiness: missing model files. Inference tabs will be available after copying trained models into weights/."
    )
    with st.expander("Missing model files", expanded=False):
        for path in missing:
            st.write(f"- `{path}`")

def read_image(uploaded_file: Any) -> np.ndarray:
    pil_image = Image.open(uploaded_file).convert("RGB")
    return np.asarray(pil_image)[:, :, ::-1].copy()


def option_index(options: list[str], value: str, default: str) -> int:
    if value in options:
        return options.index(value)
    return options.index(default)


def config_signature(config: AppConfig) -> tuple[Any, ...]:
    return (
        config.get("models.label_model_path"),
        config.get("models.code_model_path"),
        config.get("models.defect_model_path"),
        config.get("inference.device"),
        config.get("inference.image_size"),
        config.get("teach.use_recipe"),
        config.get("teach.active_recipe_id"),
    )


@st.cache_resource
def build_inspector(_config: AppConfig, signature: tuple[Any, ...]) -> LabelBarcodeInspector:
    del signature
    return LabelBarcodeInspector(_config)


def count_collected_samples(config: AppConfig) -> int:
    root = Path(str(config.get("active_learning.save_dir", "data/collected")))
    if not root.exists():
        return 0
    return len(list(root.glob("*/metadata/*.json")))


def update_session_metrics(decision: str, elapsed_ms: float) -> None:
    metrics = st.session_state.setdefault(
        "inspection_metrics",
        {"total": 0, "ok": 0, "ng": 0, "latencies_ms": []},
    )
    metrics["total"] += 1
    metrics[decision.lower()] += 1
    metrics["latencies_ms"].append(elapsed_ms)


def render_metrics_panel(config: AppConfig) -> None:
    metrics = st.session_state.setdefault(
        "inspection_metrics",
        {"total": 0, "ok": 0, "ng": 0, "latencies_ms": []},
    )
    latencies = metrics["latencies_ms"]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    last_latency = float(latencies[-1]) if latencies else 0.0

    st.sidebar.divider()
    st.sidebar.header("Runtime Metrics")
    col_total, col_ok, col_ng = st.sidebar.columns(3)
    col_total.metric("Total", metrics["total"])
    col_ok.metric("OK", metrics["ok"])
    col_ng.metric("NG", metrics["ng"])
    col_last, col_avg = st.sidebar.columns(2)
    col_last.metric("Last ms", f"{last_latency:.1f}")
    col_avg.metric("Avg ms", f"{avg_latency:.1f}")
    st.sidebar.metric("Collected", count_collected_samples(config))

    with st.sidebar.expander("Model paths", expanded=False):
        for target, key_path in MODEL_TARGETS.items():
            model_path = Path(str(config.get(key_path)))
            status = "✅" if model_path.exists() else "❌"
            st.write(f"{status} **{target}**: `{model_path}`")
        st.write(f"Mode: `{config.get('inspection.mode', 'full')}`")


def apply_runtime_controls(config: AppConfig) -> None:
    with st.sidebar:
        st.header("Runtime Config")
        st.caption("Quick Teach nhẹ: chỉnh threshold/mode ở runtime, không train trên Edge.")

        conf = st.slider("Confidence threshold", 0.05, 0.95, float(config.get("inference.conf_threshold", 0.25)), 0.05)
        low_conf = st.slider(
            "Low-confidence threshold",
            0.05,
            0.95,
            float(config.get("inspection.low_conf_threshold", 0.55)),
            0.05,
        )
        mode_options = ["full", "label_only", "code_only", "defect_only"]
        mode = st.selectbox(
            "Inspection mode",
            mode_options,
            index=option_index(mode_options, str(config.get("inspection.mode", "full")), "full"),
        )
        require_code = st.checkbox("Require code detection", bool(config.get("inspection.require_code", True)))
        require_decode = st.checkbox("Require successful decode", bool(config.get("inspection.require_decode", True)))
        inspect_defect = st.checkbox("Inspect defect", bool(config.get("inspection.inspect_defect", True)))
        auto_ng = st.checkbox("Auto collect NG", bool(config.get("active_learning.auto_collect_ng", False)))
        auto_low = st.checkbox(
            "Auto collect low-confidence",
            bool(config.get("active_learning.auto_collect_low_conf", False)),
        )
        recipe_store = RecipeStore(config)
        recipes = recipe_store.list_recipes()
        use_recipe = st.checkbox("Use product recipe", bool(config.get("teach.use_recipe", False)))
        active_recipe_id = str(config.get("teach.active_recipe_id", "") or "")
        if recipes:
            recipe_ids = [recipe.recipe_id for recipe in recipes]
            if active_recipe_id not in recipe_ids:
                active_recipe_id = recipe_ids[0]
            active_recipe_id = st.selectbox(
                "Active recipe",
                recipe_ids,
                index=recipe_ids.index(active_recipe_id),
                format_func=lambda item: f"{item}",
            )
        else:
            st.caption("No product recipe yet. Create one in Teach Mode.")

        config.data.setdefault("inference", {})["conf_threshold"] = conf
        config.data.setdefault("inspection", {})["low_conf_threshold"] = low_conf
        config.data.setdefault("inspection", {})["mode"] = mode
        config.data.setdefault("inspection", {})["require_code"] = require_code
        config.data.setdefault("inspection", {})["require_decode"] = require_decode
        config.data.setdefault("inspection", {})["inspect_defect"] = inspect_defect
        config.data.setdefault("active_learning", {})["auto_collect_ng"] = auto_ng
        config.data.setdefault("active_learning", {})["auto_collect_low_conf"] = auto_low
        config.data.setdefault("teach", {})["use_recipe"] = use_recipe
        config.data.setdefault("teach", {})["active_recipe_id"] = active_recipe_id


def render_result_actions(
    *,
    inspector: LabelBarcodeInspector,
    image: np.ndarray,
    result,
    vis_full: np.ndarray,
    vis_crop: np.ndarray | None,
    key_prefix: str,
) -> None:
    col_save, col_collect = st.columns(2)
    with col_save:
        if st.button("💾 Save Result", key=f"save_{key_prefix}"):
            inspector.save_result(result, vis_full)
            st.success("Đã lưu kết quả vào thư mục outputs/")
    with col_collect:
        default_reason = result.collection_reason or ("NG" if result.decision == "NG" else "MANUAL")
        reason = st.selectbox(
            "Collect reason",
            ["MANUAL", "NG", "LOW_CONFIDENCE"],
            index=["MANUAL", "NG", "LOW_CONFIDENCE"].index(default_reason),
            key=f"reason_{key_prefix}",
        )
        if st.button("📥 Collect for Training", key=f"collect_{key_prefix}"):
            record = inspector.collect_for_training(
                image=image,
                result=result,
                reason=reason,
                visualization=vis_full,
                crop=vis_crop,
            )
            st.success(f"Đã collect sample: {record.source_image_path}")


def auto_collect_if_needed(
    *,
    inspector: LabelBarcodeInspector,
    image: np.ndarray,
    result,
    vis_full: np.ndarray,
    vis_crop: np.ndarray | None,
    key_prefix: str,
) -> None:
    if not result.collection_recommended:
        return

    auto_key = f"{key_prefix}:{result.collection_reason}:{result.decision}:{result.total_confidence:.3f}"
    collected_keys = st.session_state.setdefault("auto_collected_keys", set())
    if auto_key in collected_keys:
        st.caption(f"Auto-collect skipped duplicate sample for {key_prefix} in this UI session.")
        return

    record = inspector.collect_for_training(
        image=image,
        result=result,
        reason=result.collection_reason or "MANUAL",
        visualization=vis_full,
        crop=vis_crop,
    )
    collected_keys.add(auto_key)
    st.info(f"Auto-collected sample for {result.collection_reason}: {record.source_image_path}")


def render_inspection_result(
    *,
    inspector: LabelBarcodeInspector,
    image: np.ndarray,
    image_name: str,
    key_prefix: str,
) -> None:
    start = time.perf_counter()
    result, vis_full, vis_crop = inspector.run(image, image_name=image_name)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    update_session_metrics(result.decision, elapsed_ms)

    auto_collect_if_needed(
        inspector=inspector,
        image=image,
        result=result,
        vis_full=vis_full,
        vis_crop=vis_crop,
        key_prefix=key_prefix,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(bgr_to_rgb(vis_full), caption="Visualization - Full Image", use_container_width=True)
    with col2:
        if vis_crop is not None:
            st.image(bgr_to_rgb(vis_crop), caption="Visualization - Label Crop", use_container_width=True)
        else:
            st.warning("Không có label crop để hiển thị.")

    status = "✅ OK" if result.decision == "OK" else "❌ NG"
    low_conf = " | ⚠️ Low-confidence" if result.is_low_confidence else ""
    metric_cols = st.columns(4)
    metric_cols[0].metric("Decision", status)
    metric_cols[1].metric("Confidence", f"{result.total_confidence:.3f}")
    metric_cols[2].metric("Latency", f"{elapsed_ms:.1f} ms")
    metric_cols[3].metric("Mode", result.runtime.mode if result.runtime else "n/a")
    st.markdown(f"**Decision:** `{status}` | **Confidence:** `{result.total_confidence:.3f}`{low_conf}")

    if result.decode_result.success:
        st.success(f"Decode: {result.decode_result.code_type} - {result.decode_result.decoded_text}")
    else:
        st.warning("Chưa decode được code từ ảnh.")
    if result.notes:
        st.warning("; ".join(result.notes))
    if result.recipe_id:
        st.info(
            f"Recipe `{result.recipe_id}` | model={result.model_confidence or 0:.3f} | "
            f"recipe={result.recipe_confidence or 0:.3f}"
        )
        if result.reason_codes:
            st.caption("Reason codes: " + ", ".join(result.reason_codes))

    st.json(result.model_dump(mode="json"))
    render_result_actions(
        inspector=inspector,
        image=image,
        result=result,
        vis_full=vis_full,
        vis_crop=vis_crop,
        key_prefix=key_prefix,
    )


def inference_tab(config: AppConfig) -> None:
    missing = missing_model_paths(config)
    if missing:
        st.error("Chua the chay inference vi thieu model files.")
        for path in missing:
            st.write(f"- `{path}`")
        return
    st.subheader("Inference & Active Learning")
    st.caption("Edge chỉ chạy inference ổn định, lưu sample NG/low-conf để mang về PC annotate/fine-tune.")
    uploader = st.file_uploader("Upload ảnh (single/multi)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if not uploader:
        st.info("Vui lòng upload ảnh để bắt đầu.")
        return

    inspector = build_inspector(config, config_signature(config))
    for idx, item in enumerate(uploader):
        st.divider()
        st.subheader(f"Ảnh: {item.name}")
        image = read_image(item)
        render_inspection_result(
            inspector=inspector,
            image=image,
            image_name=item.name,
            key_prefix=f"upload_{idx}_{item.name}",
        )

def camera_tab(config: AppConfig) -> None:
    missing = missing_model_paths(config)
    if missing:
        st.error("Chua the inspect camera frame vi thieu model files.")
        for path in missing:
            st.write(f"- `{path}`")
        return
    st.subheader("Camera Snapshot")
    st.caption("Chụp một frame từ USB camera/webcam rồi chạy cùng pipeline inference. Realtime loop sẽ được tối ưu ở phase sau.")
    camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    if not st.button("📷 Capture & Inspect"):
        st.info("Bấm Capture & Inspect để lấy snapshot từ camera.")
        return

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        st.error(f"Không import được OpenCV/cv2 cho camera capture: {exc}")
        st.info("Trên Windows hãy đảm bảo requirements đã cài `opencv-python-headless` hoặc package OpenCV phù hợp camera runtime.")
        return

    cap = cv2.VideoCapture(int(camera_index))
    try:
        if not cap.isOpened():
            st.error(f"Không mở được camera index {camera_index}.")
            return
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        st.error("Không đọc được frame từ camera.")
        return

    st.image(bgr_to_rgb(frame), caption="Captured frame", use_container_width=True)
    inspector = build_inspector(config, config_signature(config))
    render_inspection_result(
        inspector=inspector,
        image=frame,
        image_name=f"camera_{int(camera_index)}.jpg",
        key_prefix=f"camera_{int(camera_index)}_{int(time.time())}",
    )


def class_names_for_recipe(recipe: ProductRecipe, target: str) -> list[str]:
    if target == "label":
        return recipe.label.classes or DEFAULT_CLASSES["label"]
    if target == "code":
        return recipe.code.classes or DEFAULT_CLASSES["code"]
    return recipe.defect.classes or DEFAULT_CLASSES["defect"]


def update_recipe_classes(recipe: ProductRecipe, target: str, class_name: str) -> ProductRecipe:
    if target == "label" and class_name not in recipe.label.classes:
        recipe.label.classes.append(class_name)
    elif target == "code" and class_name not in recipe.code.classes:
        recipe.code.classes.append(class_name)
    elif target == "defect" and class_name not in recipe.defect.classes:
        recipe.defect.classes.append(class_name)
    return recipe


def update_recipe_roi(recipe: ProductRecipe, target: str, xyxy: tuple[int, int, int, int]) -> ProductRecipe:
    if target == "label" and recipe.label.expected_roi_xyxy is None:
        recipe.label.expected_roi_xyxy = xyxy
    elif target == "code" and recipe.code.expected_roi_xyxy is None:
        recipe.code.expected_roi_xyxy = xyxy
    elif target == "defect" and recipe.defect.roi_xyxy is None:
        recipe.defect.roi_xyxy = xyxy
    return recipe


def preview_annotation(image: np.ndarray, annotation: ApprovedAnnotation | None) -> np.ndarray:
    if annotation is None:
        return image
    box = {
        "xyxy": annotation.xyxy,
        "confidence": annotation.confidence or 1.0,
        "class_name": annotation.class_name,
    }
    return visualize_boxes(image, [box], (0, 180, 255), annotation.target.upper())


def render_recipe_create_form(store: RecipeStore) -> None:
    with st.expander("Create new product recipe", expanded=False):
        recipe_id = st.text_input("Recipe id", value="default_label_recipe")
        product_name = st.text_input("Product name", value="Default Label")
        if st.button("Create recipe"):
            try:
                recipe = store.create(recipe_id=recipe_id, product_name=product_name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(f"Created recipe: {recipe.recipe_id}")
                st.rerun()


def teach_mode_tab(config: AppConfig) -> None:
    st.subheader("Teach Mode / Product Recipe")
    st.caption("Tao recipe, approve bbox/ROI va luu dataset YOLO dung chuan ngay tai Edge.")

    store = RecipeStore(config)
    writer = TeachDatasetWriter(config)
    render_recipe_create_form(store)
    recipes = store.list_recipes()
    if not recipes:
        st.info("Chua co recipe. Hay tao recipe dau tien de bat dau Teach Mode.")
        return

    selected_recipe_id = st.selectbox(
        "Active recipe",
        [recipe.recipe_id for recipe in recipes],
        format_func=lambda item: f"{item} - {next(r.product_name for r in recipes if r.recipe_id == item)}",
    )
    recipe = store.load(selected_recipe_id)

    col_info, col_counts = st.columns([2, 1])
    with col_info:
        st.write(f"Product: `{recipe.product_name}`")
        st.write(f"Version: `{recipe.version}`")
    with col_counts:
        st.metric("Approved", recipe.approved_samples)
        st.metric("Negative", recipe.negative_samples)

    target = st.segmented_control(
        "Teach target",
        ["label", "code", "defect"],
        format_func=lambda item: TARGET_TITLES[item],
        default="label",
    )
    if target is None:
        target = "label"
    target = str(target)

    uploaded = st.file_uploader(
        "Upload teach image",
        type=["png", "jpg", "jpeg"],
        key=f"teach_upload_{selected_recipe_id}_{target}",
    )
    if uploaded is None:
        st.info("Upload mot anh teach, sau do nhap bbox/ROI va bam Save approved sample.")
        return

    image = read_image(uploaded)
    height, width = image.shape[:2]
    classes = class_names_for_recipe(recipe, target)
    class_name = st.selectbox("Class name", classes, index=0)
    custom_class = st.text_input("Custom class name", value="")
    if custom_class.strip():
        class_name = custom_class.strip()

    negative = st.checkbox("Mark as negative sample (no object/defect to train as positive)", value=False)
    annotation: ApprovedAnnotation | None = None
    if not negative:
        st.write("Bounding box / ROI")
        c1, c2, c3, c4 = st.columns(4)
        x1 = c1.number_input("x1", min_value=0, max_value=max(0, width - 1), value=0, step=1)
        y1 = c2.number_input("y1", min_value=0, max_value=max(0, height - 1), value=0, step=1)
        x2 = c3.number_input("x2", min_value=1, max_value=width, value=width, step=1)
        y2 = c4.number_input("y2", min_value=1, max_value=height, value=height, step=1)
        try:
            annotation = ApprovedAnnotation(
                target=target,  # type: ignore[arg-type]
                class_name=class_name,
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
        except ValueError as exc:
            st.error(str(exc))

    st.image(
        bgr_to_rgb(preview_annotation(image, annotation)),
        caption=f"Teach preview - {TARGET_TITLES[target]}",
        use_container_width=True,
    )

    if st.button("Save approved teach sample"):
        if annotation is None and not negative:
            st.error("BBox chua hop le, chua the luu sample.")
            return
        annotations = [] if negative else [annotation]  # type: ignore[list-item]
        try:
            sample = writer.save_sample(
                image=image,
                image_name=uploaded.name,
                recipe_id=recipe.recipe_id,
                target=target,  # type: ignore[arg-type]
                annotations=annotations,
                class_names=classes,
                negative=negative,
                source="upload",
                model_snapshot={
                    "label": str(config.get("models.label_model_path")),
                    "code": str(config.get("models.code_model_path")),
                    "defect": str(config.get("models.defect_model_path")),
                },
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        if not negative and annotation is not None:
            recipe = update_recipe_classes(recipe, target, annotation.class_name)
            recipe = update_recipe_roi(recipe, target, annotation.xyxy)
            recipe.approved_samples += 1
        else:
            recipe.negative_samples += 1
        store.save(recipe)
        st.success(f"Saved teach sample: {sample.sample_id}")
        st.info("Dataset YOLO da duoc cap nhat trong data/teach/exports/.")


def quick_teach_tab(config: AppConfig) -> None:
    st.subheader("Quick Teach & Model Management")
    st.caption("Theo Scope Ver 2: chỉ chỉnh threshold/mode và apply model mới; không fine-tune trực tiếp trên Jetson.")
    st.markdown("### Model hiện tại")
    for target, key_path in MODEL_TARGETS.items():
        model_path = Path(str(config.get(key_path)))
        exists = "✅" if model_path.exists() else "❌"
        st.write(f"{exists} **{target}**: `{model_path}`")
    registry = ModelRegistry(config)
    st.markdown("### Apply model artifact mới")
    target = st.selectbox("Target model", list(MODEL_TARGETS))
    uploaded_model = st.file_uploader(
        "Upload/copy model mới (.pt hoặc .engine)",
        type=["pt", "engine"],
        key="model_artifact_uploader",
    )
    note = st.text_input("Version note", value="Validated on PC/workstation")
    if uploaded_model is not None and st.button("Stage & apply model vào runtime"):
        try:
            record = registry.stage_artifact(
                target=target,
                source_name=uploaded_model.name,
                payload=uploaded_model.getvalue(),
                note=note,
            )

        except ValueError as exc:
            st.error(str(exc))
        else:
            config.data.setdefault("models", {})[f"{target}_model_path"] = str(record.artifact_path)
            build_inspector.clear()
            st.success(f"Đã stage model mới cho {target}: {record.artifact_path}")
            st.info("Model path đã cập nhật trong runtime. Nếu đang chạy camera/loop, hãy restart session để reload sạch.")

    st.markdown("### Staged model history / rollback nhanh")
    history = registry.list_artifacts(target)
    if history:
        latest_first = list(reversed(history))
        selected = st.selectbox(
            "Chọn staged artifact để apply lại",
            latest_first,
            format_func=lambda item: f"{item.timestamp} | {item.source_name} | {item.artifact_path}",
        )
        if st.button("Apply selected staged model"):
            config.data.setdefault("models", {})[f"{target}_model_path"] = str(selected.artifact_path)
            build_inspector.clear()
            st.success(f"Đã apply lại model staged: {selected.artifact_path}")
    else:
        st.info("Chưa có staged artifact cho target này.")

    st.markdown("### Quy trình PC fine-tune chuẩn")
    st.write(
        "1. Copy `data/collected/` từ Jetson về PC.\n"
        "2. Annotate bằng Label Studio/Roboflow.\n"
        "3. Fine-tune YOLOv11 trên PC/workstation.\n"
        "4. Export `.engine` TensorRT hoặc `.pt` đã validate.\n"
        "5. Copy model mới vào Jetson và apply tại tab này."
    )


def main() -> None:
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        st.error("Thiếu configs/config.yaml. Hãy copy từ config.example.yaml")
        st.stop()

    config = load_config(config_path)
    st.set_page_config(
        page_title=config.get("ui.page_title", "Edge AI Inspection"),
        page_icon=config.get("ui.page_icon", "🧠"),
        layout="wide",
    )
    render_app_style()

    st.title("Edge AI Label & Barcode Inspection – Hybrid Edition")
    st.caption("Pipeline: Label Detection → Crop/Preprocess → Code + Defect + Decode → OK/NG")
    render_model_readiness(config)
    apply_runtime_controls(config)
    render_metrics_panel(config)

    tab_infer, tab_camera, tab_teach, tab_qt = st.tabs(
        ["Inference & Collect", "Camera Snapshot", "Teach Mode", "Quick Teach / Models"]
    )
    with tab_infer:
        inference_tab(config)
    with tab_camera:
        camera_tab(config)
    with tab_teach:
        teach_mode_tab(config)
    with tab_qt:
        quick_teach_tab(config)


if __name__ == "__main__":
    main()
