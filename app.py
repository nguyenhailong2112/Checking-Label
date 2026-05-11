from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from edge_inspector.core.inspector import LabelBarcodeInspector
from edge_inspector.training.fine_tune import FineTuneManager, FineTuneRequest
from edge_inspector.utils.config import AppConfig, load_config


def read_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể đọc ảnh đã upload")
    return image


@st.cache_resource
def build_inspector(config: AppConfig) -> LabelBarcodeInspector:
    return LabelBarcodeInspector(config)


def inference_tab(config: AppConfig) -> None:
    uploader = st.file_uploader("Upload ảnh (single/multi)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if not uploader:
        st.info("Vui lòng upload ảnh để bắt đầu.")
        return

    inspector = build_inspector(config)
    for item in uploader:
        st.divider()
        st.subheader(f"Ảnh: {item.name}")
        image = read_image(item)

        result, vis_full, vis_crop = inspector.run(image, image_name=item.name)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(vis_full, cv2.COLOR_BGR2RGB), caption="Visualization - Full Image", use_container_width=True)
        with col2:
            if vis_crop is not None:
                st.image(cv2.cvtColor(vis_crop, cv2.COLOR_BGR2RGB), caption="Visualization - Label Crop", use_container_width=True)

        st.markdown(f"**Decision:** `{result.decision}` | **Confidence:** `{result.total_confidence:.3f}`")
        if result.decode_result.success:
            st.success(f"Decode: {result.decode_result.code_type} - {result.decode_result.decoded_text}")
        else:
            st.warning("Chưa decode được code từ ảnh.")

        st.json(result.model_dump(mode="json"))
        if st.button(f"Save Result - {item.name}"):
            inspector.save_result(result, vis_full)
            st.success("Đã lưu kết quả vào thư mục outputs/")


def finetune_tab(config: AppConfig) -> None:
    st.subheader("Fine-tune trực tiếp trên UI")
    st.caption("Workflow: upload 1-10 ảnh -> gán nhãn -> train nhanh -> cập nhật model")

    model_target = st.selectbox("Target model", ["label", "code", "defect"])
    epochs = st.slider("Epochs", 1, 20, 5)
    images = st.file_uploader("Upload data fine-tune (1-10 ảnh)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="ft_uploader")

    classes_map = FineTuneManager.SUPPORTED_CLASSES[model_target]
    assignments: list[tuple[Any, str]] = []
    for i, up in enumerate(images or []):
        label = st.selectbox(f"Label for {up.name}", classes_map, key=f"cls_{i}_{up.name}")
        assignments.append((up, label))

    if st.button("Train nhanh & cập nhật model"):
        total = len(assignments)
        if total == 0 or total > 10:
            st.error("Số ảnh phải trong khoảng 1-10.")
            return

        manager = FineTuneManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            local_pairs: list[tuple[Path, str]] = []
            for up, label in assignments:
                img_path = tmpdir_path / up.name
                img_path.write_bytes(up.getvalue())
                local_pairs.append((img_path, label))

            base_model = Path(config.get(f"models.{model_target}_model_path"))
            request = FineTuneRequest(
                model_type=model_target,
                image_label_pairs=local_pairs,
                base_model_path=base_model,
                output_dir=Path(config.get("training.output_dir", "runs/fine_tune")),
                epochs=epochs,
                image_size=int(config.get("inference.image_size", 640)),
            )

            try:
                with st.spinner("Đang fine-tune model, vui lòng chờ..."):
                    data_yaml = manager.prepare_dataset(request)
                    best_pt = manager.run_training(request, data_yaml)
                st.success(f"Fine-tune hoàn tất. Best model: {best_pt}")
                config.data["models"][f"{model_target}_model_path"] = str(best_pt)
                st.info("Đã cập nhật model path trong runtime hiện tại. Hãy refresh để áp dụng hoàn toàn.")
            except Exception as exc:
                st.error(f"Fine-tune thất bại: {exc}")


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

    st.title("Edge AI Label & Barcode Inspection")
    st.caption("Pipeline: Label Detection → Crop/Preprocess → Code + Defect + Decode → OK/NG")

    with st.sidebar:
        st.header("Runtime Config")
        conf = st.slider("Confidence", 0.05, 0.95, float(config.get("inference.conf_threshold", 0.25)), 0.05)
        config.data.setdefault("inference", {})["conf_threshold"] = conf

    tab_infer, tab_ft = st.tabs(["Inference", "Fine-tune Model"])
    with tab_infer:
        inference_tab(config)
    with tab_ft:
        finetune_tab(config)


if __name__ == "__main__":
    main()