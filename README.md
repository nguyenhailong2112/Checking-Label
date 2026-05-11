# Edge AI Label & Barcode Inspection – Hybrid Edition

Hệ thống kiểm tra label, barcode/QR 1D/2D và defect ngoại quan theo kiến trúc **Hybrid Ver 2**:

1. **Jetson/Edge Device**: chạy inference cascaded pipeline tốc độ cao, visualization, logging, JSON output, Active Learning data collection và Quick Teach nhẹ.
2. **PC/Workstation**: annotate data, fine-tune YOLOv11, evaluate model cũ/mới và export `.pt`/TensorRT `.engine` để deploy lại Edge.

Pipeline chính:

1. Label detection
2. Crop & preprocess
3. Code detection + defect detection + barcode decoding
4. Decision engine OK/NG + low-confidence handling
5. JSON output + visualization + collect NG/low-confidence samples

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
mkdir -p weights
# copy model vào weights: label_model.pt, code_model.pt, defect_model.pt
streamlit run app.py
```

## Current audit checkpoint

Checkpoint mới nhất được ghi tại `docs/ProjectAuditCheckpoint.md`. Tóm tắt trạng thái hiện tại:

- Đã có nền tảng Hybrid đúng scope: cascaded inference pipeline, Streamlit UI, Active Learning collection, Quick Teach runtime controls và model registry staging/rollback.
- Repo hiện phù hợp cho demo software skeleton và chuẩn bị test với weights thật.
- Các hạng mục lớn còn thiếu: camera realtime tối ưu, benchmark chính thức trên Jetson/TensorRT, Docker/systemd deployment và validation trên data thực tế.

Xem checklist đầy đủ, gap analysis và roadmap tiếp theo tại `docs/ProjectAuditCheckpoint.md`.

## Scope Ver 2: Edge không fine-tune trực tiếp

Phiên bản hiện tại **không train/fine-tune trực tiếp trên Jetson qua Streamlit**. UI Edge tập trung vào:

- Chạy inference ổn định.
- Chỉnh threshold/mode trong runtime.
- Stage/apply model artifact mới (`.pt` hoặc `.engine`) và lưu manifest rollback nhanh.
- Collect ảnh NG/low-confidence/manual kèm metadata để đưa về PC training.

Quy trình cải tiến model chuẩn:

1. Edge chạy inference bình thường.
2. Operator bấm **Collect for Training** hoặc bật auto-collect NG/low-confidence.
3. Copy `data/collected/` về PC.
4. Annotate bằng Label Studio/Roboflow.
5. Fine-tune YOLOv11 trên PC/workstation.
6. Export `.pt` hoặc TensorRT `.engine`.
7. Copy model mới về Jetson, stage vào `weights/staged/manifest.jsonl` và apply trong tab **Quick Teach / Models**.

## Active Learning data structure

```text
data/collected/
  ng/
    images/
    metadata/
    visualizations/
    crops/
  low_confidence/
    images/
    metadata/
    visualizations/
    crops/
  manual/
    images/
    metadata/
    visualizations/
    crops/
```

Mỗi file metadata JSON lưu `InspectionResult`, reason collect, đường dẫn ảnh gốc, visualization và crop để phục vụ vòng lặp annotate → train → deploy.

## Cấu trúc code

- `src/edge_inspector/core/inspector.py`: Pipeline chính `LabelBarcodeInspector`, inference orchestration và visualization.
- `src/edge_inspector/core/decision.py`: DecisionEngine rule-based OK/NG, confidence, low-confidence và collection recommendation.
- `src/edge_inspector/core/active_learning.py`: Lưu sample NG/low-confidence/manual cho PC training loop.
- `src/edge_inspector/core/models.py`: Wrapper YOLO lazy-load cho `.pt`/`.engine`.
- `src/edge_inspector/core/model_registry.py`: Stage model artifact, ghi manifest và hỗ trợ rollback nhanh trong UI.
- `src/edge_inspector/core/schemas.py`: JSON schema Pydantic (`InspectionResult`, `RuntimeSettings`, `CollectionRecord`).
- `src/edge_inspector/utils/image_ops.py`: Crop, enhance, visualize.
- `src/edge_inspector/utils/config.py`: Load cấu hình YAML.
- `src/edge_inspector/training/fine_tune.py`: Helper fine-tune dùng cho PC/workstation workflow, không expose như Edge UI chính.
- `scripts/benchmark.py`: Benchmark CLI cho latency/FPS trên folder ảnh.
- `scripts/export_tensorrt.py`: Helper export YOLO `.pt` sang TensorRT `.engine`.
- `docs/TrainingWorkflow.md`: Workflow PC annotate/fine-tune/evaluate/export/deploy.
- `app.py`: Streamlit Hybrid UI.


## Benchmark & TensorRT helpers

Benchmark folder ảnh sau khi đã chuẩn bị đủ model:

```bash
python scripts/benchmark.py --config configs/config.yaml --input datasets/test_images --warmup 5 --output-json reports/benchmark.json
```

Export TensorRT engine trên máy có GPU/TensorRT phù hợp:

```bash
python scripts/export_tensorrt.py --model runs/train/label_v2/weights/best.pt --imgsz 640 --device 0 --half
```

Workflow PC training chi tiết nằm tại `docs/TrainingWorkflow.md`.

## Tài liệu chi tiết
- Xem tài liệu đầy đủ tại `docs/UserManual.md`.