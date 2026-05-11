# User Manual – Edge AI Label & Barcode Inspection Hybrid Edition

## 1. Mục tiêu tài liệu

Tài liệu này mô tả cách vận hành hệ thống theo **Scope Ver 2 / Hybrid Edition**. Trọng tâm của phase hiện tại là xây dựng một Edge AI demo ổn định, dễ bàn giao và có vòng lặp cải tiến model rõ ràng:

- Edge Device/Jetson AGX Orin chạy inference, visualization, logging, JSON output, Active Learning và Quick Teach nhẹ.
- PC/Workstation thực hiện annotate, fine-tune, evaluate và export model mới.
- Không fine-tune trực tiếp trên Jetson trong phase này.

## 2. Kiến trúc tổng thể

### 2.1 Edge Device

Edge Device đảm nhiệm:

1. Nhận ảnh từ upload/webcam/camera.
2. Chạy cascaded inference pipeline.
3. Hiển thị ảnh gốc, vùng crop và bounding boxes.
4. Xuất `InspectionResult` JSON.
5. Lưu kết quả OK/NG nếu operator yêu cầu.
6. Collect ảnh NG, low-confidence hoặc manual sample cho vòng lặp training.
7. Quick Teach nhẹ: chỉnh threshold/mode, stage/apply model artifact mới và rollback staged model khi cần.

### 2.2 PC/Workstation

PC/Workstation đảm nhiệm:
1. Copy `data/collected/` từ Edge về.
2. Annotate bbox bằng Label Studio/Roboflow.
3. Fine-tune YOLOv11 cho label/code/defect model.
4. Evaluate model cũ/mới.
5. Export `.pt` hoặc TensorRT `.engine`.
6. Copy model mới về Edge để stage/apply qua manifest `weights/staged/manifest.jsonl`.

## 3. Pipeline xử lý

Pipeline hiện tại là cascaded pipeline:
1. **Label Detection**: detect vùng label, chọn top-1 box confidence cao nhất.
2. **Crop & Preprocess**: crop vùng label, enhance contrast/brightness và sharpen.
3. **Code Inspection**: detect `Code1D`/`Code2D`, optionally decode trên crop của code box tốt nhất.
4. **Defect Inspection**: detect `DefectNG` trên vùng crop label.
5. **Decision Engine**:
   - Không có label → NG.
   - Require code nhưng không detect code → NG.
   - Require decode nhưng decode fail → NG.
   - Có defect → NG.
   - Confidence thấp hơn `inspection.low_conf_threshold` → đánh dấu low-confidence và đề xuất collect.
6. **Output**: trả về JSON Pydantic, visualization full image và crop image.

## 4. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
mkdir -p weights
```

Copy 3 model vào thư mục `weights/`:

```text
weights/label_model.pt
weights/code_model.pt
weights/defect_model.pt
```

Khởi động UI:

```bash
streamlit run app.py
```

## 5. Cấu hình quan trọng

File mẫu nằm tại `configs/config.example.yaml`.

### 5.1 Models

```yaml
models:
  label_model_path: "weights/label_model.pt"
  code_model_path: "weights/code_model.pt"
  defect_model_path: "weights/defect_model.pt"
  deploy_dir: "weights/staged"
```

### 5.2 Inference

```yaml
inference:
  image_size: 640
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_det: 50
  device: "cpu"
```

Trên Jetson có thể đổi `device` theo môi trường Ultralytics/TensorRT thực tế.

### 5.3 Inspection mode

```yaml
inspection:
  mode: "full"
  low_conf_threshold: 0.55
  require_code: true
  require_decode: true
  inspect_defect: true
  decode_on_code_crop: true
```

Các mode hỗ trợ:

- `full`: label + code + decode + defect.
- `label_only`: chỉ kiểm tra label.
- `code_only`: label crop + code/decode.
- `defect_only`: label crop + defect.

### 5.4 Active Learning

```yaml
active_learning:
  save_dir: "data/collected"
  auto_collect_ng: false
  auto_collect_low_conf: false
  save_visualization: true
  save_crop: true
```

Khuyến nghị demo ban đầu: để auto-collect tắt, operator chủ động bấm **Collect for Training**. Khi chạy line/stress test có thể bật auto-collect NG/low-confidence.

## 6. Vận hành UI

### 6.1 Sidebar Runtime Config

Sidebar cho phép chỉnh nhanh:

- Confidence threshold.
- Low-confidence threshold.
- Inspection mode.
- Require code detection.
- Require successful decode.
- Inspect defect.
- Auto collect NG.
- Auto collect low-confidence.

Các thay đổi này áp dụng runtime, không ghi đè file YAML.

### 6.2 Tab Inference & Collect

Thao tác:

1. Upload một hoặc nhiều ảnh.
2. Xem visualization full image và crop label.
3. Xem decision, confidence, decode result và JSON.
4. Bấm **Save Result** nếu muốn lưu output vào `outputs/`.
5. Bấm **Collect for Training** nếu ảnh là case khó/NG/low-confidence.

Nếu bật auto-collect, hệ thống tự lưu sample khi kết quả là NG hoặc low-confidence tùy cấu hình.

### 6.3 Tab Quick Teach / Models

Tab này dùng để:

- Xem trạng thái 3 model hiện tại.
- Upload/stage/apply model mới `.pt` hoặc `.engine` vào `weights/staged/`.
- Ghi manifest `weights/staged/manifest.jsonl` để audit version và rollback nhanh.
- Apply lại staged artifact cũ nếu model mới chưa đạt.
- Cập nhật model path runtime cho session hiện tại.
- Nhắc lại workflow PC fine-tune chuẩn.

Lưu ý: tab này **không train model trực tiếp trên Edge**.

## 7. Active Learning folder structure

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

Metadata JSON chứa:

- `reason`: `NG`, `LOW_CONFIDENCE` hoặc `MANUAL`.
- `source_image_path`.
- `visualization_path`.
- `crop_path`.
- Toàn bộ `InspectionResult`.
- Runtime settings tại thời điểm inference.

## 8. PC fine-tune workflow

1. Copy `data/collected/` từ Jetson về PC.
2. Chọn những sample có giá trị, lọc trùng/lỗi.
3. Annotate bbox bằng Label Studio/Roboflow.
4. Fine-tune YOLOv11 riêng cho từng model:
   - `label_model`
   - `code_model`
   - `defect_model`
5. Evaluate precision/recall/mAP và test thực tế.
6. Export `.pt` hoặc TensorRT `.engine`.
7. Copy model mới về Jetson.
8. Apply trong tab **Quick Teach / Models**.

## 9. Tiêu chí nghiệm thu phase hiện tại

- Pipeline chạy ổn định theo cascaded flow.
- UI đúng Scope Ver 2, không còn fine-tune trực tiếp trên Edge.
- Có decision engine rõ ràng cho label/code/decode/defect.
- Có Active Learning collect NG/low-confidence/manual với metadata đầy đủ.
- Có Quick Teach nhẹ để chỉnh threshold/mode và apply model mới.
- Có test cho schema/config/data collection/fine-tune helper/decision engine/model registry/benchmark summary.

## 10. Ghi chú triển khai Jetson

- Production nên ưu tiên TensorRT `.engine`.
- Benchmark FPS/memory/stability sau khi có model thật.
- USB camera/webcam là input phase đầu; GigE/HIK và PLC để phase sau.
- Nếu dùng `pyzbar`, cần cài thêm thư viện hệ thống `zbar` để decode ổn định.



## 11. Camera, benchmark và TensorRT helper

### 11.1 Camera Snapshot

UI hiện có tab **Camera Snapshot** để chụp một frame từ USB camera/webcam theo camera index rồi chạy pipeline inference giống tab upload ảnh. Đây là bước đầu cho camera mode; realtime loop tối ưu FPS sẽ làm ở phase tiếp theo.

### 11.2 Benchmark CLI

Sau khi có đủ 3 model thật và folder ảnh test, chạy:

```bash
python scripts/benchmark.py --config configs/config.yaml --input datasets/test_images --warmup 5 --output-json reports/benchmark.json
```

Report gồm latency trung bình, P50/P95, FPS average, OK/NG distribution và danh sách latency từng ảnh.

### 11.3 TensorRT export helper

Trên máy có GPU/TensorRT phù hợp, export model:

```bash
python scripts/export_tensorrt.py --model runs/train/label_v2/weights/best.pt --imgsz 640 --device 0 --half
```

Lưu ý: TensorRT `.engine` nên build/test trên đúng Jetson/runtime target vì engine phụ thuộc môi trường phần cứng và TensorRT runtime.

### 11.4 Training workflow chi tiết

Xem `docs/TrainingWorkflow.md` để thực hiện luồng: copy `data/collected/` → annotate Label Studio/Roboflow → fine-tune YOLOv11 → evaluate → export TensorRT → stage/apply model trên Edge.

## 12. Audit checkpoint hiện tại

Tài liệu audit checkpoint đầy đủ nằm tại `docs/ProjectAuditCheckpoint.md`. Đây là tài liệu nên đọc cùng User Manual để biết chính xác hệ thống đã làm được gì, còn thiếu gì và roadmap ưu tiên tiếp theo.

Tóm tắt nhanh:

- Đã hoàn thiện nền tảng Hybrid: Edge inference, Streamlit UI, Active Learning, Quick Teach runtime controls và model staging/rollback.
- Chưa hoàn thiện production Edge deployment: camera realtime tối ưu, TensorRT benchmark chính thức trên Jetson, Docker/systemd, logging rotation và validation với dataset thực tế.
- Hướng triển khai tiếp theo: integration tests với mock YOLO, output folder theo decision, realtime camera loop, model evaluation report và deployment package.