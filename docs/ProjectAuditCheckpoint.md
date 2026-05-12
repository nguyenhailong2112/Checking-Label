# Project Audit Checkpoint – Edge AI Label & Barcode Inspection Hybrid Edition

Ngày audit: 11/05/2026  
Baseline scope: `ScopeOfWorkVer2.md` – Hybrid Edition  
Mục tiêu checkpoint: định vị trạng thái hiện tại của repo, những gì đã hoàn thành, những gì còn thiếu, rủi ro còn lại và roadmap triển khai tiếp theo.

---

## 1. Executive Summary

Repo hiện tại đã đi đúng hướng **Scope Ver 2 / Hybrid Edition**:

- Edge/Jetson tập trung vào inference, UI, logging/output, Active Learning collection và Quick Teach nhẹ.
- PC/Workstation chịu trách nhiệm annotate, fine-tune, evaluate và export model.
- Không còn định hướng train/fine-tune trực tiếp trên Edge UI.

Mức độ hoàn thiện tổng thể tại checkpoint này: **khoảng 65–70% cho một demo software skeleton đúng kiến trúc**, nhưng mới khoảng **35–45% cho production Edge deployment hoàn chỉnh**, vì chưa có model thật trong repo, chưa có benchmark chính thức trên Jetson, camera hiện mới là snapshot mode, chưa có Docker/systemd và TensorRT export workflow vẫn cần test trên phần cứng đích.

Nói ngắn gọn: **nền móng kiến trúc, UI demo, data collection, model management, DecisionEngine, benchmark CLI, camera snapshot và training/export guide đã có; phần còn lại cần tập trung vào inference thực tế với weights thật, realtime camera, benchmark Jetson, deploy và validation trên data thật.**

---

## 2. Scope Ver 2 Checklist

Legend:

- ✅ Done: đã có code/tài liệu/test ở mức usable.
- 🟡 Partial: đã có skeleton hoặc bản đầu, cần hoàn thiện thêm.
- ❌ Missing: chưa triển khai.
- ⏭️ Later phase: đúng scope nhưng để phase sau.

### 2.1 Edge Device – Jetson / Runtime Inference

| Hạng mục | Trạng thái | Đánh giá hiện tại | Việc cần làm tiếp |
|---|---:|---|---|
| Cascaded pipeline Label → Crop → Code/Defect/Decode | ✅ | `LabelBarcodeInspector` đã chạy theo flow label detect, crop/preprocess, code detect, defect detect, decode và decision. | Test với 3 weights thật và ảnh thực tế. |
| 3 model riêng label/code/defect | ✅ | `YOLOModel` lazy-load 3 model path riêng từ config. | Chuẩn hóa naming/version cho từng model thật. |
| Label top-1 detection | ✅ | Pipeline sort theo confidence và lấy label box cao nhất. | Thêm option reject nếu label class không thuộc whitelist theo product. |
| Crop + preprocess | 🟡 | Có crop, contrast/brightness, sharpen. Config có `deskew` nhưng chưa implement deskew thật. | Implement deskew hoặc bỏ config cho đến khi có thuật toán. |
| Code detection 1D/2D | ✅/🟡 | Có stage code model và boxes. | Validate class names `Code1D`, `Code2D` với weights thật. |
| Barcode/QR decode | 🟡 | Có `pyzbar` decode, fallback khi thiếu zbar/pyzbar. | Decode theo từng code crop, thử nhiều barcode/QR, thêm multi-code result nếu cần. |
| Defect detection single class `DefectNG` | ✅/🟡 | Có stage defect model, decision NG nếu có defect. | Validate confidence threshold riêng cho defect. |
| Decision engine | ✅ | Đã tách `DecisionEngine` riêng và có unit tests cho OK/NG/missing label/missing code/decode fail/defect/low-confidence. | Tiếp tục bổ sung integration tests với mock YOLO predictions. |
| Selective inspection mode | ✅ | Hỗ trợ `full`, `label_only`, `code_only`, `defect_only`. | Thêm UI explain rõ mode nào dùng model nào. |
| Low-confidence handling | ✅ | Có `low_conf_threshold`, flag và collection reason. | Tinh chỉnh metric confidence sau test thực tế. |
| Visualization full + crop | ✅ | Có full image label bbox và crop bbox code/defect. | Cải thiện màu sắc/status overlay cho demo đẹp hơn. |
| Save result JSON + visualization | ✅ | `save_result()` lưu JSON và ảnh visualization. | Tách output OK/NG theo folder nếu cần vận hành line. |
| Active Learning collect NG/Low-conf/Manual | ✅ | Có `ActiveLearningCollector`, metadata JSON, images, crops, visualizations. | Thêm export zip hoặc copy helper cho PC. |
| Auto-collect NG/Low-conf | ✅ | Có config và tránh duplicate trong một Streamlit session. | Cần policy retention/cleanup khi chạy lâu. |
| Quick Teach nhẹ | ✅ | UI chỉnh threshold/mode, stage/apply model mới, rollback staged model. | Persist runtime config ra YAML nếu operator muốn lưu cấu hình. |
| Model registry/version management | ✅ | Có manifest JSONL trong `weights/staged`, list/apply staged artifact. | Thêm model evaluation summary trong manifest. |
| Streamlit UI chạy trên Edge | ✅/🟡 | UI có upload ảnh, inference, collect, quick teach. | Cần camera realtime mode và test thực tế trên Jetson. |
| USB camera/webcam input | 🟡 | Đã có tab Camera Snapshot chụp một frame từ camera index. | Tối ưu realtime loop, start/stop và FPS display ở phase kế tiếp. |
| Performance ≥ 20 FPS trên Jetson | 🟡 | Đã có `scripts/benchmark.py` để đo latency/FPS trên folder ảnh. | Cần weights thật + Jetson + TensorRT engine để lấy số liệu chính thức. |

### 2.2 PC / Training Workstation

| Hạng mục | Trạng thái | Đánh giá hiện tại | Việc cần làm tiếp |
|---|---:|---|---|
| Hướng dẫn annotate Label Studio/Roboflow | ✅ | Đã có `docs/TrainingWorkflow.md` mô tả Label Studio/Roboflow, fine-tune, evaluate, export và deploy. | Bổ sung converter nếu export format thực tế yêu cầu. |
| Fine-tune YOLOv11 trên PC | 🟡 | Có `FineTuneManager` helper, nhưng chỉ tạo dataset mini full-image bbox. | Cần workflow YOLO dataset thật từ annotation tool. |
| Evaluate model cũ/mới | ❌ | Chưa có script compare metrics. | Thêm evaluate script + report mAP/precision/recall/FPS. |
| Export TensorRT engine | 🟡 | Đã có helper `scripts/export_tensorrt.py`. | Cần test trên máy có GPU/TensorRT/Jetson. |
| Deploy model mới về Edge | 🟡 | UI stage/apply artifact có rồi. | Thêm checksum/version/evaluation metadata. |

### 2.3 Deployment / Production Readiness

| Hạng mục | Trạng thái | Đánh giá hiện tại | Việc cần làm tiếp |
|---|---:|---|---|
| Config YAML + Pydantic-like access | ✅ | Có `configs/config.example.yaml`, `configs/config.yaml`, `AppConfig.get()`. | Thêm validation schema cho config để fail-fast. |
| Requirements | ✅/🟡 | Có requirements, dùng `opencv-python-headless`. | Pin version hoặc thêm Jetson-specific requirements nếu cần. |
| Docker | ❌ | Chưa có Dockerfile. | Thêm Dockerfile/devcontainer sau khi chốt runtime deps. |
| systemd service | ❌ | Chưa có service file. | Thêm `deploy/systemd/edge-inspector.service`. |
| Logging | 🟡 | Có Python logger trong inspector, nhưng chưa setup file handler toàn app. | Thêm logging config/file rotation. |
| Tests | 🟡 | Có tests cho config/schema/fine-tune helper/active learning/model registry. | Thêm tests cho decision engine, model wrapper mock, UI helper. |
| CI | ❌ | Chưa có GitHub Actions. | Thêm workflow pytest/compileall. |
| Handover package | 🟡 | README/UserManual đã có, audit doc này bổ sung checkpoint. | Thêm demo script và sample assets không nhạy cảm. |

### 2.4 Out-of-Scope Ver 2

Các hạng mục dưới đây **chưa cần làm ở phase hiện tại** và việc chưa có là đúng scope:

| Hạng mục | Trạng thái | Ghi chú |
|---|---:|---|
| Fine-tune/training trực tiếp trên Jetson | ⏭️ Later / Out of scope | Đã loại khỏi UI Edge. |
| PLC đầy đủ Modbus/EtherNet/IP/OPC UA | ⏭️ Later | Phase sau. |
| Multi-camera đồng thời | ⏭️ Later | Phase sau. |
| LoRA/few-shot trực tiếp trên Edge | ⏭️ Later | Phase sau. |
| Cloud sync tự động | ⏭️ Later | Phase sau. |
| Annotation tool tích hợp hoàn chỉnh trong Streamlit | ⏭️ Later | Dùng Label Studio/Roboflow trên PC. |

---

## 3. Repo Inventory – Đã có gì trong codebase

### 3.1 Application UI

- `app.py`
  - Upload single/multi image.
  - Runtime controls: confidence, low-confidence, mode, require code/decode, inspect defect, auto-collect.
  - Tab `Inference & Collect`.
  - Tab `Quick Teach / Models`.
  - Manual collect and save result actions.
  - Model staging/apply/rollback via `ModelRegistry`.

### 3.2 Core modules

- `src/edge_inspector/core/inspector.py`
  - Main cascaded pipeline.
  - Runtime settings.
  - Decode fallback.
  - Decision rules.
  - Save output and collect hook.

- `src/edge_inspector/core/models.py`
  - YOLO lazy-load wrapper.
  - Validate model path.

- `src/edge_inspector/core/schemas.py`
  - `BoundingBox`, `DecodeResult`, `RuntimeSettings`, `InspectionResult`, `CollectionRecord`.

- `src/edge_inspector/core/active_learning.py`
  - Persist images, visualizations, crops and metadata for training loop.

- `src/edge_inspector/core/model_registry.py`
  - Stage `.pt`/`.engine`, write manifest JSONL, list history, copy to weights path.

### 3.3 Utilities

- `src/edge_inspector/utils/config.py`
  - YAML load and nested config access.

- `src/edge_inspector/utils/image_ops.py`
  - BGR/RGB conversion, image write, crop, enhance, visualize boxes.

- `src/edge_inspector/utils/time.py`
  - Timezone-aware UTC timestamp helper.

### 3.4 Training helper

- `src/edge_inspector/training/fine_tune.py`
  - Fine-tune request dataclass.
  - Dataset preparation helper.
  - Ultralytics train wrapper.

Current limitation: helper này vẫn là PC-side skeleton, chưa thay thế được workflow annotation bbox thật.

### 3.5 Tests

- `tests/test_config.py`
- `tests/test_schema.py`
- `tests/test_fine_tune.py`
- `tests/test_active_learning.py`
- `tests/test_model_registry.py`

Current test coverage is useful for utilities/schemas/workflows, but chưa đủ cho model inference behavior vì chưa có mock YOLO result tests.

---

## 4. Major Gaps / Rủi ro hiện tại

### 4.1 Chưa có model weights thật trong repo/runtime

Do `weights/*.pt` bị ignore và chưa có artifact mẫu, UI sẽ cần operator copy đúng model vào `weights/`. Đây là đúng thực tế vận hành, nhưng demo nội bộ cần checklist chuẩn bị model rõ ràng.

### 4.2 Camera mới ở mức snapshot

UI đã có tab **Camera Snapshot** để chụp một frame từ USB camera/webcam. Để demo giống line thực tế hơn, phase kế tiếp cần:

- Start/stop camera session.
- Realtime loop with FPS display.
- Manual/auto trigger mode.
- Later: GStreamer/GigE/HIK.

### 4.3 Chưa có benchmark chính thức trên Jetson/TensorRT

Success criteria yêu cầu ≥ 20 FPS trên Jetson AGX Orin. Repo đã có benchmark CLI và TensorRT export helper, nhưng hiện chưa có số liệu chính thức vì cần:

- Weights thật.
- TensorRT engine build trên phần cứng đích.
- FPS/memory report.
- Stress test report.

### 4.4 Decision engine cần thêm integration tests

Decision rules đã được tách thành `DecisionEngine` và có unit tests cho các case chính. Việc cần làm tiếp:

- Integration tests với mock YOLO prediction.
- Test inspector orchestration khi model trả về empty/multiple boxes.
- Kiểm thử metric confidence trên data thực tế.

### 4.5 Deskew config chưa có implementation thật

Config có `preprocess.deskew`, nhưng image ops chưa thực hiện deskew. Cần quyết định:

- Implement deskew bằng contour/minAreaRect/Hough, hoặc
- Tạm bỏ config để tránh hiểu nhầm.

### 4.6 Fine-tune helper còn tối giản

`FineTuneManager.prepare_dataset()` hiện tạo bbox full-image, phù hợp prototype rất nhỏ nhưng không phù hợp object detection training nghiêm túc. Workflow chuẩn vẫn phải dựa vào annotation tool tạo YOLO labels thật.

---

## 5. Recommended Next Roadmap

### Sprint 1 – Demo robustness and operator experience

1. ✅ Tách `DecisionEngine` thành module riêng.
2. ✅ Thêm unit tests cho OK/NG/missing label/missing code/decode fail/defect/low-confidence.
3. 🟡 Thêm UI metrics panel: latency per image, model paths, current mode, collected count.
4. Thêm output folder theo decision: `outputs/ok`, `outputs/ng` nếu cần.
5. Thêm sample-check page: kiểm tra model files có tồn tại trước khi inference.

### Sprint 2 – Camera input and benchmark

1. 🟡 Implement camera capture tab hoặc section:
   - ✅ snapshot inference
   - start/stop camera
   - optional realtime inference loop
2. ✅ Thêm benchmark CLI:
   - ✅ input folder
   - ✅ warmup
   - ✅ FPS average/p95
   - memory optional
   - ✅ JSON report
3. Test trên PC trước, sau đó Jetson.

### Sprint 3 – PC training workflow package

1. ✅ Viết `docs/TrainingWorkflow.md`.
2. Thêm script convert/export từ Label Studio/Roboflow nếu cần.
3. Thêm evaluate script so sánh model cũ/mới.
4. ✅ Thêm export TensorRT helper.
5. Gắn evaluation summary vào model registry manifest.

### Sprint 4 – Edge deployment package

1. Dockerfile hoặc Jetson setup guide.
2. systemd service.
3. Logging file handler + rotation.
4. Backup/restore config and model artifacts.
5. Handover checklist.

---

## 6. Acceptance Readiness Snapshot

| Tiêu chí nghiệm thu Ver 2 | Trạng thái hiện tại | Ghi chú |
|---|---:|---|
| Code sạch, modular | 🟡/✅ | Đã tách DecisionEngine và thêm helper scripts; cần tiếp tục tách camera/metrics nếu UI lớn hơn. |
| Pipeline chạy ổn định trên Jetson | ❌ | Chưa test Jetson. |
| UI demo đẹp và dễ dùng | 🟡 | UI có metrics và camera snapshot, cần polish realtime/camera UX. |
| Quy trình collect → PC fine-tune → deploy rõ ràng | 🟡/✅ | Docs và model registry có rồi, thiếu training guide chi tiết. |
| Documentation đầy đủ | 🟡/✅ | README/UserManual/Audit/TrainingWorkflow có rồi, cần Deployment docs. |
| Demo với data thực tế | ❌ | Cần weights và ảnh test thật. |
| ≥ 20 FPS Jetson | ❌ | Có benchmark CLI, nhưng chưa có số liệu Jetson chính thức. |
| Accuracy ≥ 95% test thực tế | ❌ | Cần dataset/test report. |

---

## 7. Kết luận checkpoint

Dự án hiện đã qua giai đoạn “ý tưởng/skeleton rời rạc” và đã có **khung sản phẩm Hybrid đúng hướng**:

- Pipeline core có hình dạng đúng.
- UI vận hành đúng scope Edge inference.
- Active Learning loop đã có nền tảng lưu sample/metadata.
- Quick Teach không train trên Edge mà stage/apply/rollback model artifact.
- Docs đã phản ánh chiến lược Hybrid.

Checkpoint tiếp theo nên tập trung vào **biến skeleton đúng kiến trúc thành demo thực chiến**:

1. Test với weights thật.
2. Chạy benchmark CLI trên PC và Jetson.
3. Nâng Camera Snapshot thành realtime camera mode.
4. Thêm integration tests với mock YOLO predictions.
5. Viết deployment workflow chi tiết.

---

## 8. Scope Ver 2 Plus - Teach Mode Direction

Sau buoi trao doi ngay 12/05/2026, huong phat trien tiep theo duoc chot la **Hybrid Edge AI + Teach Mode / Product Recipe**.

Tai lieu bo sung:

- `docs/ScopeVer2PlusTeachMode.md`
- `docs/TeachModeImplementationCheckpoint.md`

Muc tieu khong phai bien Jetson thanh training server lon. Muc tieu la them mot lop adaptation tai bien:

1. Operator tao recipe cho product/label moi.
2. Operator ve bbox/ROI cho label, code va defect tren 1-20 anh.
3. He thong luu sample approved thanh dataset dung chuan.
4. Runtime ket hop YOLO confidence voi recipe confidence.
5. Code duoc validate bang ROI, decode va pattern.
6. Defect dung ROI, threshold rieng va negative samples.
7. Edge Micro Fine-tune chi la che do nang cao, co benchmark va rollback.

Uu tien code sau checkpoint nay:

1. Them `src/edge_inspector/teach/` cho schema, recipe, dataset va scoring.
2. Them Teach Mode UI MVP de nhap/ve bbox.
3. Them recipe-aware label scoring.
4. Mo rong code inspection sang multi-code/pattern validation.
5. Mo rong defect inspection sang ROI/threshold rieng/negative samples.
6. Sau cung moi them Edge Micro Fine-tune candidate model.
