EDGE AI FOR LABEL & BARCODE 1D, 2D & DEFECT DETECTION & INSPECTION SYSTEM
Tài liệu Dự án & Hướng dẫn Triển khai Toàn diện

Ngày: 11 tháng 05 năm 2026
Tác giả: Nguyễn Hải Long - R&D Vision Engineer

Mục đích: Làm tài liệu hướng dẫn chi tiết Scope Of Work, System Prompt cho Codex, và roadmap thực thi dự án từ đầu đến cuối.
1. Giới thiệu Dự án
1.1. Bối cảnh
Các thiết bị thương mại (Cognex, Datalogic, Zebra, Honeywell…) đang sử dụng smart camera + model AI nhúng tại biên để thực hiện detect label, check defect và đọc barcode/QR với tốc độ cao và độ chính xác vượt trội. Dự án này nhằm tự triển khai một giải pháp Edge AI tương đương, có khả năng tùy chỉnh cao, chi phí hợp lý và dễ dàng mở rộng.
1.2. Mục tiêu chính

Xây dựng hệ thống xử lý hoàn toàn tại biên (Edge AI).
Đạt độ chính xác cao và tốc độ thực tế (target ≥ 20–30 FPS trên Jetson AGX Orin).
Demo đẹp mắt, trực quan trên PC trước, sau đó deploy lên Jetson AGX Orin 32GB.
Hệ thống đơn giản, dễ sử dụng, dễ bàn giao cho người dùng cuối (“một phát ăn ngay”).
Hỗ trợ fine-tuning nhanh trên dữ liệu mới (1–10 ảnh) qua UI.

1.3. Phạm vi (Scope of Work)
Trong phạm vi:

Detect label → Crop tự động → Preprocess
Detect & Decode code 1D/2D
Detect defect (OK/NG)
Visualization + JSON output
Web UI demo (Streamlit)
Deploy lên Jetson AGX Orin
Logging & báo cáo kết quả

Ngoài phạm vi giai đoạn này:

Kết nối PLC đầy đủ (Modbus, Ethernet/IP…) – sẽ làm sau nếu cần
Multi-camera đồng thời
Training model từ đầu trên Edge (chỉ fine-tune)

2. Kiến trúc Hệ thống (Cascaded Pipeline)
Luồng xử lý bắt buộc:

Input Acquisition
Ảnh từ camera (webcam cho demo, sau là USB/GigE/HIK)

Stage 1: Label Detection
Model: YOLOv11s (label_model.pt)
Detect toàn bộ vùng label
Chọn box có confidence cao nhất (hoặc xử lý multi-label)

Stage 2: Crop & Preprocess
Crop vùng label
Resize (640×640 hoặc 960×960)
Enhancement: tăng contrast, sharpen, adjust brightness
Deskew / perspective correction nếu label bị nghiêng

Stage 3: Inspection trên vùng Label đã crop
Code Detection: code_model.pt → detect 1D & 2D
Defect Detection: defect_model.pt → single class defect
Code Decoding: pyzbar (crop trước → decode sau)

Stage 4: Decision & Output
Tổng hợp kết quả
Tính confidence tổng
Ra quyết định OK / NG
Generate JSON + ảnh visualize


3. Models & Dataset
3.1. Ba models chính

Trên datasets của weights model đã train được hiện tại gồm:
Label Detection: 6 class: PCLabel, SanDisk, WesternDigitalType1, WesternDigitalType2, WesternDigitalType3, WesternDigitalType4
Code Detection: 2 class: Code1D và Code2D
Defect Detection: 1 class: DefectNG

3.2. Training Guidelines

Sử dụng Ultralytics YOLOv11
Augmentation mạnh (đặc biệt cho defect và code méo, nhòe, nghiêng)
Train riêng từng model
Export sang ONNX → TensorRT (FP16/INT8)

4. Công nghệ & Thư viện
Core:

Python 3.10+
Ultralytics (YOLO)
OpenCV
pyzbar + zbar
Pillow
NumPy

UI:

Streamlit (demo & fine-tune UI)

Edge:

TensorRT (Jetson)
ONNX Runtime / NCNN (dự phòng)

Khác:

logging, pathlib, dataclasses, pydantic (JSON schema)

5. Yêu cầu Web UI (Streamlit)
Chức năng bắt buộc:

Upload single / multiple ảnh
Run Inference
Hiển thị:
Ảnh gốc
Vùng Label được crop
Bounding boxes + label + confidence
Kết quả decode code
Defect status (OK/NG)
JSON output

Nút “Save Result”
Tab “Fine-tune Model” (upload 1–10 ảnh + label → train nhanh → cập nhật model)

Phong cách: Sạch sẽ, hiện đại, màu sắc công nghiệp (xanh dương + trắng + cam warning)
6. Checklist Triển khai Chi tiết
Phase 0: Model Preparation

Hoàn tất annotation & merge dataset
Train 3 models
Đánh giá mAP, Precision, Recall, Confusion Matrix
Test trên dữ liệu thực tế mới
Export models

Phase 1: Core Pipeline

Tạo class LabelBarcodeInspector
Implement full cascaded pipeline
Viết functions preprocess (crop, enhance, deskew)
Viết visualization module
JSON output schema
Logging system
Unit tests cho từng stage

Phase 2: Web Demo

Xây dựng Streamlit app (app.py)
Inference realtime
Fine-tune prototype (sử dụng LoRA hoặc train lại ít epoch nếu có thể)
Beautiful visualization

Phase 3: Optimization

Quantization & TensorRT conversion
Benchmark FPS trên PC và Jetson AGX Orin
Memory usage optimization
Camera integration (OpenCV + GStreamer)

Phase 4: Production Readiness

Đóng gói ứng dụng (requirements.txt, Dockerfile)
Auto-start trên Jetson (systemd)
Remote model update mechanism
Backup & restore model
User Guide / Manual

Phase 5: Testing & Validation

Test set thực tế đa dạng
Stress test (ánh sáng, góc quay, tốc độ, blur, nhòe…)
Active Learning: thu thập ảnh NG → retrain
So sánh với thiết bị commercial (nếu có)

7. Coding Standards & Best Practices

Code sạch, comment tiếng Việt + English
Sử dụng type hints
Modular & class-based
Error handling đầy đủ
Config qua file YAML
Logging rõ ràng (INFO, WARNING, ERROR)
Không hardcode path

8. System Prompt Chuẩn cho AI Coding Assistants
textBạn là Senior Edge AI & Full-stack Vision Engineer.

Dự án: Edge AI Label & Barcode Inspection System

Yêu cầu nghiêm ngặt:
- Phải tuân thủ kiến trúc Cascaded: Label Detection → Crop & Preprocess → Code + Defect Inspection → Decoding.
- Ưu tiên hiệu suất cao trên Jetson AGX Orin 32GB (TensorRT).
- Code phải sạch, modular, dễ maintain, comment rõ ràng.
- Demo chính sử dụng Streamlit, đẹp mắt và trực quan.
- Hỗ trợ fine-tuning nhanh qua UI.
- Luôn ưu tiên tính thực tế, đơn giản và dễ bàn giao cho người dùng cuối.

Các model: label_model.pt, code_model.pt, defect_model.pt
Thư viện chính: Ultralytics, OpenCV, pyzbar, Streamlit.

Mọi code bạn sinh ra phải chạy được ngay và tuân thủ pipeline trên.
9. Tiêu chí Thành công

Pipeline chạy ổn định, tốc độ ≥ 20 FPS trên Jetson
Accuracy đạt yêu cầu thực tế (≥ 95% tổng thể)
UI demo đẹp, dễ sử dụng, thể hiện rõ giá trị
Có thể fine-tune nhanh trên dữ liệu mới
Code sạch, dễ handover
Hệ thống deploy được trên Jetson AGX Orin và chạy “một phát ăn ngay”

10. Rủi ro & Giải pháp

Model chưa generalize tốt → Thu thập thêm data thực tế + augmentation
Tốc độ chậm trên Edge → Quantization mạnh + model pruning
Defect khó detect → Fine-tune thêm hoặc kết hợp rule-based
Label bị nghiêng nhiều → Tăng deskew mạnh